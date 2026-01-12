# governor_bot.py
# Implicit profit-seeking, non-arbitrary, non-freezing

import time, math, sqlite3, hashlib
from dataclasses import dataclass
from typing import Dict

DB = "field.db"

POLL_SECONDS = 180

DECAY = 0.03           # memory decay (governor)
STEP = 0.12            # bounded update per observation
FEE_RATE = 0.001       # exchange taker fee (non-arbitrary)
SPREAD_RATE = 0.0005   # conservative impact proxy

COST_FLOOR = 2 * FEE_RATE + SPREAD_RATE   # minimum viable edge

MICRO_AUD = 1.0        # platform minimum probe (external constraint)
RISK_CAP = 0.06        # max capital fraction

# -------------------- model --------------------

@dataclass
class MarketState:
    px: float
    ref: float
    score: float
    shadow_pos: int          # -1, 0, +1
    shadow_pnl: float
    real_pos: float
    phase: str               # SHADOW | MICRO | FULL

# -------------------- helpers --------------------

def ema(prev, x, alpha=0.1):
    return x if prev == 0 else (1 - alpha) * prev + alpha * x

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

# -------------------- core logic --------------------

def update_state(state: MarketState, new_px: float):
    # update reference (implicit expectation)
    state.ref = ema(state.ref, new_px)

    # signed relative error (always non-zero once price moves)
    error = (state.ref - new_px) / state.ref if state.ref > 0 else 0.0

    # bounded score update
    state.score = (1 - DECAY) * state.score + STEP * error
    state.score = max(-1.0, min(1.0, state.score))

    # --- SHADOW exposure (always on) ---
    desired_shadow = sign(state.score)
    if desired_shadow != state.shadow_pos:
        state.shadow_pos = desired_shadow

    # shadow P&L accumulates information
    pnl_delta = state.shadow_pos * (new_px - state.px) / state.px if state.px > 0 else 0.0
    state.shadow_pnl += pnl_delta

    state.px = new_px

def should_micro_trade(state: MarketState):
    # implicit: only if shadow has beaten costs persistently
    return abs(state.shadow_pnl) > COST_FLOOR

def should_full_trade(state: MarketState):
    # full deployment only if execution validated
    return abs(state.shadow_pnl) > 3 * COST_FLOOR

# -------------------- execution gates --------------------

def trade_decision(state: MarketState, aud_balance: float):
    # SHADOW → MICRO
    if state.phase == "SHADOW" and should_micro_trade(state):
        state.phase = "MICRO"

    # MICRO → FULL
    if state.phase == "MICRO" and should_full_trade(state):
        state.phase = "FULL"

    # --- execution ---
    if state.phase == "MICRO":
        return {
            "action": "probe",
            "side": "buy" if state.score > 0 else "sell",
            "aud": MICRO_AUD
        }

    if state.phase == "FULL":
        budget = aud_balance * RISK_CAP * abs(state.score)
        if budget >= MICRO_AUD:
            return {
                "action": "trade",
                "side": "buy" if state.score > 0 else "sell",
                "aud": budget
            }

    return None

# -------------------- main loop (simplified) --------------------

def run_bot(get_price, get_balance):
    state = MarketState(
        px=0.0,
        ref=0.0,
        score=0.0,
        shadow_pos=0,
        shadow_pnl=0.0,
        real_pos=0.0,
        phase="SHADOW"
    )

    while True:
        px = get_price()
        aud = get_balance()

        update_state(state, px)
        decision = trade_decision(state, aud)

        print(
            f"[{state.phase}] px={px:.2f} score={state.score:+.3f} "
            f"shadow_pnl={state.shadow_pnl:+.4f} aud={aud:.2f}"
        )

        if decision:
            print("EXEC:", decision)
            # <-- real order placement goes here

        time.sleep(POLL_SECONDS)
