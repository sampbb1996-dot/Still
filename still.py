#!/usr/bin/env python3
"""
ECG Bot (Replit-ready) — CoinSpot v2
-----------------------------------
A minimal, survival-first trading bot that implements:

- Error-ceiling governor (risk/position constrained by an evolving error budget)
- "False signals are positive" (losses tighten constraints and increase caution; they are treated as information)
- Auto-reinvest profits (INTERNAL only): position envelope expands slowly as equity grows
- Multi-market support via CoinSpot pubapi/v2/latest (default AUD pairs only)
- SQLite persistence so restarts keep state (nonce, positions, price history, budgets)

IMPORTANT SAFETY DEFAULT:
- DRY_RUN=1 by default. It will NOT place trades unless you explicitly set DRY_RUN=0.

CoinSpot API docs (v2):
- Public latest: https://www.coinspot.com.au/pubapi/v2/latest
- Read-only balances: https://www.coinspot.com.au/api/v2/ro/my/balances
- Buy now: https://www.coinspot.com.au/api/v2/my/buy/now
- Sell now: https://www.coinspot.com.au/api/v2/my/sell/now
"""

from __future__ import annotations

import os
import time
import json
import hmac
import math
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


# =========================
# Config (env overrides)
# =========================

DB_PATH = os.getenv("DB_PATH", "ecg_field.db")

COINSPOT_KEY = os.getenv("COINSPOT_KEY", "").strip()
COINSPOT_SECRET = os.getenv("COINSPOT_SECRET", "").strip()

# Safety: default DRY_RUN on
DRY_RUN = os.getenv("DRY_RUN", "1").strip() != "0"

# Poll cadence: 3–5 min with small jitter
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "240"))  # base (4 min)

# Which coins to trade (comma-separated). Use CoinSpot coin codes (BTC, ETH, etc.)
# Default keeps it conservative.
TARGET_COINS = [c.strip().upper() for c in os.getenv("TARGET_COINS", "BTC,ETH").split(",") if c.strip()]

# Markets: keep it simple/survivable: AUD only by default.
# CoinSpot public API symbol keys are lowercase: btc, eth, btc_usdt, etc.
MARKET = os.getenv("MARKET", "AUD").strip().upper()  # "AUD" or "USDT" etc. (we default to AUD)

# Minimum trade (skip if below; do NOT force up)
MIN_TRADE_AUD = float(os.getenv("MIN_TRADE_AUD", "1.00"))

# Base trade sizing
BASE_TRADE_AUD = float(os.getenv("BASE_TRADE_AUD", "5.00"))  # starting nominal spend per buy
MAX_FRACTION_OF_AUD = float(os.getenv("MAX_FRACTION_OF_AUD", "0.06"))  # cap per trade as fraction of available AUD

# Governor / signal params (bounded, boring)
FAST_WINDOW = int(os.getenv("FAST_WINDOW", "3"))     # fast samples
SLOW_WINDOW = int(os.getenv("SLOW_WINDOW", "12"))   # slow samples
SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", "0.60"))  # action threshold
CONFIRM_STREAK = int(os.getenv("CONFIRM_STREAK", "2"))           # require consecutive confirmations

# Score/budget dynamics
SCORE_DECAY = float(os.getenv("SCORE_DECAY", "0.05"))      # decay toward 0 per cycle
SCORE_STEP = float(os.getenv("SCORE_STEP", "0.18"))        # bounded update per observation
BUDGET_DECAY = float(os.getenv("BUDGET_DECAY", "0.02"))    # error budget drifts down slowly
LOSS_BUDGET_ADD = float(os.getenv("LOSS_BUDGET_ADD", "0.25"))  # how much a loss tightens
WIN_BUDGET_SUB = float(os.getenv("WIN_BUDGET_SUB", "0.10"))    # how much a win loosens

# Auto-reinvest envelope expansion (slow and capped)
REINVEST_RATE = float(os.getenv("REINVEST_RATE", "0.25"))  # fraction of profit that expands risk envelope
MAX_FRACTION_CAP = float(os.getenv("MAX_FRACTION_CAP", "0.12"))  # absolute ceiling regardless of profit

# Network
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))


# =========================
# CoinSpot endpoints
# =========================

PUB_LATEST = "https://www.coinspot.com.au/pubapi/v2/latest"
API_ROOT = "https://www.coinspot.com.au/api/v2"
RO_ROOT = "https://www.coinspot.com.au/api/v2/ro"

EP_BALANCES = f"{RO_ROOT}/my/balances"
EP_BUY_NOW = f"{API_ROOT}/my/buy/now"
EP_SELL_NOW = f"{API_ROOT}/my/sell/now"


# =========================
# Helpers
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def now_ms() -> int:
    return int(time.time() * 1000)

def coin_key_for_market(coin: str, market: str) -> str:
    """
    CoinSpot pubapi/v2/latest uses:
    - 'btc' for BTC/AUD
    - 'btc_usdt' for BTC/USDT
    """
    coin_l = coin.lower()
    if market.upper() == "AUD":
        return coin_l
    return f"{coin_l}_{market.lower()}"

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# =========================
# Storage
# =========================

def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db() -> None:
    con = db()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS kv (
        k TEXT PRIMARY KEY,
        v TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        coin TEXT NOT NULL,
        ts INTEGER NOT NULL,
        last REAL NOT NULL,
        PRIMARY KEY (coin, ts)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS state (
        coin TEXT PRIMARY KEY,
        score REAL NOT NULL,
        confirm_pos INTEGER NOT NULL,
        confirm_neg INTEGER NOT NULL,
        error_budget REAL NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS position (
        coin TEXT PRIMARY KEY,
        qty REAL NOT NULL,
        entry REAL NOT NULL,
        last_action_ts INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ledger (
        id TEXT PRIMARY KEY,
        ts INTEGER NOT NULL,
        coin TEXT NOT NULL,
        side TEXT NOT NULL,
        qty REAL NOT NULL,
        px REAL NOT NULL,
        aud_est REAL NOT NULL,
        pnl_aud REAL NOT NULL
    );
    """)

    con.commit()
    con.close()

def kv_get(k: str, default: Optional[str] = None) -> Optional[str]:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT v FROM kv WHERE k = ?", (k,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else default

def kv_set(k: str, v: str) -> None:
    con = db()
    cur = con.cursor()
    cur.execute("INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    con.commit()
    con.close()

def get_or_init_state(coin: str) -> Tuple[float, int, int, float]:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT score, confirm_pos, confirm_neg, error_budget FROM state WHERE coin=?", (coin,))
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO state (coin, score, confirm_pos, confirm_neg, error_budget) VALUES (?, ?, ?, ?, ?)",
                    (coin, 0.0, 0, 0, 0.0))
        con.commit()
        row = (0.0, 0, 0, 0.0)
    con.close()
    return float(row[0]), int(row[1]), int(row[2]), float(row[3])

def set_state(coin: str, score: float, cpos: int, cneg: int, budget: float) -> None:
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO state (coin, score, confirm_pos, confirm_neg, error_budget)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(coin) DO UPDATE SET
            score=excluded.score,
            confirm_pos=excluded.confirm_pos,
            confirm_neg=excluded.confirm_neg,
            error_budget=excluded.error_budget
    """, (coin, score, cpos, cneg, budget))
    con.commit()
    con.close()

def get_position(coin: str) -> Tuple[float, float, int]:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT qty, entry, last_action_ts FROM position WHERE coin=?", (coin,))
    row = cur.fetchone()
    if not row:
        con.close()
        return 0.0, 0.0, 0
    con.close()
    return float(row[0]), float(row[1]), int(row[2])

def set_position(coin: str, qty: float, entry: float, ts: int) -> None:
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO position (coin, qty, entry, last_action_ts)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(coin) DO UPDATE SET
            qty=excluded.qty,
            entry=excluded.entry,
            last_action_ts=excluded.last_action_ts
    """, (coin, qty, entry, ts))
    con.commit()
    con.close()

def add_price(coin: str, ts: int, last: float) -> None:
    con = db()
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO prices (coin, ts, last) VALUES (?, ?, ?)", (coin, ts, last))
    # keep history bounded (small + boring)
    cur.execute("""
        DELETE FROM prices
        WHERE coin = ?
          AND ts NOT IN (
              SELECT ts FROM prices WHERE coin=? ORDER BY ts DESC LIMIT 200
          )
    """, (coin, coin))
    con.commit()
    con.close()

def get_last_prices(coin: str, n: int) -> List[float]:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT last FROM prices WHERE coin=? ORDER BY ts DESC LIMIT ?", (coin, n))
    rows = cur.fetchall()
    con.close()
    return [float(r[0]) for r in rows][::-1]  # oldest -> newest

def record_trade(coin: str, side: str, qty: float, px: float, aud_est: float, pnl_aud: float) -> None:
    tid = hashlib.sha256(f"{coin}:{side}:{qty}:{px}:{now_ms()}".encode()).hexdigest()
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO ledger (id, ts, coin, side, qty, px, aud_est, pnl_aud)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (tid, now_ms(), coin, side, qty, px, aud_est, pnl_aud))
    con.commit()
    con.close()


# =========================
# CoinSpot signing (v2)
# =========================

def next_nonce() -> int:
    """
    Nonce must be strictly increasing. We persist last_nonce in DB.
    """
    last = int(kv_get("last_nonce", "0") or "0")
    n = now_ms()
    if n <= last:
        n = last + 1
    kv_set("last_nonce", str(n))
    return n

def sign_payload(payload: dict) -> Tuple[Dict[str, str], bytes]:
    """
    CoinSpot v2 signing: HMAC-SHA512 over raw POST body.
    """
    if not COINSPOT_KEY or not COINSPOT_SECRET:
        raise RuntimeError("Missing COINSPOT_KEY or COINSPOT_SECRET environment variables.")

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sig = hmac.new(COINSPOT_SECRET.encode("utf-8"), body, hashlib.sha512).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "key": COINSPOT_KEY,
        "sign": sig,
    }
    return headers, body

def post_signed(url: str, payload: dict) -> dict:
    payload = dict(payload)
    payload["nonce"] = next_nonce()
    headers, body = sign_payload(payload)
    r = requests.post(url, data=body, headers=headers, timeout=HTTP_TIMEOUT)
    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response from {url}: HTTP {r.status_code} {r.text[:300]}")
    return j


# =========================
# Data fetch
# =========================

def fetch_latest_prices() -> dict:
    r = requests.get(PUB_LATEST, timeout=HTTP_TIMEOUT)
    j = r.json()
    if j.get("status") != "ok":
        raise RuntimeError(f"pubapi latest error: {j}")
    return j["prices"]

def fetch_balances() -> dict:
    j = post_signed(EP_BALANCES, {})
    if j.get("status") != "ok":
        raise RuntimeError(f"balances error: {j}")
    # balances is array of {"AUD": {...}} etc. Convert to dict.
    out: Dict[str, dict] = {}
    for obj in j.get("balances", []):
        for k, v in obj.items():
            out[k.upper()] = v
    return out


# =========================
# ECG / Error logic core
# =========================

def compute_momentum_signal(prices: List[float]) -> float:
    """
    Minimal, parameter-light signal:
    - Use fast and slow log returns
    - Combine via a bounded nonlinearity
    Returns in [-1, 1]
    """
    if len(prices) < max(FAST_WINDOW, SLOW_WINDOW) + 1:
        return 0.0

    # log returns
    rets = []
    for i in range(1, len(prices)):
        a, b = prices[i - 1], prices[i]
        if a <= 0 or b <= 0:
            rets.append(0.0)
        else:
            rets.append(math.log(b / a))

    fast = sum(rets[-FAST_WINDOW:])
    slow = sum(rets[-SLOW_WINDOW:])

    # Normalize by a tiny volatility estimate to avoid overreacting
    window = rets[-SLOW_WINDOW:]
    mean = sum(window) / max(1, len(window))
    var = sum((x - mean) ** 2 for x in window) / max(1, len(window))
    vol = math.sqrt(var) + 1e-9

    # signed strength
    fast_s = clamp(fast / (3.0 * vol), -2.0, 2.0)
    slow_s = clamp(slow / (3.0 * vol), -2.0, 2.0)

    # "False signals are positive" framing:
    # We do NOT hard-commit to agreement. Disagreement contributes to caution (handled by governor),
    # but the signal itself remains bounded and modest.
    raw = 0.65 * fast_s + 0.35 * slow_s

    # squash to [-1,1]
    return math.tanh(raw)

def update_score_and_confirms(score: float, cpos: int, cneg: int, obs: float) -> Tuple[float, int, int]:
    """
    Bounded score update + confirmation streaks.
    obs in [-1, 1].
    """
    # decay toward 0
    score = score * (1.0 - SCORE_DECAY)

    # bounded update
    score = clamp(score + SCORE_STEP * obs, -1.0, 1.0)

    # confirmation streaks
    if score >= SIGNAL_THRESHOLD:
        cpos += 1
        cneg = 0
    elif score <= -SIGNAL_THRESHOLD:
        cneg += 1
        cpos = 0
    else:
        # mid-zone resets
        cpos = 0
        cneg = 0

    return score, cpos, cneg

def governor_trade_fraction(
    available_aud: float,
    starting_equity: float,
    current_equity: float,
    error_budget: float
) -> float:
    """
    Compute max fraction of available AUD allowed for a new BUY.
    - expands slowly with profit (auto-reinvest)
    - tightens with error_budget
    Always bounded by MAX_FRACTION_CAP.
    """
    base = MAX_FRACTION_OF_AUD

    # profit expansion (slow + capped)
    profit = max(0.0, current_equity - starting_equity)
    if starting_equity > 0:
        profit_frac = profit / starting_equity
    else:
        profit_frac = 0.0

    expanded = base + REINVEST_RATE * profit_frac * base

    # error ceiling: tighten as budget grows (losses increase budget)
    # budget in [0, 1+] -> multiplier in (0, 1]
    tighten = 1.0 / (1.0 + 1.8 * clamp(error_budget, 0.0, 5.0))

    out = expanded * tighten
    out = clamp(out, 0.0, min(MAX_FRACTION_CAP, 0.25))  # hard absolute cap
    return out

def convex_aud_size(available_aud: float, frac_cap: float, score: float) -> float:
    """
    Convex sizing: small near threshold, grows nonlinearly as confidence increases.
    """
    if abs(score) < SIGNAL_THRESHOLD:
        return 0.0
    # map [threshold..1] -> [0..1]
    x = (abs(score) - SIGNAL_THRESHOLD) / max(1e-9, (1.0 - SIGNAL_THRESHOLD))
    x = clamp(x, 0.0, 1.0)

    # convex curve (square)
    w = x * x

    cap_aud = available_aud * frac_cap
    target = BASE_TRADE_AUD * (0.5 + 1.5 * w)  # between 0.5x and 2.0x base
    size = min(target, cap_aud)

    # Skip-if-below rule (do not force minimum)
    if size < MIN_TRADE_AUD:
        return 0.0
    return round(size, 2)  # AUD precision


# =========================
# Trading (Buy/Sell Now)
# =========================

def place_buy_now(coin: str, aud_amount: float) -> dict:
    payload = {
        "cointype": coin.upper(),
        "amounttype": "aud",
        "amount": f"{aud_amount:.2f}",
    }
    return post_signed(EP_BUY_NOW, payload)

def place_sell_now(coin: str, coin_amount: float) -> dict:
    payload = {
        "cointype": coin.upper(),
        "amounttype": "coin",
        "amount": f"{coin_amount:.8f}",
    }
    return post_signed(EP_SELL_NOW, payload)


# =========================
# Equity / starting point
# =========================

def compute_equity_aud(balances: Dict[str, dict]) -> float:
    """
    balances from /ro/my/balances includes audbalance per coin already.
    We'll sum audbalance.
    """
    total = 0.0
    for sym, obj in balances.items():
        ab = safe_float(obj.get("audbalance"))
        if ab is not None:
            total += ab
    return float(total)

def ensure_starting_equity(current_equity: float) -> float:
    v = kv_get("starting_equity_aud", "")
    if v:
        try:
            return float(v)
        except Exception:
            pass
    kv_set("starting_equity_aud", str(current_equity))
    return current_equity


# =========================
# Main loop
# =========================

def main() -> None:
    init_db()

    if not COINSPOT_KEY or not COINSPOT_SECRET:
        print("ERROR: Set COINSPOT_KEY and COINSPOT_SECRET in Replit Secrets (Environment).")
        return

    print("ECG Bot started.")
    print(f"DRY_RUN={1 if DRY_RUN else 0} | MARKET={MARKET} | TARGET_COINS={TARGET_COINS}")
    print(f"Poll ~{POLL_SECONDS}s | threshold={SIGNAL_THRESHOLD} | confirm={CONFIRM_STREAK}")
    print("Public latest endpoint:", PUB_LATEST)

    while True:
        cycle_ts = now_ms()

        try:
            prices_blob = fetch_latest_prices()
            balances = fetch_balances()

            current_equity = compute_equity_aud(balances)
            starting_equity = ensure_starting_equity(current_equity)

            aud_avail = safe_float(balances.get("AUD", {}).get("balance")) or 0.0

            # For each coin: update price history + score, then maybe act
            for coin in TARGET_COINS:
                key = coin_key_for_market(coin, MARKET)
                px_obj = prices_blob.get(key)
                if not px_obj:
                    print(f"[{coin}] No price key '{key}' in pubapi latest; skipping.")
                    continue

                last = safe_float(px_obj.get("last"))
                bid = safe_float(px_obj.get("bid"))
                ask = safe_float(px_obj.get("ask"))
                if last is None:
                    print(f"[{coin}] Missing last price; skipping.")
                    continue

                add_price(coin, cycle_ts, float(last))
                hist = get_last_prices(coin, max(SLOW_WINDOW, FAST_WINDOW) + 1)

                obs = compute_momentum_signal(hist)

                score, cpos, cneg, budget = get_or_init_state(coin)

                # budget decays slowly by default (stability bias)
                budget = max(0.0, budget * (1.0 - BUDGET_DECAY))

                score, cpos, cneg = update_score_and_confirms(score, cpos, cneg, obs)

                qty, entry, last_action_ts = get_position(coin)

                frac_cap = governor_trade_fraction(
                    available_aud=aud_avail,
                    starting_equity=starting_equity,
                    current_equity=current_equity,
                    error_budget=budget
                )

                # Decide actions
                want_buy = (qty <= 0.0) and (cpos >= CONFIRM_STREAK)
                want_sell = (qty > 0.0) and (cneg >= CONFIRM_STREAK)

                # Print state line (compact)
                print(
                    f"[{coin}] px={last:.6g} score={score:+.3f} obs={obs:+.3f} "
                    f"c+={cpos} c-={cneg} budget={budget:.3f} "
                    f"pos={qty:.8f}@{entry:.6g} aud={aud_avail:.2f} cap={frac_cap:.3f}"
                )

                # BUY
                if want_buy:
                    aud_size = convex_aud_size(aud_avail, frac_cap, score)
                    if aud_size <= 0:
                        # Skip if below minimum or no capacity
                        continue

                    # Execute (or dry-run)
                    if DRY_RUN:
                        print(f"  -> DRY_RUN BUY {coin} for ~{aud_size:.2f} AUD (buy/now)")
                        # pretend fill at ask/last
                        fill_px = ask or last
                        fill_qty = (aud_size / fill_px) if fill_px > 0 else 0.0
                        set_position(coin, fill_qty, float(fill_px), cycle_ts)
                        record_trade(coin, "BUY", fill_qty, float(fill_px), aud_size, 0.0)
                    else:
                        resp = place_buy_now(coin, aud_size)
                        if resp.get("status") == "ok":
                            # Response provides amount (coin) and total (aud)
                            fill_qty = safe_float(resp.get("amount")) or 0.0
                            # "rate" not returned in buy/now response; use last/ask as estimate
                            fill_px = ask or last
                            set_position(coin, float(fill_qty), float(fill_px), cycle_ts)
                            record_trade(coin, "BUY", float(fill_qty), float(fill_px), aud_size, 0.0)
                            print(f"  -> BUY OK {coin}: qty={fill_qty} est_px={fill_px} resp={resp}")
                        else:
                            print(f"  -> BUY ERROR {coin}: {resp}")

                    # reset confirms after action
                    cpos = 0
                    cneg = 0

                # SELL
                if want_sell:
                    sell_qty = qty
                    if sell_qty <= 0:
                        continue

                    if DRY_RUN:
                        fill_px = bid or last
                        pnl = (fill_px - entry) * sell_qty
                        print(f"  -> DRY_RUN SELL {coin} qty={sell_qty:.8f} est_px={fill_px:.6g} pnl≈{pnl:.4f} AUD")
                        set_position(coin, 0.0, 0.0, cycle_ts)
                        record_trade(coin, "SELL", sell_qty, float(fill_px), float(fill_px * sell_qty), float(pnl))

                        # "False signals are positive": loss tightens (budget up); win loosens (budget down)
                        if pnl < 0:
                            budget = budget + LOSS_BUDGET_ADD
                        else:
                            budget = max(0.0, budget - WIN_BUDGET_SUB)

                    else:
                        resp = place_sell_now(coin, sell_qty)
                        if resp.get("status") == "ok":
                            fill_px = safe_float(resp.get("rate")) or (bid or last)
                            total = safe_float(resp.get("total")) or (fill_px * sell_qty)
                            pnl = (fill_px - entry) * sell_qty
                            set_position(coin, 0.0, 0.0, cycle_ts)
                            record_trade(coin, "SELL", sell_qty, float(fill_px), float(total), float(pnl))
                            print(f"  -> SELL OK {coin}: qty={sell_qty} px={fill_px} total={total} pnl≈{pnl:.4f} resp={resp}")

                            if pnl < 0:
                                budget = budget + LOSS_BUDGET_ADD
                            else:
                                budget = max(0.0, budget - WIN_BUDGET_SUB)
                        else:
                            print(f"  -> SELL ERROR {coin}: {resp}")

                    # reset confirms after action
                    cpos = 0
                    cneg = 0

                # Persist updated state
                set_state(coin, score, cpos, cneg, budget)

        except Exception as e:
            print("Cycle error:", repr(e))

        # Sleep with small jitter (bounded)
        jitter = int((hashlib.sha256(str(now_ms()).encode()).digest()[0] % 21) - 10)  # [-10..+10]
        delay = max(60, POLL_SECONDS + jitter)
        time.sleep(delay)


if __name__ == "__main__":
    main()
