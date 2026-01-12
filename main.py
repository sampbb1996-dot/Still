import time
import math
import traceback

# ---------------- CONFIG ----------------

POLL_SECONDS = 180
DECAY = 0.03
STEP = 0.12

# ---------------- STATE ----------------

state = {
    "px": 0.0,
    "ref": 0.0,
    "score": 0.0,
    "shadow_pos": 0,
    "shadow_pnl": 0.0,
}

# ---------------- HELPERS ----------------

def ema(prev, x, alpha=0.1):
    return x if prev == 0 else (1 - alpha) * prev + alpha * x

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

# ---------------- DATA SOURCES (STUBS) ----------------
# Replace these later with CoinSpot / exchange calls

def get_price():
    # TEMP placeholder so the bot runs
    # Replace with real price fetch
    return 100.0 + math.sin(time.time() / 60)

def get_balance():
    # TEMP placeholder
    return 37.38

# ---------------- CORE LOOP ----------------

def run_bot():
    print("Bot started, entering main loop", flush=True)

    while True:
        try:
            px = get_price()

            # update reference
            state["ref"] = ema(state["ref"], px)

            # compute error
            if state["ref"] > 0:
                error = (state["ref"] - px) / state["ref"]
            else:
                error = 0.0

            # update score
            state["score"] = (1 - DECAY) * state["score"] + STEP * error
            state["score"] = max(-1.0, min(1.0, state["score"]))

            # shadow exposure
            new_shadow = sign(state["score"])
            state["shadow_pos"] = new_shadow

            if state["px"] > 0:
                pnl_delta = state["shadow_pos"] * (px - state["px"]) / state["px"]
                state["shadow_pnl"] += pnl_delta

            state["px"] = px

            print(
                f"px={px:.2f} ref={state['ref']:.2f} "
                f"score={state['score']:+.3f} "
                f"shadow_pnl={state['shadow_pnl']:+.4f}",
                flush=True
            )

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print("LOOP ERROR:", e, flush=True)
            traceback.print_exc()
            time.sleep(5)

# ---------------- ENTRYPOINT ----------------

if __name__ == "__main__":
    try:
        run_bot()
    except Exception as e:
        print("FATAL ERROR:", e, flush=True)
        traceback.print_exc()
        # keep Railway alive no matter what
        while True:
            time.sleep(60)
