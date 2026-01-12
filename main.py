import time
import math
import threading
import traceback
import os
from flask import Flask

# ===============================
# Flask app (MUST be main thread)
# ===============================

app = Flask(__name__)

@app.route("/")
def home():
    return "OK"

# ===============================
# CONFIG
# ===============================

POLL_SECONDS = 180
DECAY = 0.03
STEP = 0.12

# ===============================
# STATE
# ===============================

state = {
    "px": 0.0,
    "ref": 0.0,
    "score": 0.0,
    "shadow_pos": 0,
    "shadow_pnl": 0.0,
}

# ===============================
# HELPERS
# ===============================

def ema(prev, x, alpha=0.1):
    return x if prev == 0 else (1 - alpha) * prev + alpha * x

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0

# ===============================
# DATA SOURCE (stub)
# ===============================

def get_price():
    return 100.0 + math.sin(time.time() / 60)

# ===============================
# BOT LOOP (background thread)
# ===============================

def run_bot():
    print("Bot started, entering main loop", flush=True)

    while True:
        try:
            px = get_price()
            state["ref"] = ema(state["ref"], px)

            error = (state["ref"] - px) / state["ref"] if state["ref"] else 0.0
            state["score"] = (1 - DECAY) * state["score"] + STEP * error
            state["score"] = max(-1.0, min(1.0, state["score"]))

            state["shadow_pos"] = sign(state["score"])

            if state["px"]:
                state["shadow_pnl"] += (
                    state["shadow_pos"] * (px - state["px"]) / state["px"]
                )

            state["px"] = px

            print(
                f"px={px:.2f} ref={state['ref']:.2f} "
                f"score={state['score']:+.3f} "
                f"shadow_pnl={state['shadow_pnl']:+.4f}",
                flush=True
            )

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print("BOT ERROR:", e, flush=True)
            traceback.print_exc()
            time.sleep(5)

# ===============================
# ENTRYPOINT
# ===============================

if __name__ == "__main__":
    # Start bot in background
    threading.Thread(target=run_bot, daemon=True).start()

    # Start Flask in main thread (Railway requirement)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
