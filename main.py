import time
import math
import os
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# ===============================
# Railway keepalive (MANDATORY)
# ===============================

def keepalive_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            return  # silence

    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()

# ===============================
# CONFIG (non-arbitrary)
# ===============================

POLL_SECONDS = 180          # scan cadence
DECAY = 0.03                # governor memory decay
STEP = 0.12                 # bounded update strength

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
    if prev == 0:
        return x
    return (1 - alpha) * prev + alpha * x

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

# ===============================
# DATA SOURCES
# (replace later with CoinSpot)
# ===============================

def get_price():
    # Placeholder deterministic movement
    return 100.0 + math.sin(time.time() / 60)

# ===============================
# CORE BOT LOOP
# ===============================

def run_bot():
    print("Bot started, entering main loop", flush=True)

    while True:
        try:
            px = get_price()

            # Update reference (implicit expectation)
            state["ref"] = ema(state["ref"], px)

            # Relative signed error
            if state["ref"] > 0:
                error = (state["ref"] - px) / state["ref"]
            else:
                error = 0.0

            # Governor score update
            state["score"] = (1 - DECAY) * state["score"] + STEP * error
            state["score"] = max(-1.0, min(1.0, state["score"]))

            # Shadow exposure (always on)
            state["shadow_pos"] = sign(state["score"])

            # Shadow P&L (information pressure)
            if state["px"] > 0:
                pnl_delta = (
                    state["shadow_pos"]
                    * (px - state["px"])
                    / state["px"]
                )
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

# ===============================
# ENTRYPOINT (Railway-safe)
# ===============================

if __name__ == "__main__":
    # Start keepalive server so Railway keeps container alive
    threading.Thread(target=keepalive_server, daemon=True).start()

    try:
        run_bot()
    except Exception as e:
        print("FATAL ERROR:", e, flush=True)
        traceback.print_exc()
        # Never exit
        while True:
            time.sleep(60)
