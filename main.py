import express from "express";

const app = express();
const PORT = process.env.PORT || 5000;

/* -------------------- BOT LOGIC -------------------- */

// replace this with your existing bot logic
async function botTick() {
  // scan / evaluate / maybe trade
  console.log("Bot tick");
}

// start the loop ONCE and never exit
function startBotLoop() {
  console.log("Bot loop starting...");

  // run once immediately
  botTick().catch(err => {
    console.error("botTick error:", err);
  });

  // then repeat forever
  setInterval(() => {
    botTick().catch(err => {
      console.error("botTick error:", err);
    });
  }, 60_000); // keep your existing interval
}

/* -------------------- SERVER -------------------- */

app.get("/", (_req, res) => {
  res.status(200).send("ok");
});

app.get("/health", (_req, res) => {
  res.status(200).json({ ok: true, ts: Date.now() });
});

// THIS is what keeps the process alive
app.listen(PORT, "0.0.0.0", () => {
  console.log(`[express] serving on port ${PORT}`);
  startBotLoop();
});

/* -------------------- SAFETY -------------------- */

// never let crashes exit the process
process.on("unhandledRejection", err => {
  console.error("unhandledRejection:", err);
});

process.on("uncaughtException", err => {
  console.error("uncaughtException:", err);
});
