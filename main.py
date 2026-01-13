#!/usr/bin/env python3
"""
mispricing.py — Suggestive, error-based mispricing engine (non-trading).

Core guarantees:
- No privileged default-to-0: "inaction" is an action like any other, never a baked-in baseline.
- Suggestive (not coercive): no time decay, no forced exploration. Action happens only if it dominates.
- Error-based: decisions are driven by expected reduction in error bounds, not arbitrary scores.
- Governed openness: it is valid to do nothing when nothing dominates.

This file expects "anchors" to be provided (see anchor.py) and uses adapter hooks for fetching listings.
Adapters are intentionally minimal placeholders — wire in Gumtree/Facebook/etc however you already do.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class Listing:
    platform: str
    anchor_id: str
    listing_id: str
    title: str
    price: float
    url: str
    suburb: Optional[str] = None
    posted_ts: Optional[int] = None
    raw: Optional[dict] = None


@dataclass(frozen=True)
class Observation:
    """
    An observation is any piece of evidence about a listing that can tighten bounds.
    Example: comps found, known MSRP, sold history, manual label, etc.
    """
    listing_id: str
    kind: str
    payload: Dict[str, Any]
    ts: int


@dataclass(frozen=True)
class ErrorBounds:
    """
    Error bounds for a quantity we care about: e.g., "true value" of item.
    We keep (low, high) and treat width as uncertainty.
    """
    low: float
    high: float

    @property
    def width(self) -> float:
        return max(0.0, self.high - self.low)


@dataclass(frozen=True)
class Candidate:
    """
    A candidate listing + current belief about true value bounds.
    """
    listing: Listing
    value_bounds: ErrorBounds
    evidence_count: int
    last_updated_ts: int


@dataclass(frozen=True)
class Action:
    """
    An action proposal.
    We avoid a single scalar "score"; instead we carry structured info used for dominance checks.
    """
    name: str  # "notify", "fetch_comps", "skip"
    listing_id: Optional[str]
    anchor_id: str
    rationale: str
    expected_error_reduction: float  # width reduction expected (>=0)
    confidence: float  # 0..1 (model/heuristic confidence)
    cost: float  # abstract "effort" units
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ----------------------------
# Persistence
# ----------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS listings (
  listing_id TEXT PRIMARY KEY,
  platform TEXT NOT NULL,
  anchor_id TEXT NOT NULL,
  title TEXT NOT NULL,
  price REAL NOT NULL,
  url TEXT NOT NULL,
  suburb TEXT,
  posted_ts INTEGER,
  raw_json TEXT,
  first_seen_ts INTEGER NOT NULL,
  last_seen_ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
  listing_id TEXT PRIMARY KEY,
  value_low REAL NOT NULL,
  value_high REAL NOT NULL,
  evidence_count INTEGER NOT NULL,
  last_updated_ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  listing_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  anchor_id TEXT NOT NULL,
  action_name TEXT NOT NULL,
  listing_id TEXT,
  expected_error_reduction REAL NOT NULL,
  confidence REAL NOT NULL,
  cost REAL NOT NULL,
  rationale TEXT NOT NULL,
  metadata_json TEXT
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def upsert_listings(self, listings: Iterable[Listing]) -> None:
        now = int(time.time())
        cur = self.conn.cursor()
        for L in listings:
            cur.execute(
                """
                INSERT INTO listings(listing_id, platform, anchor_id, title, price, url, suburb, posted_ts, raw_json, first_seen_ts, last_seen_ts)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(listing_id) DO UPDATE SET
                  platform=excluded.platform,
                  anchor_id=excluded.anchor_id,
                  title=excluded.title,
                  price=excluded.price,
                  url=excluded.url,
                  suburb=excluded.suburb,
                  posted_ts=excluded.posted_ts,
                  raw_json=excluded.raw_json,
                  last_seen_ts=excluded.last_seen_ts
                """,
                (
                    L.listing_id,
                    L.platform,
                    L.anchor_id,
                    L.title,
                    float(L.price),
                    L.url,
                    L.suburb,
                    L.posted_ts,
                    json.dumps(L.raw or {}),
                    now,
                    now,
                ),
            )
        self.conn.commit()

    def get_candidate(self, listing_id: str) -> Optional[Candidate]:
        cur = self.conn.cursor()
        row = cur.execute(
            """
            SELECT l.platform,l.anchor_id,l.listing_id,l.title,l.price,l.url,l.suburb,l.posted_ts,l.raw_json,
                   c.value_low,c.value_high,c.evidence_count,c.last_updated_ts
            FROM candidates c
            JOIN listings l ON l.listing_id=c.listing_id
            WHERE c.listing_id=?
            """,
            (listing_id,),
        ).fetchone()
        if not row:
            return None
        platform, anchor_id, lid, title, price, url, suburb, posted_ts, raw_json, low, high, evc, upd = row
        listing = Listing(
            platform=platform,
            anchor_id=anchor_id,
            listing_id=lid,
            title=title,
            price=float(price),
            url=url,
            suburb=suburb,
            posted_ts=posted_ts,
            raw=json.loads(raw_json or "{}"),
        )
        return Candidate(
            listing=listing,
            value_bounds=ErrorBounds(float(low), float(high)),
            evidence_count=int(evc),
            last_updated_ts=int(upd),
        )

    def upsert_candidate(self, listing_id: str, bounds: ErrorBounds, evidence_count: int) -> None:
        now = int(time.time())
        self.conn.execute(
            """
            INSERT INTO candidates(listing_id,value_low,value_high,evidence_count,last_updated_ts)
            VALUES(?,?,?,?,?)
            ON CONFLICT(listing_id) DO UPDATE SET
              value_low=excluded.value_low,
              value_high=excluded.value_high,
              evidence_count=excluded.evidence_count,
              last_updated_ts=excluded.last_updated_ts
            """,
            (listing_id, bounds.low, bounds.high, int(evidence_count), now),
        )
        self.conn.commit()

    def add_observation(self, obs: Observation) -> None:
        self.conn.execute(
            """
            INSERT INTO observations(listing_id,kind,payload_json,ts)
            VALUES(?,?,?,?)
            """,
            (obs.listing_id, obs.kind, json.dumps(obs.payload), int(obs.ts)),
        )
        self.conn.commit()

    def record_decision(self, anchor_id: str, action: Action) -> None:
        now = int(time.time())
        self.conn.execute(
            """
            INSERT INTO decisions(ts,anchor_id,action_name,listing_id,expected_error_reduction,confidence,cost,rationale,metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                now,
                anchor_id,
                action.name,
                action.listing_id,
                float(action.expected_error_reduction),
                float(action.confidence),
                float(action.cost),
                action.rationale,
                json.dumps(action.metadata or {}),
            ),
        )
        self.conn.commit()

    def list_recent_candidates(self, anchor_id: str, limit: int = 200) -> List[Candidate]:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT l.platform,l.anchor_id,l.listing_id,l.title,l.price,l.url,l.suburb,l.posted_ts,l.raw_json,
                   c.value_low,c.value_high,c.evidence_count,c.last_updated_ts
            FROM candidates c
            JOIN listings l ON l.listing_id=c.listing_id
            WHERE l.anchor_id=?
            ORDER BY c.last_updated_ts DESC
            LIMIT ?
            """,
            (anchor_id, int(limit)),
        ).fetchall()

        out: List[Candidate] = []
        for row in rows:
            platform, a_id, lid, title, price, url, suburb, posted_ts, raw_json, low, high, evc, upd = row
            listing = Listing(
                platform=platform,
                anchor_id=a_id,
                listing_id=lid,
                title=title,
                price=float(price),
                url=url,
                suburb=suburb,
                posted_ts=posted_ts,
                raw=json.loads(raw_json or "{}"),
            )
            out.append(
                Candidate(
                    listing=listing,
                    value_bounds=ErrorBounds(float(low), float(high)),
                    evidence_count=int(evc),
                    last_updated_ts=int(upd),
                )
            )
        return out


# ----------------------------
# Error model (minimal, non-arbitrary)
# ----------------------------

def initial_bounds(price: float, prior_width_ratio: float) -> ErrorBounds:
    """
    Minimal prior: value in [price*(1-r), price*(1+r)].
    This is a prior assumption you can make explicit in config.
    """
    r = max(0.01, float(prior_width_ratio))
    low = max(0.0, price * (1.0 - r))
    high = price * (1.0 + r)
    return ErrorBounds(low, high)


def apply_observation(bounds: ErrorBounds, obs: Observation) -> ErrorBounds:
    """
    Tighten bounds using observations. This is deliberately conservative:
    - If observation provides a tighter interval, intersect.
    - Otherwise keep bounds.
    Observation payload formats supported:
      - {"value_low": x, "value_high": y}
      - {"point_estimate": x, "radius": r}
    """
    b = bounds
    p = obs.payload or {}

    if "value_low" in p and "value_high" in p:
        lo = float(p["value_low"])
        hi = float(p["value_high"])
        # intersection (conservative)
        return ErrorBounds(max(b.low, lo), min(b.high, hi))

    if "point_estimate" in p and "radius" in p:
        x = float(p["point_estimate"])
        r = max(0.0, float(p["radius"]))
        lo = max(0.0, x - r)
        hi = x + r
        return ErrorBounds(max(b.low, lo), min(b.high, hi))

    return b


def implied_mispricing(bounds: ErrorBounds, price: float) -> Tuple[float, float]:
    """
    Returns (best_case_margin, worst_case_margin) where margin = (value - price)/price.
    If bounds straddle price, worst_case may be negative.
    """
    if price <= 0:
        return (0.0, 0.0)
    best = (bounds.high - price) / price
    worst = (bounds.low - price) / price
    return (best, worst)


# ----------------------------
# Suggestive dominance logic
# ----------------------------

def dominates(a: Action, b: Action, *, min_conf: float, min_gain: float) -> bool:
    """
    a dominates b if:
      - a meets confidence gate
      - expected error reduction exceeds b by a meaningful margin
      - and cost is not strictly worse without compensation

    This is not a scalar score. It's a partial order.
    """
    if a.confidence < min_conf:
        return False

    # meaningful improvement condition
    if (a.expected_error_reduction - b.expected_error_reduction) < min_gain:
        return False

    # cost condition: allow higher cost only if error reduction compensates
    # (still non-arbitrary: just a gate, not a reward optimization)
    if a.cost > b.cost and (a.expected_error_reduction - b.expected_error_reduction) < (min_gain * 2.0):
        return False

    return True


def choose_action(proposals: List[Action], *, min_conf: float, min_gain: float) -> Action:
    """
    Suggestive chooser:
    - Adds an explicit "skip" proposal (inaction) so it's in the same comparison set.
    - Selects an action only if it is non-dominated AND dominates skip.
    - If none dominates skip, returns skip (governed openness).

    This prevents "default to 0" while also avoiding coercion.
    """
    assert proposals, "need proposals"
    # Ensure a skip option exists (never privileged; just present).
    if not any(p.name == "skip" for p in proposals):
        proposals = proposals + [
            Action(
                name="skip",
                listing_id=None,
                anchor_id=proposals[0].anchor_id,
                rationale="No action dominates under current error + confidence constraints.",
                expected_error_reduction=0.0,
                confidence=1.0,
                cost=0.0,
                metadata={},
            )
        ]

    # identify skip (there may be multiple; take minimal cost)
    skip = min((p for p in proposals if p.name == "skip"), key=lambda x: x.cost)

    # Find any action that dominates skip and is non-dominated by others
    best: Optional[Action] = None
    for a in proposals:
        if a.name == "skip":
            continue
        if not dominates(a, skip, min_conf=min_conf, min_gain=min_gain):
            continue
        # non-dominated check
        dominated_by_other = False
        for other in proposals:
            if other is a:
                continue
            if dominates(other, a, min_conf=min_conf, min_gain=min_gain):
                dominated_by_other = True
                break
        if dominated_by_other:
            continue
        # tie-break: prefer higher reduction, then lower cost (still not a single score)
        if best is None:
            best = a
        else:
            if (a.expected_error_reduction > best.expected_error_reduction) or (
                math.isclose(a.expected_error_reduction, best.expected_error_reduction) and a.cost < best.cost
            ):
                best = a

    return best if best is not None else skip


# ----------------------------
# Proposal generation
# ----------------------------

def propose_for_candidate(
    cand: Candidate,
    *,
    notify_margin_worst: float,
    fetch_if_width_over: float,
    min_conf_notify: float,
    min_conf_fetch: float,
) -> List[Action]:
    """
    Generate actions from current error state, without coercion.
    """
    L = cand.listing
    b = cand.value_bounds
    width = b.width

    best_margin, worst_margin = implied_mispricing(b, L.price)
    proposals: List[Action] = []

    # Proposal: fetch more evidence if uncertainty is still large
    # Expected reduction = some conservative fraction of current width (not arbitrary "reward", just an estimate).
    if width > fetch_if_width_over:
        proposals.append(
            Action(
                name="fetch_comps",
                listing_id=L.listing_id,
                anchor_id=L.anchor_id,
                rationale=f"Uncertainty width {width:.2f} exceeds threshold; more evidence can tighten bounds.",
                expected_error_reduction=min(width * 0.35, width),  # conservative expected shrink
                confidence=min_conf_fetch,
                cost=1.0,
                metadata={"width": width, "best_margin": best_margin, "worst_margin": worst_margin},
            )
        )

    # Proposal: notify if even worst-case margin is good enough (robust mispricing)
    if worst_margin >= notify_margin_worst:
        proposals.append(
            Action(
                name="notify",
                listing_id=L.listing_id,
                anchor_id=L.anchor_id,
                rationale=f"Worst-case margin {worst_margin:.2%} clears robust threshold.",
                expected_error_reduction=0.0,  # notify doesn't reduce uncertainty; it acts on it
                confidence=min_conf_notify,
                cost=0.2,
                metadata={"best_margin": best_margin, "worst_margin": worst_margin, "width": width},
            )
        )

    # Always include a local skip option (not privileged, just present)
    proposals.append(
        Action(
            name="skip",
            listing_id=L.listing_id,
            anchor_id=L.anchor_id,
            rationale="No dominating action justified for this candidate right now.",
            expected_error_reduction=0.0,
            confidence=1.0,
            cost=0.0,
            metadata={"best_margin": best_margin, "worst_margin": worst_margin, "width": width},
        )
    )

    return proposals


# ----------------------------
# Adapters (placeholders)
# ----------------------------

def fetch_listings_for_anchor(anchor: Dict[str, Any]) -> List[Listing]:
    """
    Replace this with your real scraper / API logic.
    Must return deterministic listing_id per platform.
    """
    # Placeholder: no-op.
    return []


def fetch_comps_for_listing(listing: Listing) -> Optional[Observation]:
    """
    Replace with your comps fetch logic.
    Return an Observation that tightens value bounds.
    """
    return None


def notify(listing: Listing, metadata: Dict[str, Any]) -> None:
    """
    Replace with Telegram/console/email notification.
    """
    logging.info("NOTIFY: %s | $%.2f | %s | %s", listing.title, listing.price, listing.url, json.dumps(metadata))


# ----------------------------
# Main loop
# ----------------------------

def run_once(
    store: Store,
    anchors: List[Dict[str, Any]],
    *,
    prior_width_ratio: float,
    fetch_if_width_over: float,
    notify_margin_worst: float,
    min_conf: float,
    min_gain: float,
    min_conf_fetch: float,
    min_conf_notify: float,
) -> None:
    now = int(time.time())

    # 1) Fetch listings per anchor and upsert
    for a in anchors:
        anchor_id = a["id"]
        listings = fetch_listings_for_anchor(a)
        if listings:
            store.upsert_listings(listings)

        # 2) Create/update candidates with an explicit prior if new
        for L in listings:
            existing = store.get_candidate(L.listing_id)
            if existing is None:
                b = initial_bounds(L.price, prior_width_ratio=prior_width_ratio)
                store.upsert_candidate(L.listing_id, b, evidence_count=0)

    # 3) For each anchor, evaluate recent candidates and decide suggestively
    for a in anchors:
        anchor_id = a["id"]
        candidates = store.list_recent_candidates(anchor_id, limit=200)

        # proposals pooled across candidates (suggestive field at anchor level)
        pooled: List[Action] = []
        listing_by_id: Dict[str, Listing] = {}

        for c in candidates:
            listing_by_id[c.listing.listing_id] = c.listing
            pooled.extend(
                propose_for_candidate(
                    c,
                    notify_margin_worst=notify_margin_worst,
                    fetch_if_width_over=fetch_if_width_over,
                    min_conf_notify=min_conf_notify,
                    min_conf_fetch=min_conf_fetch,
                )
            )

        if not pooled:
            # Still record that we did not have proposals (no observed opportunities)
            store.record_decision(
                anchor_id,
                Action(
                    name="skip",
                    listing_id=None,
                    anchor_id=anchor_id,
                    rationale="No candidates/proposals observed for this anchor.",
                    expected_error_reduction=0.0,
                    confidence=1.0,
                    cost=0.0,
                    metadata={"ts": now},
                ),
            )
            continue

        chosen = choose_action(pooled, min_conf=min_conf, min_gain=min_gain)
        store.record_decision(anchor_id, chosen)

        # 4) Execute chosen action (if any)
        if chosen.name == "skip":
            continue

        if chosen.listing_id is None:
            continue

        L = listing_by_id.get(chosen.listing_id)
        if L is None:
            continue

        if chosen.name == "fetch_comps":
            obs = fetch_comps_for_listing(L)
            if obs:
                store.add_observation(obs)
                # update candidate bounds
                cand = store.get_candidate(L.listing_id)
                if cand:
                    new_b = apply_observation(cand.value_bounds, obs)
                    store.upsert_candidate(L.listing_id, new_b, evidence_count=cand.evidence_count + 1)

        elif chosen.name == "notify":
            notify(L, chosen.metadata)

        else:
            logging.info("Unknown action: %s", chosen.name)


def load_anchors(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("anchors json must be a list")
    for a in data:
        if "id" not in a:
            raise ValueError("each anchor must have an 'id'")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mispricing.db")
    ap.add_argument("--anchors", default="anchors.json")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=60.0)

    # Explicit, non-hidden assumptions
    ap.add_argument("--prior-width-ratio", type=float, default=0.60, help="prior bounds are price*(1±r)")
    ap.add_argument("--fetch-if-width-over", type=float, default=80.0, help="trigger evidence fetch when bounds width > X")
    ap.add_argument("--notify-margin-worst", type=float, default=0.25, help="notify when worst-case margin >= this")

    # Dominance gates (suggestive)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--min-gain", type=float, default=5.0, help="meaningful error reduction delta for dominance")
    ap.add_argument("--min-conf-fetch", type=float, default=0.65)
    ap.add_argument("--min-conf-notify", type=float, default=0.70)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.exists(args.anchors):
        raise SystemExit(f"Missing anchors file: {args.anchors}")

    store = Store(args.db)
    anchors = load_anchors(args.anchors)

    while True:
        run_once(
            store,
            anchors,
            prior_width_ratio=args.prior_width_ratio,
            fetch_if_width_over=args.fetch_if_width_over,
            notify_margin_worst=args.notify_margin_worst,
            min_conf=args.min_conf,
            min_gain=args.min_gain,
            min_conf_fetch=args.min_conf_fetch,
            min_conf_notify=args.min_conf_notify,
        )
        if not args.loop:
            break
        time.sleep(max(1.0, args.sleep))


if __name__ == "__main__":
    main()
