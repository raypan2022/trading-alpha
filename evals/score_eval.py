"""
Evaluation harness — scorer.

Reads logged verdicts and, for any past its horizon, scores it against the
stock's actual forward move:
  - Directional accuracy (BUY/SELL/HOLD vs realized return, with a HOLD band)
  - Target-price error (MAPE vs the realized price)
  - Confidence calibration (do high-confidence calls actually hit more often?)

Calibration is the point: a model that's right 60% of the time but *knows* when
it's confident is far more useful than one with the same accuracy and flat
confidence. This is the metric that tells you whether the agent's self-assessment
means anything.

Run from the project root:
    python -m evals.score_eval
"""
import os
import json
from datetime import datetime, timezone, timedelta

import yfinance as yf

RESULTS_PATH = os.getenv(
    "EVAL_RESULTS_PATH", os.path.join(os.path.dirname(__file__), "results.jsonl")
)
HOLD_BAND = 0.02   # +/-2% realized move counts as "flat" when judging a HOLD


def _load() -> list:
    if not os.path.exists(RESULTS_PATH):
        return []
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _exit_price(ticker: str, entry_dt: datetime, horizon_days: int):
    """Closing price on the first trading day at/after the horizon date."""
    start = entry_dt.date()
    end = start + timedelta(days=horizon_days + 7)
    try:
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
        if hist.empty:
            return None
        target = start + timedelta(days=horizon_days)
        hist = hist.reset_index()
        for _, row in hist.iterrows():
            if row["Date"].date() >= target:
                return float(row["Close"])
        return float(hist.iloc[-1]["Close"])  # horizon beyond available data
    except Exception:
        return None


def _directional_correct(action: str, ret: float) -> bool:
    if action == "BUY":
        return ret > HOLD_BAND
    if action == "SELL":
        return ret < -HOLD_BAND
    if action == "HOLD":
        return abs(ret) <= HOLD_BAND
    return False


def main():
    rows = _load()
    if not rows:
        print("No logged verdicts yet. Run: python -m evals.run_eval")
        return

    now = datetime.now(timezone.utc)
    scored, pending = [], 0

    for r in rows:
        entry_dt = datetime.fromisoformat(r["timestamp"])
        horizon = r.get("horizon_days", 30)
        if now < entry_dt + timedelta(days=horizon):
            pending += 1
            continue
        entry = r.get("entry_price")
        if not entry:
            continue
        exit_price = _exit_price(r["ticker"], entry_dt, horizon)
        if not exit_price:
            continue
        ret = (exit_price - entry) / entry
        scored.append({
            **r,
            "exit_price": exit_price,
            "return": ret,
            "correct": _directional_correct(r["action"], ret),
            "target_err": abs(r["target_price_30d"] - exit_price) / exit_price,
        })

    print("=" * 50)
    print("  SCORECARD")
    print("=" * 50)
    print(f"  Scored: {len(scored)}   Pending horizon: {pending}")
    if not scored:
        print("  Nothing matured yet — check back after the horizon elapses.")
        return

    acc = sum(1 for s in scored if s["correct"]) / len(scored)
    mape = sum(s["target_err"] for s in scored) / len(scored)
    print(f"  Directional accuracy: {acc:.0%}")
    print(f"  Target price MAPE:    {mape:.1%}")

    print("\n  Calibration (confidence bucket -> actual hit rate):")
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        bucket = [s for s in scored if lo <= s["confidence"] < hi]
        if bucket:
            hit = sum(1 for s in bucket if s["correct"]) / len(bucket)
            print(f"    {lo:.0%}-{min(hi,1.0):.0%}: {hit:.0%} hit rate  (n={len(bucket)})")

    print("\n  Per-verdict:")
    for s in scored:
        mark = "OK " if s["correct"] else "X  "
        print(f"    {mark}{s['ticker']:<6} {s['action']:<4} conf={s['confidence']:.0%} "
              f"ret={s['return']:+.1%}  (entry {s['entry_price']:.2f} -> exit {s['exit_price']:.2f})")


if __name__ == "__main__":
    main()
