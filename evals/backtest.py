"""
Point-in-time backtest.

Runs the committee "as of" a past date — with every data source restricted to
what was knowable then (historical price, filings made by that date, news up to
that date) — and scores the verdict against the actual 30-day forward move
immediately. No 30-day wait.

Two traps this design avoids:
  1. Data lookahead: the as_of date is threaded through every tool, so the agent
     never sees a price, filing, or headline from after the decision date.
  2. Model-memory lookahead: the LOCAL model may already "remember" how a stock
     did if the date predates its training cutoff. So backtest dates MUST be
     AFTER the local model's knowledge cutoff. Pick recent dates accordingly.

Each date also needs to be at least HORIZON_DAYS before today so the realized
move actually exists.

Run from the project root:
    python -m evals.backtest                      # default case set
    python -m evals.backtest AAPL 2026-04-15      # single ad-hoc case
"""
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

from graph import build_graph
from tools import price_on
from evals.score_eval import _directional_correct

load_dotenv()

HORIZON_DAYS = 30

# (ticker, as_of) — EDIT these to dates after your local model's knowledge
# cutoff and at least HORIZON_DAYS before today.
DEFAULT_CASES = [
    ("AAPL", "2026-04-15"),
    ("NVDA", "2026-04-15"),
    ("MSFT", "2026-05-01"),
    ("RDDT", "2026-05-01"),
    ("TSLA", "2026-04-22"),
]


def _run_case(app, ticker: str, as_of: str):
    result = app.invoke({
        "ticker": ticker,
        "as_of": as_of,
        "bull_report": "",
        "bear_report": "",
        "final_verdict": None,
    })
    return result.get("final_verdict")


def main():
    if len(sys.argv) == 3:
        cases = [(sys.argv[1].upper(), sys.argv[2])]
    else:
        cases = DEFAULT_CASES

    app = build_graph()
    scored = []

    for ticker, as_of in cases:
        print(f"\n{'#' * 50}\n# Backtesting {ticker} as of {as_of}\n{'#' * 50}")
        verdict = _run_case(app, ticker, as_of)
        if verdict is None:
            print(f"  (no verdict for {ticker})")
            continue

        entry = price_on(ticker, as_of)
        exit_date = (date.fromisoformat(as_of) + timedelta(days=HORIZON_DAYS)).isoformat()
        exit_price = price_on(ticker, exit_date)
        if not entry or not exit_price:
            print(f"  (missing price data for {ticker}: entry={entry}, exit={exit_price})")
            continue

        ret = (exit_price - entry) / entry
        scored.append({
            "ticker": ticker,
            "as_of": as_of,
            "action": verdict.action,
            "confidence": verdict.confidence_score,
            "target": verdict.target_price_30d,
            "entry": entry,
            "exit": exit_price,
            "return": ret,
            "correct": _directional_correct(verdict.action, ret),
            "target_err": abs(verdict.target_price_30d - exit_price) / exit_price,
        })

    print("\n" + "=" * 50)
    print("  BACKTEST SCORECARD")
    print("=" * 50)
    if not scored:
        print("  No scorable cases.")
        return

    acc = sum(1 for s in scored if s["correct"]) / len(scored)
    mape = sum(s["target_err"] for s in scored) / len(scored)
    print(f"  Cases scored:         {len(scored)}")
    print(f"  Directional accuracy: {acc:.0%}")
    print(f"  Target price MAPE:    {mape:.1%}")

    print("\n  Calibration (confidence bucket -> actual hit rate):")
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        bucket = [s for s in scored if lo <= s["confidence"] < hi]
        if bucket:
            hit = sum(1 for s in bucket if s["correct"]) / len(bucket)
            print(f"    {lo:.0%}-{min(hi,1.0):.0%}: {hit:.0%} hit rate  (n={len(bucket)})")

    print("\n  Per-case:")
    for s in scored:
        mark = "OK " if s["correct"] else "X  "
        print(f"    {mark}{s['ticker']:<6} {s['as_of']}  {s['action']:<4} conf={s['confidence']:.0%} "
              f"ret={s['return']:+.1%}  (entry {s['entry']:.2f} -> exit {s['exit']:.2f})")


if __name__ == "__main__":
    main()
