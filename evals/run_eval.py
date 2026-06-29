"""
Evaluation harness — runner.

Runs the full committee over a basket of tickers and logs each verdict with the
entry price and timestamp, so it can be scored later against the stock's actual
forward move (see score_eval.py).

Two kinds of signal:
  1. IMMEDIATE robustness checks (no price ground truth needed): did every
     ticker produce a complete, valid verdict with non-empty bull/bear reports?
     This catches regressions like the empty-thesis loop bug across the whole
     basket, not just one hand-run ticker.
  2. DEFERRED outcome scoring: logged here, scored after the horizon elapses.

Run from the project root:
    python -m evals.run_eval                 # full basket
    python -m evals.run_eval AAPL RDDT       # ad-hoc subset (quick checks)
"""
import os
import sys
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
import yfinance as yf

from graph import build_graph

load_dotenv()

BASKET = ["AAPL", "NVDA", "MSFT", "RDDT", "TSLA", "AMD"]
HORIZON_DAYS = 30
RESULTS_PATH = os.getenv(
    "EVAL_RESULTS_PATH", os.path.join(os.path.dirname(__file__), "results.jsonl")
)

MIN_REPORT_CHARS = 50   # below this a report is treated as an empty/stub failure


def _entry_price(ticker: str):
    try:
        return yf.Ticker(ticker).info.get("currentPrice")
    except Exception:
        return None


def _check(result: dict, verdict) -> list:
    """Return a list of problems with a run; empty means clean."""
    problems = []
    if verdict is None:
        problems.append("no verdict")
    else:
        if verdict.action not in {"BUY", "HOLD", "SELL"}:
            problems.append(f"bad action '{verdict.action}'")
        if not (0.0 <= verdict.confidence_score <= 1.0):
            problems.append(f"confidence out of range ({verdict.confidence_score})")
        if verdict.target_price_30d <= 0:
            problems.append("non-positive target price")
    if len((result.get("bull_report") or "").strip()) < MIN_REPORT_CHARS:
        problems.append("bull report empty/stub")
    if len((result.get("bear_report") or "").strip()) < MIN_REPORT_CHARS:
        problems.append("bear report empty/stub")
    return problems


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        return

    basket = [t.upper() for t in sys.argv[1:]] or BASKET
    app = build_graph()
    rows, checks = [], []

    for ticker in basket:
        print(f"\n{'#' * 50}\n# Evaluating {ticker}\n{'#' * 50}")
        try:
            result = app.invoke({
                "ticker": ticker,
                "bull_report": "",
                "bear_report": "",
                "final_verdict": None,
            })
        except Exception as e:
            checks.append((ticker, False, f"crashed: {e}"))
            continue

        verdict = result.get("final_verdict")
        problems = _check(result, verdict)
        checks.append((ticker, not problems, "; ".join(problems) if problems else "ok"))

        if verdict is not None:
            rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "action": verdict.action,
                "confidence": verdict.confidence_score,
                "target_price_30d": verdict.target_price_30d,
                "entry_price": _entry_price(ticker),
                "horizon_days": HORIZON_DAYS,
            })

    if rows:
        with open(RESULTS_PATH, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # --- Immediate robustness report ---
    print("\n" + "=" * 50)
    print("  ROBUSTNESS REPORT (immediate, no ground truth)")
    print("=" * 50)
    for ticker, ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {ticker:<6} {msg}")
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {passed}/{len(checks)} produced clean, complete verdicts.")

    if rows:
        actions = {}
        for r in rows:
            actions[r["action"]] = actions.get(r["action"], 0) + 1
        mean_conf = sum(r["confidence"] for r in rows) / len(rows)
        print(f"  Action mix: {actions}   Mean confidence: {mean_conf:.0%}")
        print(f"\n  Logged {len(rows)} verdict(s) -> {RESULTS_PATH}")
        print(f"  Score after the {HORIZON_DAYS}d horizon with: python -m evals.score_eval")


if __name__ == "__main__":
    main()
