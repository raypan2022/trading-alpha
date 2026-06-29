"""
Hybrid Multi-Agent Financial Forecasting System (v1)

Dependencies:
    pip install langgraph langchain-openai langchain-ollama langchain-core yfinance pydantic

Requires:
    - Ollama running locally with qwen3.5:4b pulled
    - OPENAI_API_KEY environment variable set
"""
import os
import sys
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        return

    if len(sys.argv) > 1:
        ticker = sys.argv[1].strip().upper()
    else:
        ticker = input("Enter ticker symbol: ").strip().upper()

    if not ticker:
        print("Ticker cannot be empty.")
        return

    print(f"\n{'='*50}")
    print(f"  FINANCIAL COMMITTEE CONVENING: {ticker}")
    print(f"{'='*50}")
    print("  Bull and Bear analysts running in parallel...")

    app = build_graph()
    result = app.invoke({
        "ticker": ticker,
        "as_of": None,
        "bull_report": "",
        "bear_report": "",
        "final_verdict": None,
    })

    verdict = result["final_verdict"]

    print(f"\n{'='*50}")
    print(f"  FINAL VERDICT: {verdict.ticker}")
    print(f"{'='*50}")
    print(f"  Action:           \033[1m{verdict.action}\033[0m")
    print(f"  30d Target Price: ${verdict.target_price_30d:.2f}")
    print(f"  Confidence:       {verdict.confidence_score:.0%}")
    print(f"\n  Bull Concession:  {verdict.bull_concession}")
    print(f"  Bear Concession:  {verdict.bear_concession}")
    print(f"\n  Rationale:\n  {verdict.core_rationale}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
