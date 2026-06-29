# Trading Alpha

A hybrid multi-agent financial forecasting system. Two isolated local LLM agents (Bull and Bear) debate a stock ticker in parallel, then a cloud model arbitrates a structured trading verdict.

## Architecture

```
[User Input: Ticker]
        │
┌───────┴───────┐
▼               ▼
Bull Analyst   Bear Analyst
(Qwen 3.5 4B)  (Qwen 3.5 4B)
└───────┬───────┘
        ▼
 Portfolio Manager
   (Cloud Judge)
        │
        ▼
 Structured Verdict
 BUY / HOLD / SELL
```

Bull and Bear agents run in parallel with zero-knowledge isolation — they never see each other's reasoning, only the same raw market data. The cloud judge cross-examines both dispatches and returns a validated `TradingVerdict`.

## Setup

**Requirements**
- Python 3.11+
- [Ollama](https://ollama.com) running locally with `qwen3.5:4b` pulled
- OpenAI API key (for the cloud judge)

```bash
ollama pull qwen3.5:4b
pip install langgraph langchain-openai langchain-ollama langchain-core yfinance pydantic python-dotenv
```

Add your key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

## Usage

```bash
python main.py
```

Enter any valid ticker (e.g. `AAPL`, `NVDA`, `TSLA`) and the committee will convene.

## Output

```
  Action:           BUY
  30d Target Price: $198.50
  Confidence:       82%

  Bull Concession:  Strong revenue growth but near 52-week high limits upside.
  Bear Concession:  High PE overridden by accelerating forward earnings estimates.

  Rationale:        ...
```

## Tools

Each agent autonomously decides which tools to call (up to a bounded budget) before writing its thesis. Data providers live in `sources/`.

| Tool | Source | Data |
|---|---|---|
| `get_price_snapshot` | yfinance | Current price, 52-week range, market cap, beta, and P/E computed from SEC EPS |
| `get_fundamentals` | SEC EDGAR | Authoritative XBRL filing data: revenue, net income, cash flow, assets, equity, EPS, plus derived net margin / ROE / debt-to-assets |
| `get_recent_news` | Finnhub | Symbol-scoped recent headlines, headline-relevance filtered and deduplicated |

Cold data (SEC filings) is cached to disk with a 24h TTL; the ticker→CIK map and company names are cached longer. Fundamentals are filtered to core statement lines before reaching the model — the raw XBRL blob is never injected.

## Evaluation

The agent is measured, not vibe-checked. The harness in `evals/` provides two kinds of signal:

**Immediate robustness** (no market outcome needed) — runs the committee over a basket of tickers and verifies every run produces a complete, schema-valid verdict with non-empty bull *and* bear theses. This catches regressions (e.g. an agent ending on an empty report) across a whole basket automatically.

```bash
python -m evals.run_eval              # full basket
python -m evals.run_eval AAPL RDDT    # ad-hoc subset
```

**Deferred outcome scoring** — each verdict is logged with its entry price and timestamp, then scored against the stock's actual move once the 30-day horizon elapses:

- **Directional accuracy** — did BUY/SELL/HOLD match the realized move (with a ±2% flat band for HOLD)?
- **Target-price error** — MAPE of the 30-day target vs. the realized price.
- **Confidence calibration** — do high-confidence calls actually hit more often than low-confidence ones? A model that knows *when* it's right is more useful than one with flat confidence.

```bash
python -m evals.score_eval            # scores any matured verdicts
```

Scoring uses forward price data only (no lookahead), so results accumulate honestly over time. Logged verdicts live in `evals/results.jsonl` (gitignored — it's a local, append-only run log).

**Point-in-time backtest** — for fast iteration without the 30-day wait, the agent can be run "as of" a past date and scored immediately against the realized move. Every data source is restricted to what was knowable on that date: historical price + trailing 52-week range, SEC filings made by then, and news up to then.

```bash
python -m evals.backtest                   # default case set
python -m evals.backtest AAPL 2026-04-15   # single ad-hoc case
```

Two lookahead traps it avoids:
- **Data leakage** — an `as_of` date is threaded through every tool, so the agent never sees a price, filing, or headline from after the decision date.
- **Model-memory leakage** — the *local model* may already "remember" a stock's move if the date predates its training cutoff. Backtest dates must be **after the local model's knowledge cutoff** (and ≥30 days before today so the outcome exists). Set them in `DEFAULT_CASES`.
