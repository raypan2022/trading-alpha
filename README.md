# Trading Alpha

A hybrid multi-agent financial forecasting system. Two isolated local LLM agents (Bull and Bear) debate a stock ticker in parallel, then a cloud model arbitrates a structured trading verdict.

## Architecture

```
[User Input: Ticker]
        │
        ▼
 Macro Strategist        ─┐ produces the market-regime read,
 (Qwen 3.5 9B)            │ shared as context with everyone below
        │                 │
        ▼                 │
   Bull Research  ────────┤  ← isolated: each forms an initial
   (Qwen 3.5 9B)          │    thesis without seeing the other
        │                 │
        ▼                 │
   Bear Research  ────────┘
   (Qwen 3.5 9B)
        │
        ▼
   ┌─► Debate Round ◄─┐      ← bull & bear now SEE each other
   │  (bull rebuts,   │        and rebut, looping until a round
   └── bear counters)─┘        cap or both run out of points
        │
        ▼
 Portfolio Manager
   (Cloud Judge)
        │
        ▼
 Structured Verdict
 BUY / HOLD / SELL
```

A **Macro Strategist** runs first and produces a top-down market-regime read (trend, volatility, rates), shared as context with both analysts and the judge — so the single-name analysis happens *inside* a known macro backdrop.

The **Bull** and **Bear** analysts then run in two phases. First, **isolated research**: each gathers data (autonomously choosing tools) and forms an initial thesis without seeing the other — preserving orthogonal viewpoints. Then a **multi-round debate**: they finally see each other's case and rebut directly, looping via a conditional graph edge until a round cap or both sides have nothing new. This cycle is what makes it a real agent system rather than a one-shot pipeline.

The **cloud judge** then weighs which points survived rebuttal — against the macro regime — and returns a validated `TradingVerdict`. The committee runs sequentially to keep streamed output readable; parallelizing the research phase is a planned upgrade.

## Setup

**Requirements**
- Python 3.11+
- [Ollama](https://ollama.com) running locally with `qwen3.5:9b` pulled
- OpenAI API key (for the cloud judge)

```bash
ollama pull qwen3.5:9b
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

The Macro Strategist additionally reads a **market-regime** signal (`sources/market.py`) — S&P 500 / Nasdaq trend vs moving averages, VIX level, and 10Y yield direction, all from yfinance index history. Unlike the three tools above, this isn't agent-selected: it's computed once and injected as shared context for everyone.

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
- **Model-memory leakage** — the *local model* may already "remember" a stock's move if the date predates its training cutoff. Backtest dates must be **after the local model's knowledge cutoff** (and ≥30 days before today so the outcome exists).

The grid is a list of `TICKERS` × a list of `AS_OF_DATES` (cross product), guarded by a `MODEL_KNOWLEDGE_CUTOFF` constant — any date on/before the cutoff or without an elapsed horizon is skipped automatically.
