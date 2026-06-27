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
pip install langgraph langchain-openai langchain-ollama langchain-core yfinance pydantic
export OPENAI_API_KEY=your_key_here
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

## Tools (v1)

All market data is sourced from `yfinance` with no API key required. News is filtered to trusted outlets only (Reuters, Bloomberg, WSJ, CNBC, FT, etc.) and deduplicated before reaching the agents.

| Tool | Data |
|---|---|
| `get_price_snapshot` | Current price, 52-week range, market cap, beta |
| `get_fundamentals` | PE, margins, debt/equity, revenue growth, EPS, ROE |
| `get_recent_news` | Last 5 headlines from trusted sources |
