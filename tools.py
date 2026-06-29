"""
Agent-facing tool surface.

Thin wrappers the agents call. Price stays on yfinance (free, and Finnhub gates
candles behind premium); fundamentals come from SEC EDGAR (authoritative);
news comes from Finnhub (symbol-scoped, cleaner than yfinance's feed). The
actual data-provider logic lives in sources/.

Every tool accepts an optional as_of ISO date. When set (backtesting), the tool
returns only what was knowable on that date — historical price, filings made by
then, and news up to then — so a backtested verdict has no lookahead.
"""
from datetime import date, timedelta

import yfinance as yf

from sources import cache
from sources.sec import get_sec_fundamentals, get_diluted_eps
from sources.finnhub import get_finnhub_news


def price_on(ticker: str, on_date: str):
    """Closing price on the first trading day at/after on_date (ISO). None if unavailable."""
    try:
        start = date.fromisoformat(on_date)
        end = start + timedelta(days=7)
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=end.isoformat())
        if hist.empty:
            return None
        return float(hist.iloc[0]["Close"])
    except Exception:
        return None


def get_price_snapshot(ticker: str, as_of: str | None = None) -> str:
    ck = f"price_{ticker.upper()}_{as_of or 'live'}"
    hit = cache.get(ck, cache.ttl_for(as_of))
    if hit is not None:
        return hit
    try:
        eps = get_diluted_eps(ticker, as_of=as_of)

        if as_of is None:
            info = yf.Ticker(ticker).info
            price = info.get("currentPrice")
            high52 = info.get("fiftyTwoWeekHigh", "N/A")
            low52 = info.get("fiftyTwoWeekLow", "N/A")
            mktcap = info.get("marketCap", "N/A")
            avgvol = info.get("averageVolume", "N/A")
            beta = info.get("beta", "N/A")
        else:
            # Historical: last close on/before as_of, 52-wk range from the
            # trailing year. Market cap and beta need point-in-time share counts
            # / regressions we don't fetch, so they're marked N/A in backtest.
            asof_date = date.fromisoformat(as_of)
            start = (asof_date - timedelta(days=370)).isoformat()
            end = (asof_date + timedelta(days=1)).isoformat()
            hist = yf.Ticker(ticker).history(start=start, end=end)
            if hist.empty:
                return f"ERROR: no price history for {ticker} as of {as_of}."
            price = float(hist.iloc[-1]["Close"])
            high52 = round(float(hist["High"].max()), 2)
            low52 = round(float(hist["Low"].min()), 2)
            avgvol = int(hist["Volume"].tail(30).mean())
            mktcap = "N/A (point-in-time)"
            beta = "N/A (point-in-time)"

        pe_line = "P/E (price/SEC EPS): N/A"
        if isinstance(price, (int, float)) and eps and eps > 0:
            pe_line = f"P/E (price/SEC EPS): {price / eps:.1f}"

        price_str = f"{price:.2f}" if isinstance(price, float) else price
        result = (
            f"Current Price:  {price_str}\n"
            f"52-Week High:   {high52}\n"
            f"52-Week Low:    {low52}\n"
            f"Market Cap:     {mktcap}\n"
            f"Avg Volume:     {avgvol}\n"
            f"Beta:           {beta}\n"
            f"{pe_line}"
        )
        cache.set(ck, result)
        return result
    except Exception as e:
        return f"ERROR: {e}"


def get_fundamentals(ticker: str, as_of: str | None = None) -> str:
    """Authoritative fundamentals from SEC EDGAR filings."""
    return get_sec_fundamentals(ticker, as_of=as_of)


def get_recent_news(ticker: str, as_of: str | None = None) -> str:
    """Recent symbol-scoped news from Finnhub."""
    return get_finnhub_news(ticker, as_of=as_of)
