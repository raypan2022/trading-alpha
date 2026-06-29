"""
Agent-facing tool surface.

Thin wrappers the agents call. Price stays on yfinance (free, and Finnhub gates
candles behind premium); fundamentals come from SEC EDGAR (authoritative);
news comes from Finnhub (symbol-scoped, cleaner than yfinance's feed). The
actual data-provider logic lives in sources/.
"""
import yfinance as yf

from sources.sec import get_sec_fundamentals, get_diluted_eps
from sources.finnhub import get_finnhub_news


def get_price_snapshot(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice")

        # P/E computed here (yfinance price / SEC EPS) so the agents get a real,
        # correct multiple instead of inventing one. yfinance's own PE fields are
        # unreliable; SEC EPS is authoritative.
        pe_line = "P/E (price/SEC EPS): N/A"
        eps = get_diluted_eps(ticker)
        if price is not None and eps and eps > 0:
            pe_line = f"P/E (price/SEC EPS): {price / eps:.1f}"

        return (
            f"Current Price:  {info.get('currentPrice', 'N/A')}\n"
            f"52-Week High:   {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52-Week Low:    {info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"Market Cap:     {info.get('marketCap', 'N/A')}\n"
            f"Avg Volume:     {info.get('averageVolume', 'N/A')}\n"
            f"Beta:           {info.get('beta', 'N/A')}\n"
            f"{pe_line}"
        )
    except Exception as e:
        return f"ERROR: {e}"


def get_fundamentals(ticker: str) -> str:
    """Authoritative fundamentals from SEC EDGAR filings."""
    return get_sec_fundamentals(ticker)


def get_recent_news(ticker: str) -> str:
    """Recent symbol-scoped news from Finnhub."""
    return get_finnhub_news(ticker)
