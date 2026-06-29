"""
Market regime.

A top-down read on the overall US market, so single-name analysis happens inside
a known macro context (a BUY in a calm uptrend is not the same call as a BUY with
the VIX at 35). All free from yfinance index history — no new API.

Signals:
  - S&P 500 / Nasdaq trend vs 50- and 200-day moving averages, plus 1m/3m return
  - VIX level (fear gauge)
  - 10Y Treasury yield level and recent direction

Threads as_of like the other sources so backtests stay point-in-time (the regime
is computed from history up to the decision date, never after).
"""
from datetime import date, timedelta

import yfinance as yf


def _closes(ticker: str, as_of: str | None):
    tk = yf.Ticker(ticker)
    if as_of:
        end = (date.fromisoformat(as_of) + timedelta(days=1)).isoformat()
        start = (date.fromisoformat(as_of) - timedelta(days=400)).isoformat()
        hist = tk.history(start=start, end=end)
    else:
        hist = tk.history(period="1y")
    return hist["Close"].dropna()


def _trend(closes) -> str:
    if len(closes) < 50:
        return "insufficient history"
    last = closes.iloc[-1]
    ma50 = closes.iloc[-50:].mean()
    ma200 = closes.iloc[-200:].mean() if len(closes) >= 200 else None
    if ma200 is None:
        return "above 50d MA" if last > ma50 else "below 50d MA"
    if last > ma50 and ma50 > ma200:
        return "uptrend (above 50d & 200d MA)"
    if last < ma50 and ma50 < ma200:
        return "downtrend (below 50d & 200d MA)"
    return "mixed (between MAs)"


def _ret(closes, days: int):
    if len(closes) <= days:
        return None
    return closes.iloc[-1] / closes.iloc[-1 - days] - 1


def _classify_trend(trend: str) -> str:
    """Map a trend label to one of 'up' / 'down' / 'mixed'."""
    if "uptrend" in trend or "above" in trend:
        return "up"
    if "downtrend" in trend or "below" in trend:
        return "down"
    return "mixed"


def _index_line(label: str, ticker: str, as_of: str | None) -> tuple:
    """Return (display_line, direction) for one index; direction in up/down/mixed/None."""
    try:
        closes = _closes(ticker, as_of)
        if closes.empty:
            return f"  {label + ':':<11}N/A", None
        last = closes.iloc[-1]
        trend = _trend(closes)
        r1 = _ret(closes, 21)
        r3 = _ret(closes, 63)
        r1s = f"{r1*100:+.1f}%" if r1 is not None else "N/A"
        r3s = f"{r3*100:+.1f}%" if r3 is not None else "N/A"
        line = f"  {label + ':':<11}{last:,.2f} | trend: {trend} | 1mo {r1s} | 3mo {r3s}"
        return line, _classify_trend(trend)
    except Exception:
        return f"  {label + ':':<11}N/A", None


def _vix(as_of: str | None) -> tuple:
    try:
        closes = _closes("^VIX", as_of)
        if closes.empty:
            return "  VIX:        N/A", None
        v = closes.iloc[-1]
        if v < 15:
            mood = "calm (low volatility)"
        elif v < 20:
            mood = "normal"
        elif v < 30:
            mood = "elevated (caution)"
        else:
            mood = "high stress (fear)"
        return f"  VIX:        {v:.2f} | {mood}", v
    except Exception:
        return "  VIX:        N/A", None


def _ten_year(as_of: str | None) -> str:
    try:
        closes = _closes("^TNX", as_of)
        if closes.empty:
            return "  10Y Yield:  N/A"
        last = closes.iloc[-1]
        direction = ""
        if len(closes) > 21:
            prev = closes.iloc[-22]
            if last > prev + 0.05:
                direction = " | rising over last month"
            elif last < prev - 0.05:
                direction = " | falling over last month"
            else:
                direction = " | roughly flat over last month"
        return f"  10Y Yield:  {last:.2f}{direction}"
    except Exception:
        return "  10Y Yield:  N/A"


def get_market_regime(as_of: str | None = None) -> str:
    """Public tool: a compact, point-in-time read of the overall market regime."""
    when = as_of if as_of else "today"

    sp_line, sp_dir = _index_line("S&P 500", "^GSPC", as_of)
    nq_line, _ = _index_line("Nasdaq", "^IXIC", as_of)
    vix_line, vix_val = _vix(as_of)
    tnx_line = _ten_year(as_of)

    # Synthesize an overall label from trend + volatility.
    if sp_dir == "up" and vix_val is not None and vix_val < 20:
        overall = "RISK-ON"
    elif sp_dir == "down" or (vix_val is not None and vix_val > 28):
        overall = "RISK-OFF"
    else:
        overall = "NEUTRAL / MIXED"

    return (
        f"MARKET REGIME (as of {when})\n"
        f"  Overall:    {overall}\n"
        f"{sp_line}\n"
        f"{nq_line}\n"
        f"{vix_line}\n"
        f"{tnx_line}"
    )


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else None
    print(get_market_regime(as_of=a))
