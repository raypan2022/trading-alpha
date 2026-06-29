import yfinance as yf
from difflib import SequenceMatcher

TRUSTED_SOURCES = {
    "Reuters", "Associated Press", "Bloomberg", "The Wall Street Journal",
    "Financial Times", "CNBC", "MarketWatch", "Barron's", "AP News",
}

# Words too generic to use as company name signals
_STOP_WORDS = {"inc", "corp", "ltd", "llc", "co", "the", "and", "group", "holdings", "company"}


def _is_relevant(title: str, summary: str, ticker: str, company_name: str) -> bool:
    """Returns True if the article is actually about this ticker or company."""
    text = (title + " " + summary).lower()

    if ticker.lower() in text:
        return True

    # Match any meaningful word from the company name (e.g. "Reddit" from "Reddit Inc.")
    for word in company_name.lower().split():
        word = word.strip(".,")
        if len(word) > 3 and word not in _STOP_WORDS and word in text:
            return True

    return False


def get_price_snapshot(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return (
            f"Current Price:  {info.get('currentPrice', 'N/A')}\n"
            f"52-Week High:   {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52-Week Low:    {info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"Market Cap:     {info.get('marketCap', 'N/A')}\n"
            f"Avg Volume:     {info.get('averageVolume', 'N/A')}\n"
            f"Beta:           {info.get('beta', 'N/A')}"
        )
    except Exception as e:
        return f"ERROR: {e}"


def _pct(value) -> str:
    """Render a decimal ratio (0.286) as a percentage string (28.6%)."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def get_fundamentals(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return (
            f"Trailing PE:    {info.get('trailingPE', 'N/A')}\n"
            f"Forward PE:     {info.get('forwardPE', 'N/A')}\n"
            f"Profit Margins: {_pct(info.get('profitMargins'))}\n"
            f"Revenue Growth: {_pct(info.get('revenueGrowth'))}\n"
            f"Debt/Equity:    {info.get('debtToEquity', 'N/A')}\n"
            f"EPS (TTM):      {info.get('trailingEps', 'N/A')}\n"
            f"Return on Eq:   {_pct(info.get('returnOnEquity'))}"
        )
    except Exception as e:
        return f"ERROR: {e}"


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


def get_recent_news(ticker: str, max_articles: int = 5) -> str:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get("shortName") or info.get("longName") or ticker
        raw_news = stock.news or []

        filtered = []
        seen_titles: list[str] = []

        for article in raw_news:
            # yfinance >=0.2.x nests publisher under article["content"]["provider"]["displayName"]
            # older versions use a flat "publisher" string — handle both
            content = article.get("content", {})
            if isinstance(content, dict):
                publisher = content.get("provider", {}).get("displayName", "")
                title = content.get("title", article.get("title", ""))
                summary = content.get("summary", article.get("summary", "No summary available."))
            else:
                publisher = article.get("publisher", "")
                title = article.get("title", "")
                summary = article.get("summary", "No summary available.")

            if not any(trusted.lower() in publisher.lower() for trusted in TRUSTED_SOURCES):
                continue

            if not _is_relevant(title, summary, ticker, company_name):
                continue

            if any(_similar(title, seen) for seen in seen_titles):
                continue
            seen_titles.append(title)

            filtered.append(f"[{publisher}] {title}\n{summary}")
            if len(filtered) >= max_articles:
                break

        if not filtered:
            return "No news from trusted sources found."
        return "\n---\n".join(filtered)
    except Exception as e:
        return f"ERROR: {e}"


def fetch_all_tool_data(ticker: str) -> str:
    price = get_price_snapshot(ticker)
    fundamentals = get_fundamentals(ticker)
    news = get_recent_news(ticker)
    return (
        f"=== PRICE SNAPSHOT ===\n{price}\n\n"
        f"=== FUNDAMENTALS ===\n{fundamentals}\n\n"
        f"=== RECENT NEWS (trusted sources only) ===\n{news}"
    )
