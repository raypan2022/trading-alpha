"""
Finnhub news.

Uses Finnhub's free-tier /company-news endpoint — symbol-scoped, dated, and
already source-attributed, so it's cleaner than yfinance's aggregated feed
(no need for the ticker-relevance heuristic; Finnhub returns articles for the
requested symbol only).

Free key required (no card): https://finnhub.io — set FINNHUB_API_KEY in .env.
Free tier limit is 60 calls/min; the cache layer (added later) protects that
during parallel debate loops.
"""
import os
import requests
from datetime import date, timedelta
from difflib import SequenceMatcher

from . import cache

FINNHUB_BASE = "https://finnhub.io/api/v1"
PROFILE_TTL = 7 * 24 * 3600   # company name changes ~never

# Words too generic to use as company-name relevance signals.
_STOP_WORDS = {"inc", "corp", "ltd", "llc", "co", "the", "and", "group", "holdings", "company"}


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


def _is_relevant(headline: str, ticker: str, company_name: str) -> bool:
    """
    True if the HEADLINE centers on this ticker/company.

    Finnhub's company-news is symbol-scoped but includes articles that merely
    *mention* the company (e.g. a Wendy's short-squeeze piece that references
    Reddit users). We check the headline specifically — not the summary —
    because a tangential mention rarely makes the headline, but the real
    subject almost always does.
    """
    text = headline.lower()
    if ticker.lower() in text:
        return True
    for word in company_name.lower().split():
        word = word.strip(".,")
        if len(word) > 3 and word not in _STOP_WORDS and word in text:
            return True
    return False


def _get_company_name(ticker: str, api_key: str) -> str:
    """Resolve a ticker to its company name (cached), for relevance matching."""
    cache_key = f"finnhub_name_{ticker.upper()}"
    cached = cache.get(cache_key, PROFILE_TTL)
    if cached is not None:
        return cached
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/stock/profile2",
            params={"symbol": ticker.upper(), "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        name = resp.json().get("name", "") or ticker
    except requests.RequestException:
        name = ticker  # fall back to ticker-only matching
    cache.set(cache_key, name)
    return name


def get_finnhub_news(ticker: str, days: int = 7, max_articles: int = 5,
                     as_of: str | None = None) -> str:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return "ERROR: FINNHUB_API_KEY not set. Add a free key from finnhub.io to .env."

    ck = f"news_{ticker.upper()}_{as_of or 'live'}"
    hit = cache.get(ck, cache.ttl_for(as_of))
    if hit is not None:
        return hit

    try:
        company_name = _get_company_name(ticker, api_key)
        # as_of (ISO date): for backtesting, end the news window at that date so
        # the agent never sees headlines that hadn't been published yet.
        to_date = date.fromisoformat(as_of) if as_of else date.today()
        from_date = to_date - timedelta(days=days)
        resp = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": ticker.upper(),
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        articles = resp.json()

        if not isinstance(articles, list) or not articles:
            cache.set(ck, "No recent news found.")
            return "No recent news found."

        articles.sort(key=lambda a: a.get("datetime", 0), reverse=True)

        out = []
        seen: list[str] = []
        for art in articles:
            headline = (art.get("headline") or "").strip()
            summary = (art.get("summary") or "").strip() or "No summary."
            source = art.get("source") or "Unknown"
            if not headline:
                continue
            if not _is_relevant(headline, ticker, company_name):
                continue
            if any(_similar(headline, s) for s in seen):
                continue
            seen.append(headline)
            # Truncate long summaries to keep token usage down before LLM injection.
            if len(summary) > 300:
                summary = summary[:300].rsplit(" ", 1)[0] + "..."
            out.append(f"[{source}] {headline}\n{summary}")
            if len(out) >= max_articles:
                break

        result = "\n---\n".join(out) if out else "No recent news found."
        cache.set(ck, result)
        return result

    except requests.RequestException as e:
        return f"ERROR: Finnhub news request failed ({e}). Proceed with existing context."
    except Exception as e:
        return f"ERROR: Finnhub news parsing failed ({e})."


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(get_finnhub_news(t))
