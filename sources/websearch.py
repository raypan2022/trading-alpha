"""
Free web search (DuckDuckGo, no API key).

The open-ended "look up whatever you want" tool — a debater can search for
sentiment, recent events, analyst views, anything. This subsumes a Reddit
search (just query "reddit <ticker> sentiment").

IMPORTANT — live only. Web search returns TODAY's web, so running it for a
backtest as_of a past date would leak information published after the decision
date. When as_of is set we refuse and tell the agent to use point-in-time
LOOKUP instead, keeping backtests honest.

Confirmation-bias guardrails live in the debate prompt (neutral query framing +
the must-report-disconfirming rule); this tool just returns whatever the search
yields, unfiltered by slant, so the agent sees a range of results.
"""
from . import cache


def _ddgs_class():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            return None


def web_search(query: str, as_of: str | None = None, max_results: int = 4) -> str:
    if as_of:
        return (
            "WEB SEARCH UNAVAILABLE in backtest mode — it would return information published after "
            f"the decision date ({as_of}) and leak the future. Use a LOOKUP metric instead."
        )

    DDGS = _ddgs_class()
    if DDGS is None:
        return "ERROR: web search library not installed (pip install ddgs)."

    ck = f"search_{query.lower().strip()}"
    hit = cache.get(ck, cache.LIVE_TTL)
    if hit is not None:
        return hit

    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No web results found."
        out = []
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            if len(body) > 300:
                body = body[:300].rsplit(" ", 1)[0] + "..."
            out.append(f"- {title}: {body}")
        result = f"WEB SEARCH RESULTS for '{query}':\n" + "\n".join(out)
        cache.set(ck, result)
        return result
    except Exception as e:
        return f"ERROR: web search failed ({e})."


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Amazon AWS margin trend"
    print(web_search(q))
