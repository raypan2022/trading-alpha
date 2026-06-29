"""
SEC EDGAR fundamentals.

Pulls authoritative financial data straight from companies' XBRL filings via the
free SEC EDGAR REST API (no API key — SEC only asks for a descriptive
User-Agent with contact info, per their fair-access policy).

The hard part SEC glosses over: XBRL tags are inconsistent across companies
(revenue might be "Revenues" for one filer and
"RevenueFromContractWithCustomerExcludingAssessedTax" for another). We handle
that with an ordered list of candidate concept names per metric and take the
first that resolves.

Per the data-pipeline plan: raw SEC JSON is filtered down to core statement
lines and a few derived ratios *before* anything reaches the LLM context —
never inject the raw companyfacts blob.
"""
import requests
from . import cache

# SEC requires a User-Agent identifying the requester (name + contact email).
HEADERS = {"User-Agent": "trading-alpha raymondpan2022@gmail.com"}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

TICKER_MAP_TTL = 7 * 24 * 3600   # ticker->CIK mapping changes rarely
FACTS_TTL = 24 * 3600            # filings don't change intraday

# Ordered candidate XBRL concept names per metric (first match wins).
FLOW_CONCEPTS = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "Net Income": ["NetIncomeLoss"],
    "Operating Cash Flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}
BALANCE_CONCEPTS = {
    "Total Assets": ["Assets"],
    "Total Liabilities": ["Liabilities"],
    "Stockholders Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}
PERSHARE_CONCEPTS = {
    "EPS (Diluted)": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
}


def _get_cik(ticker: str) -> str | None:
    """Resolve a ticker to its zero-padded 10-digit CIK."""
    mapping = cache.get("sec_ticker_map", TICKER_MAP_TTL)
    if mapping is None:
        resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        # raw is { "0": {"cik_str": 320193, "ticker": "AAPL", "title": ...}, ... }
        mapping = {v["ticker"].upper(): str(v["cik_str"]) for v in raw.values()}
        cache.set("sec_ticker_map", mapping)

    cik = mapping.get(ticker.upper())
    return cik.zfill(10) if cik else None


def _fetch_facts(cik: str) -> dict:
    cached = cache.get(f"sec_facts_{cik}", FACTS_TTL)
    if cached is not None:
        return cached
    url = FACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    facts = resp.json()
    cache.set(f"sec_facts_{cik}", facts)
    return facts


def _latest_fact(usgaap: dict, candidates: list, prefer_annual: bool,
                 as_of: str | None = None) -> dict | None:
    """
    Extract the most recent value for the first matching concept.

    prefer_annual: for flow metrics (revenue, net income, cash flow) we want
    full-year figures, so prefer 10-K entries; fall back to whatever exists.
    Balance-sheet items are point-in-time, so we just take the latest end date.

    as_of (ISO date): for backtesting — only consider facts FILED on or before
    this date, so the agent never sees a filing that didn't exist yet. Each XBRL
    data point carries a "filed" date; ISO strings compare correctly as text.
    """
    for concept in candidates:
        node = usgaap.get(concept)
        if not node:
            continue
        for points in node.get("units", {}).values():
            pts = points
            if as_of:
                pts = [p for p in pts if p.get("filed", "") <= as_of]
            if not pts:
                continue
            pool = pts
            if prefer_annual:
                annual = [p for p in pts if p.get("form") == "10-K"]
                if annual:
                    pool = annual
            best = max(pool, key=lambda p: p.get("end", ""))
            return {
                "value": best.get("val"),
                "end": best.get("end"),
                "fy": best.get("fy"),
                "form": best.get("form"),
                "concept": concept,
            }
    return None


def _human(n) -> str:
    """Format a large dollar figure compactly (e.g. 391000000000 -> $391.0B)."""
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1e12:
        return f"{sign}${a/1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}${a/1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.1f}M"
    return f"{sign}${a:,.0f}"


def get_sec_fundamentals(ticker: str, as_of: str | None = None) -> str:
    """Public tool: return filtered, derived fundamentals from SEC filings.

    as_of (ISO date): only use filings made on or before this date (backtesting).
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return f"ERROR: No SEC CIK found for '{ticker}' (may be an ETF, ADR, or non-US filer)."

        facts = _fetch_facts(cik)
        usgaap = facts.get("facts", {}).get("us-gaap", {})
        if not usgaap:
            return f"ERROR: No us-gaap XBRL facts available for '{ticker}'."

        extracted = {}
        for label, candidates in FLOW_CONCEPTS.items():
            extracted[label] = _latest_fact(usgaap, candidates, prefer_annual=True, as_of=as_of)
        for label, candidates in BALANCE_CONCEPTS.items():
            extracted[label] = _latest_fact(usgaap, candidates, prefer_annual=False, as_of=as_of)
        for label, candidates in PERSHARE_CONCEPTS.items():
            extracted[label] = _latest_fact(usgaap, candidates, prefer_annual=True, as_of=as_of)

        lines = [f"SEC EDGAR Fundamentals — {ticker.upper()} (CIK {cik})"]

        for label in list(FLOW_CONCEPTS) + list(BALANCE_CONCEPTS):
            fact = extracted.get(label)
            if fact and fact["value"] is not None:
                period = f"FY{fact['fy']}, ended {fact['end']}" if fact.get("fy") else fact["end"]
                lines.append(f"  {label + ':':<22}{_human(fact['value'])}  ({period})")
            else:
                lines.append(f"  {label + ':':<22}N/A")

        eps = extracted.get("EPS (Diluted)")
        if eps and eps["value"] is not None:
            lines.append(f"  {'EPS (Diluted):':<22}{eps['value']}  (FY{eps.get('fy')})")

        # Derived ratios — computed here so the LLM gets the conclusion, not raw math.
        rev = (extracted.get("Revenue") or {}).get("value")
        ni = (extracted.get("Net Income") or {}).get("value")
        eq = (extracted.get("Stockholders Equity") or {}).get("value")
        assets = (extracted.get("Total Assets") or {}).get("value")
        liabs = (extracted.get("Total Liabilities") or {}).get("value")

        derived = []
        if rev and ni is not None and rev != 0:
            derived.append(f"  {'Net Margin:':<22}{ni / rev * 100:.1f}%")
        if eq and ni is not None and eq != 0:
            derived.append(f"  {'ROE:':<22}{ni / eq * 100:.1f}%")
        if assets and liabs is not None and assets != 0:
            derived.append(f"  {'Debt/Assets:':<22}{liabs / assets * 100:.1f}%")

        if derived:
            lines.append("Derived:")
            lines.extend(derived)

        return "\n".join(lines)

    except requests.RequestException as e:
        return f"ERROR: SEC request failed ({e}). Proceed with existing context."
    except Exception as e:
        return f"ERROR: SEC fundamentals parsing failed ({e})."


def get_diluted_eps(ticker: str, as_of: str | None = None) -> float | None:
    """Return latest diluted EPS as a float (for computing P/E), or None."""
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None
        usgaap = _fetch_facts(cik).get("facts", {}).get("us-gaap", {})
        fact = _latest_fact(usgaap, PERSHARE_CONCEPTS["EPS (Diluted)"], prefer_annual=True, as_of=as_of)
        if fact and fact["value"] is not None:
            return float(fact["value"])
    except (requests.RequestException, ValueError, TypeError):
        pass
    return None


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(get_sec_fundamentals(t))
