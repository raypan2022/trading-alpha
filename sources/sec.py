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

# Metrics whose multi-year ANNUAL trend the debaters can look up to ground a
# claim (e.g. "capex is surging" / "margins are compressing") in real numbers.
HISTORY_CONCEPTS = {
    "revenue": (FLOW_CONCEPTS["Revenue"], "money"),
    "net_income": (["NetIncomeLoss"], "money"),
    "operating_income": (["OperatingIncomeLoss"], "money"),
    "operating_cash_flow": (FLOW_CONCEPTS["Operating Cash Flow"], "money"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], "money"),
}
RATIO_HISTORY = {
    "net_margin": (["NetIncomeLoss"], FLOW_CONCEPTS["Revenue"]),
    "operating_margin": (["OperatingIncomeLoss"], FLOW_CONCEPTS["Revenue"]),
}
HISTORY_METRICS = list(HISTORY_CONCEPTS) + list(RATIO_HISTORY)


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


def _annual_series(usgaap: dict, candidates: list, as_of: str | None = None) -> dict:
    """Return {fiscal_year: point} of annual (10-K) values for the first matching concept."""
    for concept in candidates:
        node = usgaap.get(concept)
        if not node:
            continue
        for points in node.get("units", {}).values():
            pts = [p for p in points if p.get("form") == "10-K"]
            if as_of:
                pts = [p for p in pts if p.get("filed", "") <= as_of]
            if not pts:
                continue
            by_fy = {}
            for p in pts:
                fy = p.get("fy")
                if fy is None:
                    continue
                # keep the latest-ending entry per fiscal year (full-year figure)
                if fy not in by_fy or p.get("end", "") > by_fy[fy].get("end", ""):
                    by_fy[fy] = p
            if by_fy:
                return by_fy
    return {}


def get_metric_history(ticker: str, metric: str, as_of: str | None = None, periods: int = 5) -> str:
    """Return the multi-year annual trend of a single metric, point-in-time aware."""
    try:
        cik = _get_cik(ticker)
        if not cik:
            return f"ERROR: No SEC CIK found for '{ticker}'."
        usgaap = _fetch_facts(cik).get("facts", {}).get("us-gaap", {})
        metric = metric.lower().strip()

        if metric in HISTORY_CONCEPTS:
            candidates, fmt = HISTORY_CONCEPTS[metric]
            series = _annual_series(usgaap, candidates, as_of)
            if not series:
                return f"ERROR: no annual history for '{metric}' on {ticker}."
            fys = sorted(series)[-periods:]
            rows = [(fy, _human(series[fy].get("val")) if fmt == "money" else series[fy].get("val")) for fy in fys]
            first_v, last_v = series[fys[0]].get("val"), series[fys[-1]].get("val")

        elif metric in RATIO_HISTORY:
            num_c, den_c = RATIO_HISTORY[metric]
            num, den = _annual_series(usgaap, num_c, as_of), _annual_series(usgaap, den_c, as_of)
            fys = sorted(set(num) & set(den))[-periods:]
            if not fys:
                return f"ERROR: no annual history for '{metric}' on {ticker}."
            vals = {}
            rows = []
            for fy in fys:
                d = den[fy].get("val")
                v = (num[fy].get("val") / d * 100) if d else None
                vals[fy] = v
                rows.append((fy, f"{v:.1f}%" if v is not None else "N/A"))
            first_v, last_v = vals[fys[0]], vals[fys[-1]]

        else:
            return f"ERROR: unknown metric '{metric}'. Available: {', '.join(HISTORY_METRICS)}."

        trend = "flat"
        if first_v and last_v:
            if last_v > first_v * 1.05:
                trend = "rising"
            elif last_v < first_v * 0.95:
                trend = "falling"

        lines = [f"METRIC HISTORY — {ticker.upper()} {metric} (annual, last {len(rows)} FYs)"]
        lines += [f"  FY{fy}: {disp}" for fy, disp in rows]
        lines.append(f"  Trend: {trend}")
        return "\n".join(lines)

    except requests.RequestException as e:
        return f"ERROR: SEC request failed ({e})."
    except Exception as e:
        return f"ERROR: metric history failed ({e})."


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    m = sys.argv[2] if len(sys.argv) > 2 else None
    print(get_metric_history(t, m) if m else get_sec_fundamentals(t))
