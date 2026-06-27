from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState, TradingVerdict
from tools import fetch_all_tool_data

LOCAL_MODEL = "qwen3.5:4b"
CLOUD_MODEL = "gpt-5.4-mini"


def bull_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    print(f"\n\033[92m[BULL ANALYST]\033[0m Fetching data and building bull thesis for {ticker}...")

    tool_data = fetch_all_tool_data(ticker)

    llm = ChatOllama(model=LOCAL_MODEL, temperature=0.4, num_predict=700)
    messages = [
        SystemMessage(content=(
            f"You are an isolated Bull Research Analyst for {ticker}. "
            f"Your sole mandate: find the strongest possible bullish signals in the data provided. "
            f"Look for: growth momentum, undervaluation vs peers, strong margins, positive catalysts, "
            f"technical support levels near 52-week lows, and favorable news.\n"
            f"Rules:\n"
            f"- Structured bullet points only. Max 250 words.\n"
            f"- Cite specific numbers from the data (prices, ratios, percentages).\n"
            f"- Do NOT mention bear-side risks — that is a separate analyst's job.\n"
            f"- End your report with exactly: ANALYSIS_COMPLETE"
        )),
        HumanMessage(content=f"MARKET DATA FOR {ticker}:\n\n{tool_data}\n\nBuild your bull thesis now."),
    ]

    response = llm.invoke(messages)
    print(f"\033[92m[BULL]\033[0m {response.content[:120]}...")
    return {"bull_report": response.content}


def bear_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    print(f"\n\033[91m[BEAR ANALYST]\033[0m Fetching data and building bear thesis for {ticker}...")

    tool_data = fetch_all_tool_data(ticker)

    llm = ChatOllama(model=LOCAL_MODEL, temperature=0.4, num_predict=700)
    messages = [
        SystemMessage(content=(
            f"You are an isolated Bear Risk Analyst for {ticker}. "
            f"Your sole mandate: find the strongest possible bearish signals in the data provided. "
            f"Look for: overvaluation, margin compression, high debt load, negative news catalysts, "
            f"proximity to 52-week highs (downside risk), low volume (weak conviction), poor ROE.\n"
            f"Rules:\n"
            f"- Structured bullet points only. Max 250 words.\n"
            f"- Cite specific numbers from the data (prices, ratios, percentages).\n"
            f"- Do NOT mention bull-side positives — that is a separate analyst's job.\n"
            f"- End your report with exactly: ANALYSIS_COMPLETE"
        )),
        HumanMessage(content=f"MARKET DATA FOR {ticker}:\n\n{tool_data}\n\nBuild your bear thesis now."),
    ]

    response = llm.invoke(messages)
    print(f"\033[91m[BEAR]\033[0m {response.content[:120]}...")
    return {"bear_report": response.content}


def judge_node(state: AgentState) -> dict:
    print(f"\n\033[96m[PORTFOLIO MANAGER]\033[0m Arbitrating between bull and bear dispatches...")

    llm = ChatOpenAI(model=CLOUD_MODEL, temperature=0.1)
    structured_llm = llm.with_structured_output(TradingVerdict)

    messages = [
        SystemMessage(content=(
            "You are the Head Portfolio Manager running a non-cooperative trading committee. "
            "You receive two adversarial research briefs — one hyper-bull, one hyper-bear — on the same asset. "
            "Cross-examine their specific data points, identify logical leaps or cherry-picked numbers, "
            "weigh the evidence impartially, and deliver a definitive verdict. "
            "Populate every field of the output schema with precise, evidence-backed content."
        )),
        HumanMessage(content=(
            f"ASSET: {state['ticker']}\n\n"
            f"{'='*40}\n"
            f"BULL ANALYST DISPATCH:\n"
            f"{'='*40}\n"
            f"{state['bull_report']}\n\n"
            f"{'='*40}\n"
            f"BEAR ANALYST DISPATCH:\n"
            f"{'='*40}\n"
            f"{state['bear_report']}\n\n"
            f"Arbitrate and return your structured verdict now."
        )),
    ]

    verdict: TradingVerdict = structured_llm.invoke(messages)
    return {"final_verdict": verdict}
