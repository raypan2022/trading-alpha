import re
import ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState, TradingVerdict
from tools import get_price_snapshot, get_fundamentals, get_recent_news

LOCAL_MODEL = "qwen3.5:9b"
CLOUD_MODEL = "gpt-5.4-mini"
MAX_TOOL_CALLS = 3

# ANSI colors
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
GRAY  = "\033[90m"
BOLD  = "\033[1m"
RESET = "\033[0m"

TOOL_REGISTRY = {
    "get_price_snapshot": {
        "fn": get_price_snapshot,
        "desc": "Current price, 52-week high/low, market cap, beta, average volume.",
    },
    "get_fundamentals": {
        "fn": get_fundamentals,
        "desc": "SEC filing data: revenue, net income, cash flow, assets, equity, EPS, net margin, ROE, debt/assets.",
    },
    "get_recent_news": {
        "fn": get_recent_news,
        "desc": "Recent company-specific news headlines and summaries from the last 7 days.",
    },
}


def _stream_response(messages: list, label: str, color: str, stop_on_tool_call: bool = True) -> str:
    """
    Stream from ollama, printing tokens in real time. Returns accumulated content.

    When stop_on_tool_call is True, cuts the stream the moment a complete
    CALL_TOOL line is emitted — otherwise the model keeps generating and
    hallucinates the tool's result instead of waiting for the real data we feed
    back next turn. Set False for the final thesis pass, where we want the full
    write-up and no tool calls are expected.
    """
    stream = ollama.chat(
        model=LOCAL_MODEL,
        messages=messages,
        options={"temperature": 0.4, "num_predict": 700, "num_ctx": 8192},
        think=False,
        stream=True,
    )
    content = ""
    print(f"\n{color}{BOLD}[{label}]{RESET} ", end="", flush=True)
    for chunk in stream:
        if hasattr(chunk, "message"):
            token = getattr(chunk.message, "content", "") or ""
        else:
            token = chunk.get("message", {}).get("content", "") or ""
        if token:
            print(token, end="", flush=True)
            content += token
            # Once the tool name is fully written (followed by any non-word char),
            # cut the stream so the model can't fabricate the result.
            if stop_on_tool_call and re.search(r"CALL_TOOL:\s*\w+\W", content):
                break
    print()
    return content.strip()


def _strip_tool_calls(text: str) -> str:
    """Remove any stray CALL_TOOL lines so a thesis never ends as a tool stub."""
    return re.sub(r"^.*CALL_TOOL:\s*\w+.*$", "", text, flags=re.MULTILINE).strip()


def _parse_tool_call(text: str) -> str | None:
    match = re.search(r"CALL_TOOL:\s*(\w+)", text)
    if match:
        name = match.group(1).strip()
        return name if name in TOOL_REGISTRY else None
    return None


def _run_research_loop(ticker: str, role: str, color: str) -> str:
    label = f"{role.upper()} ANALYST"
    tool_menu = "\n".join(
        f"  - {name}: {info['desc']}" for name, info in TOOL_REGISTRY.items()
    )

    if role == "bull":
        mandate = "find the strongest bullish signals: growth momentum, undervaluation, strong margins, positive catalysts"
        constraint = "Do NOT acknowledge bear risks."
    else:
        mandate = "find the strongest bearish signals: overvaluation, margin compression, debt burden, negative news"
        constraint = "Do NOT acknowledge bull positives."

    system = {
        "role": "system",
        "content": (
            f"You are an isolated {role.title()} Research Analyst for {ticker}.\n"
            f"Mandate: {mandate}.\n\n"
            f"Available tools:\n{tool_menu}\n\n"
            f"TOOL CALL FORMAT — output this exact pattern on its own line to call a tool:\n"
            f"CALL_TOOL: <tool_name>\n\n"
            f"Once you have enough data, write your bullet-point thesis (max 250 words) "
            f"then end with ANALYSIS_COMPLETE on its own line.\n"
            f"{constraint}"
        ),
    }

    messages = [
        system,
        {"role": "user", "content": f"Begin research on {ticker}. Pick your first tool."},
    ]

    # Research phase: up to MAX_TOOL_CALLS tool calls. The moment the model
    # stops calling tools and writes prose, that prose IS its thesis — return it.
    for _ in range(MAX_TOOL_CALLS):
        response = _stream_response(messages, label, color)
        messages.append({"role": "assistant", "content": response})

        tool_name = _parse_tool_call(response)
        if not tool_name:
            return _strip_tool_calls(response)

        print(f"{GRAY}  ↳ Calling: {BOLD}{tool_name}{RESET}")
        result = TOOL_REGISTRY[tool_name]["fn"](ticker)
        print(f"{GRAY}{result}{RESET}")
        print(f"{GRAY}{'─' * 40}{RESET}")
        messages.append({
            "role": "user",
            "content": (
                f"TOOL RESULT ({tool_name}):\n{result}\n\n"
                f"Call another tool or write your final thesis ending with ANALYSIS_COMPLETE."
            ),
        })

    # Tool budget exhausted — force a final thesis pass with tools disabled so
    # the analyst always produces a real write-up, never ends on a tool stub.
    messages.append({
        "role": "user",
        "content": (
            "You have gathered enough data. Do NOT call any more tools. "
            "Write your final bullet-point thesis now (max 250 words) and end "
            "with ANALYSIS_COMPLETE."
        ),
    })
    final = _stream_response(messages, label, color, stop_on_tool_call=False)
    return _strip_tool_calls(final)


def bull_node(state: AgentState) -> dict:
    print(f"\n{GREEN}{'═' * 50}{RESET}")
    print(f"{GREEN}{BOLD}  BULL ANALYST — {state['ticker']}{RESET}")
    print(f"{GREEN}{'═' * 50}{RESET}")
    report = _run_research_loop(state["ticker"], "bull", GREEN)
    return {"bull_report": report}


def bear_node(state: AgentState) -> dict:
    print(f"\n{RED}{'═' * 50}{RESET}")
    print(f"{RED}{BOLD}  BEAR ANALYST — {state['ticker']}{RESET}")
    print(f"{RED}{'═' * 50}{RESET}")
    report = _run_research_loop(state["ticker"], "bear", RED)
    return {"bear_report": report}


def judge_node(state: AgentState) -> dict:
    print(f"\n{CYAN}{'═' * 50}{RESET}")
    print(f"{CYAN}{BOLD}  PORTFOLIO MANAGER ARBITRATING...{RESET}")
    print(f"{CYAN}{'═' * 50}{RESET}")

    llm = ChatOpenAI(model=CLOUD_MODEL, temperature=0.1)
    structured_llm = llm.with_structured_output(TradingVerdict)

    messages = [
        SystemMessage(content=(
            "You are the Head Portfolio Manager running a non-cooperative trading committee. "
            "You receive two adversarial briefs — one hyper-bull, one hyper-bear — on the same asset. "
            "Cross-examine their data points, identify logical leaps, weigh the evidence, "
            "and deliver a definitive verdict. Populate every field of the output schema precisely."
        )),
        HumanMessage(content=(
            f"ASSET: {state['ticker']}\n\n"
            f"{'='*40}\nBULL DISPATCH:\n{'='*40}\n{state['bull_report']}\n\n"
            f"{'='*40}\nBEAR DISPATCH:\n{'='*40}\n{state['bear_report']}\n\n"
            f"Arbitrate and return your structured verdict."
        )),
    ]

    verdict: TradingVerdict = structured_llm.invoke(messages)
    return {"final_verdict": verdict}
