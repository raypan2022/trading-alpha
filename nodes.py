import re
from difflib import SequenceMatcher

import ollama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState, TradingVerdict
from tools import get_price_snapshot, get_fundamentals, get_recent_news
from sources.market import get_market_regime

LOCAL_MODEL = "qwen3.5:9b"
CLOUD_MODEL = "gpt-5.4-mini"
MAX_TOOL_CALLS = 3
MAX_DEBATE_ROUNDS = 3          # hard cap on bull<->bear rebuttal rounds
NO_REBUTTAL = "NO_FURTHER_REBUTTAL"
DEBATE_SIMILARITY_STOP = 0.75  # if a side's new rebuttal ~repeats its last, converge

# ANSI colors
GREEN   = "\033[92m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
GRAY    = "\033[90m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

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


def _run_research_loop(ticker: str, role: str, color: str, as_of: str | None = None,
                       market_regime: str = "") -> str:
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

    regime_block = ""
    if market_regime:
        regime_block = (
            f"SHARED MARKET CONTEXT (top-down macro read all analysts share):\n"
            f"{market_regime}\n\n"
            f"Factor this regime into your thesis — e.g. a risk-off regime raises the bar "
            f"for a BUY, a risk-on regime cuts slack to bearish caution.\n\n"
        )

    system = {
        "role": "system",
        "content": (
            f"You are an isolated {role.title()} Research Analyst for {ticker}.\n"
            f"Mandate: {mandate}.\n\n"
            f"{regime_block}"
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
        result = TOOL_REGISTRY[tool_name]["fn"](ticker, as_of)
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


def macro_node(state: AgentState) -> dict:
    """Top-down macro context, computed once and shared with both analysts and the judge."""
    print(f"\n{MAGENTA}{'═' * 50}{RESET}")
    print(f"{MAGENTA}{BOLD}  MACRO STRATEGIST — market regime{RESET}")
    print(f"{MAGENTA}{'═' * 50}{RESET}")

    regime = get_market_regime(as_of=state.get("as_of"))
    print(f"{GRAY}{regime}{RESET}")
    print(f"{GRAY}{'─' * 40}{RESET}")

    messages = [
        {"role": "system", "content": (
            "You are a macro market strategist. Given the regime signals below, in 3-4 "
            "sentences give a decisive read: what regime are we in, and what does it mean "
            "for taking risk in individual equities right now? Do NOT discuss any specific "
            "stock — this is top-down context only."
        )},
        {"role": "user", "content": regime},
    ]
    assessment = _stream_response(messages, "MACRO STRATEGIST", MAGENTA, stop_on_tool_call=False)

    return {"market_regime": f"{regime}\n\nMACRO STRATEGIST READ:\n{assessment}"}


def bull_node(state: AgentState) -> dict:
    print(f"\n{GREEN}{'═' * 50}{RESET}")
    print(f"{GREEN}{BOLD}  BULL ANALYST — {state['ticker']}{RESET}")
    print(f"{GREEN}{'═' * 50}{RESET}")
    report = _run_research_loop(state["ticker"], "bull", GREEN,
                                as_of=state.get("as_of"),
                                market_regime=state.get("market_regime", ""))
    return {"bull_report": report}


def bear_node(state: AgentState) -> dict:
    print(f"\n{RED}{'═' * 50}{RESET}")
    print(f"{RED}{BOLD}  BEAR ANALYST — {state['ticker']}{RESET}")
    print(f"{RED}{'═' * 50}{RESET}")
    report = _run_research_loop(state["ticker"], "bear", RED,
                                as_of=state.get("as_of"),
                                market_regime=state.get("market_regime", ""))
    return {"bear_report": report}


def _render_transcript(transcript: list) -> str:
    return "\n\n".join(f"[{speaker}]: {text}" for speaker, text in transcript)


def _latest_of(transcript: list, speaker: str, fallback: str) -> str:
    """Most recent statement by a speaker, or the fallback (their initial thesis)."""
    items = [text for spk, text in transcript if spk == speaker]
    return items[-1] if items else fallback


def _own_points(transcript: list, speaker: str, initial: str) -> list:
    """Everything this side has already argued (initial thesis + prior rebuttals)."""
    return [initial] + [text for spk, text in transcript if spk == speaker]


def _too_similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= DEBATE_SIMILARITY_STOP


def _debate_turn(role: str, state: AgentState, opponent_claim: str,
                 own_prior_points: list, color: str) -> str:
    """
    One rebuttal. Deliberately NOT given the full transcript dump — only the
    single opponent argument to rebut and a list of its own prior points. This
    keeps a weak model from absorbing/echoing the opponent's text (the cause of
    the verbatim-copy bug), and the identity anchor is placed LAST for recency.
    """
    ticker = state["ticker"]
    opp_name = "bear" if role == "bull" else "bull"
    stance = "bullish" if role == "bull" else "bearish"
    already = "\n".join(f"- {p}" for p in own_prior_points)

    system = {
        "role": "system",
        "content": (
            f"You are the {role.title()} Analyst for {ticker}. You hold a strictly {stance} stance and must "
            f"NEVER switch sides or argue the {opp_name}'s position. Your only job is to REBUT the {opp_name}'s "
            f"latest argument using specific numbers. Raise a NEW angle each time — never restate a point you "
            f"have already made. Be sharp and brief (max 150 words). "
            f"If you genuinely have no new rebuttal, reply with ONLY: {NO_REBUTTAL}"
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"MARKET REGIME:\n{state.get('market_regime', 'N/A')}\n\n"
            f"POINTS YOU HAVE ALREADY MADE (do NOT repeat any of these):\n{already}\n\n"
            f"THE {opp_name.upper()}'S LATEST ARGUMENT — this is what you must rebut:\n\"\"\"\n{opponent_claim}\n\"\"\"\n\n"
            f"You are the {role.upper()}. Stay {stance}. Write a NEW, specific rebuttal to the {opp_name}'s "
            f"argument above — do not adopt their position. Begin your rebuttal:"
        ),
    }
    return _stream_response([system, user], f"{role.upper()} REBUTTAL", color, stop_on_tool_call=False)


def debate_node(state: AgentState) -> dict:
    """One debate round: bull rebuts the bear's latest point, then bear rebuts the bull's fresh one."""
    round_num = state.get("debate_round", 0) + 1
    print(f"\n{CYAN}{'═' * 50}{RESET}")
    print(f"{CYAN}{BOLD}  DEBATE — ROUND {round_num}{RESET}")
    print(f"{CYAN}{'═' * 50}{RESET}")

    transcript = state.get("debate_transcript", [])

    bear_latest = _latest_of(transcript, "BEAR", state["bear_report"])
    bull_rebuttal = _debate_turn(
        "bull", state, bear_latest, _own_points(transcript, "BULL", state["bull_report"]), GREEN
    )

    bear_rebuttal = _debate_turn(
        "bear", state, bull_rebuttal, _own_points(transcript, "BEAR", state["bear_report"]), RED
    )

    return {
        "debate_transcript": [("BULL", bull_rebuttal), ("BEAR", bear_rebuttal)],
        "debate_round": round_num,
    }


def should_continue_debate(state: AgentState) -> str:
    """Conditional edge: loop the debate or hand off to the judge."""
    transcript = state["debate_transcript"]

    if state.get("debate_round", 0) >= MAX_DEBATE_ROUNDS:
        return "judge"

    # Early stop: both sides explicitly out of new points this round.
    last_round = transcript[-2:]
    if last_round and all(NO_REBUTTAL in text for _, text in last_round):
        print(f"{CYAN}  [debate] both sides out of new points — ending.{RESET}")
        return "judge"

    # Convergence: this round's rebuttals nearly repeat last round's -> stop
    # rather than manufacture another stale exchange.
    if len(transcript) >= 4:
        prev_bull, prev_bear = transcript[-4][1], transcript[-3][1]
        cur_bull, cur_bear = transcript[-2][1], transcript[-1][1]
        if _too_similar(cur_bull, prev_bull) and _too_similar(cur_bear, prev_bear):
            print(f"{CYAN}  [debate] arguments converging (repetitive) — ending.{RESET}")
            return "judge"

    return "debate"


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
            f"{'='*40}\nMARKET REGIME (shared macro context):\n{'='*40}\n{state.get('market_regime', 'N/A')}\n\n"
            f"{'='*40}\nBULL DISPATCH (initial thesis):\n{'='*40}\n{state['bull_report']}\n\n"
            f"{'='*40}\nBEAR DISPATCH (initial thesis):\n{'='*40}\n{state['bear_report']}\n\n"
            f"{'='*40}\nDEBATE (rebuttals, in order):\n{'='*40}\n{_render_transcript(state.get('debate_transcript', [])) or '(no debate)'}\n\n"
            f"Weigh the single-name case against the market regime. Pay attention to which points "
            f"survived rebuttal in the debate and which were successfully knocked down. "
            f"Arbitrate and return your structured verdict."
        )),
    ]

    verdict: TradingVerdict = structured_llm.invoke(messages)
    return {"final_verdict": verdict}
