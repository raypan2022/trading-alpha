from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes import (
    macro_node, bull_node, bear_node, debate_node, judge_node, should_continue_debate,
)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("macro_strategist", macro_node)
    workflow.add_node("bull_researcher", bull_node)
    workflow.add_node("bear_researcher", bear_node)
    workflow.add_node("debate", debate_node)
    workflow.add_node("portfolio_manager", judge_node)

    # macro → bull (research) → bear (research) → debate ⟲ → judge
    #
    # Phases 1-2 are isolated research: each analyst gathers data and forms an
    # initial thesis without seeing the other. The debate node then cycles —
    # bull and bear rebut each other — until a conditional edge stops it (round
    # cap or both sides out of new points), at which point the judge arbitrates.
    # This cycle is what makes it a real agent system rather than a one-shot DAG.
    workflow.add_edge(START, "macro_strategist")
    workflow.add_edge("macro_strategist", "bull_researcher")
    workflow.add_edge("bull_researcher", "bear_researcher")
    workflow.add_edge("bear_researcher", "debate")
    workflow.add_conditional_edges(
        "debate",
        should_continue_debate,
        {"debate": "debate", "judge": "portfolio_manager"},
    )
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()
