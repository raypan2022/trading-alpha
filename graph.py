from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes import macro_node, bull_node, bear_node, judge_node


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("macro_strategist", macro_node)
    workflow.add_node("bull_researcher", bull_node)
    workflow.add_node("bear_researcher", bear_node)
    workflow.add_node("portfolio_manager", judge_node)

    # Sequential: macro → bull → bear → judge
    # Macro runs first so its regime read is shared context for both analysts
    # and the judge. Sequential keeps streaming output clean and readable;
    # parallelizing bull/bear is a v2 upgrade once output buffering is in place.
    workflow.add_edge(START, "macro_strategist")
    workflow.add_edge("macro_strategist", "bull_researcher")
    workflow.add_edge("bull_researcher", "bear_researcher")
    workflow.add_edge("bear_researcher", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()
