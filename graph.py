from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes import bull_node, bear_node, judge_node


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("bull_researcher", bull_node)
    workflow.add_node("bear_researcher", bear_node)
    workflow.add_node("portfolio_manager", judge_node)

    # Sequential: bull → bear → judge
    # Keeps terminal output clean and readable during streaming.
    # Parallel execution is a v2 upgrade once output buffering is in place.
    workflow.add_edge(START, "bull_researcher")
    workflow.add_edge("bull_researcher", "bear_researcher")
    workflow.add_edge("bear_researcher", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()
