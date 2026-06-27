from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes import bull_node, bear_node, judge_node


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("bull_researcher", bull_node)
    workflow.add_node("bear_researcher", bear_node)
    workflow.add_node("portfolio_manager", judge_node)

    # Fan out from START — bull and bear run in parallel
    workflow.add_edge(START, "bull_researcher")
    workflow.add_edge(START, "bear_researcher")

    # Both converge at the judge (LangGraph waits for both before proceeding)
    workflow.add_edge("bull_researcher", "portfolio_manager")
    workflow.add_edge("bear_researcher", "portfolio_manager")

    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()
