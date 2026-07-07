# router → retrieve → answer -> check relevance → final answer answer
#                           ↓ (not relevant)
#                     retrieve again (with rephrased query)

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from agent.nodes import answer, check_answer, make_retrieve, router
from agent.state import State


def build_graph(chunks, embedded_chunks) -> CompiledStateGraph:
    workflow = StateGraph(State)
    workflow.add_node("Answer", answer)
    workflow.add_node("Check_answer", check_answer)
    workflow.add_node("Router", router)

    # Build retrieve node from context)
    retrieve_node = make_retrieve(chunks, embedded_chunks)
    workflow.add_node("Retrieve", retrieve_node)

    workflow.add_edge(START, "Router")
    workflow.add_conditional_edges(
        "Router",
        lambda state: state["routing_decision"],
        {"RAG": "Retrieve", "General": "Answer"},
    )

    workflow.add_edge("Retrieve", "Answer")


    workflow.add_conditional_edges(
        "Answer",
        lambda state: state["routing_decision"],
        {"RAG": "Check_answer", "General": END},
    )

    workflow.add_conditional_edges(
        "Check_answer",
        lambda state: (
            "end"
            if state["answer_sufficient"] or state["rag_retry_count"] >= 3
            else "retrieve"
        ),
        {"end": END, "retrieve": "Retrieve"},
    )
    return workflow.compile()
