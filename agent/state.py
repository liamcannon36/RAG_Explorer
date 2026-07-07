from typing import Literal, TypedDict


class State(TypedDict):
    """State passed between nodes in the agentic RAG graph."""

    user_question: str | None  # the raw user input
    returned_docs: list[str] | None  # chunks retrieved from RAG
    current_agent_response: str | None  # the final response to the user
    routing_decision: Literal["RAG", "General"]  # whether to use RAG or answer directly
    rag_retry_count: int  # number of times RAG has been retried
    answer_sufficient: bool | None
