from rag.constants import MODEL


def semantic_query(user_question: str, index, topk: int = 5) -> list[int]:
    query_embedding: list[float] = MODEL.encode(user_question).tolist()
    results = index.query(vector=query_embedding, top_k=topk, include_metadata=False)
    return [int(m.id.split("-")[1]) for m in results.matches]
