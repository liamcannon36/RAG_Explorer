import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.constants import MODEL


def semantic_query(
    user_question: str, embedded_chunks: list[list[float]], docs: list[str], topk: int
):

    # embed user_question
    query_embedding: list[float] = MODEL.encode(user_question).tolist()

    distances = cosine_similarity([query_embedding], embedded_chunks)

    indices = np.argsort(distances[0])[::-1][:topk]
    return indices.tolist()
