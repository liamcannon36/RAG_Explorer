import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.constants import MODEL


def query(user_question: str, chunks: list[list[float]], docs: list[str], topk: int):

    # embed user_question
    query_embedding: list[float] = MODEL.encode(user_question).tolist()

    distances = cosine_similarity([query_embedding], chunks)

    indices = np.argsort(distances[0])[::-1][:topk]
    top_docs = [docs[i] for i in indices]
    # print([distances[0][i] for i in indices])

    return top_docs
