# MMR (Max Marginal Relevance): diversifies retrieved chunks before passing to the LLM.
# Steps: 1) embed query + chunks, 2) score each chunk by relevance to query,
# 3) greedily pick top_k chunks — each pick balances relevance vs similarity to already-selected chunks.
from math import inf

import numpy as np

from rag.constants import MODEL


def cosine_similarity(a, b):
    # Measures how similar two embeddings are. Returns 1.0 (identical) to -1.0 (opposite).
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def mmr(
    query: str,
    embedded_chunks: list[list[float]],  # pre-computed embeddings for each chunk
    chunks: list[str],  # the actual chunk texts, same order as embedded_chunks
    top_k: int,  # how many chunks to return
    lambda_var: float,  # 1.0 = pure relevance, 0.0 = pure diversity, 0.7 recommended
) -> list[str]:
    selected = []  # indices of chunks we've picked so far

    # Step 1: compute relevance score of each chunk to the query (cosine similarity)
    query_relevance_matrix = []
    query_embedding = MODEL.encode(query)
    for embedded_chunk in embedded_chunks:
        query_relevance_matrix.append(
            cosine_similarity(query_embedding, embedded_chunk)
        )

    # Step 2: greedily pick top_k chunks, balancing relevance and diversity
    for k in range(0, top_k):
        best_score = -inf
        best_idx = -1

        if not len(selected):
            # First iteration: just pick the most relevant chunk
            selected.append(np.argmax(query_relevance_matrix))

        else:
            # Subsequent iterations: score each remaining candidate by
            # MMR = lambda * relevance - (1 - lambda) * max_similarity_to_already_selected
            for i, ec in enumerate(embedded_chunks):
                if i not in selected:
                    # Find how similar this candidate is to the most similar already-selected chunk
                    max_sim = -1
                    for s in selected:
                        sim = cosine_similarity(ec, embedded_chunks[s])
                        if sim > max_sim:
                            max_sim = sim

                    # High relevance is good, high similarity to selected is bad (redundant)
                    score = (
                        lambda_var * query_relevance_matrix[i]
                        - (1 - lambda_var) * max_sim
                    )
                    if score > best_score:
                        best_score = score
                        best_idx = i

            # Add the best candidate from this iteration to selected
            selected.append(best_idx)

    # Return the actual chunk texts in the order they were selected
    return [chunks[i] for i in selected]
