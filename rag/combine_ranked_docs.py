# Reciprocal Rank Fusion (RRF): combines ranked results from keyword and semantic search.
# A chunk that appears highly in both lists scores better than one that tops only one list.
# k=60 is the standard RRF constant that dampens the influence of very high ranks.
def rrf(
    keyword_indices: list[int],
    semantic_indices: list[int],
    chunks: list[str],
    k: int = 60,
) -> list[str]:

    # score each chunk index from the semantic results
    semantic_scores = {}
    for rank, idx in enumerate(semantic_indices):
        semantic_scores[idx] = 1 / (k + rank + 1)

    # score each chunk index from the keyword results
    keyword_scores = {}
    for rank, idx in enumerate(keyword_indices):
        keyword_scores[idx] = 1 / (k + rank + 1)

    # combine: sum scores for each chunk that appeared in either list
    combined_scores = {}
    all_indices = set(list(semantic_scores.keys()) + list(keyword_scores.keys()))
    for idx in all_indices:
        combined_scores[idx] = semantic_scores.get(idx, 0) + keyword_scores.get(idx, 0)

    # sort chunk indices by combined score, highest first
    ranked_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)

    # return the actual chunk text in ranked order
    return [chunks[i] for i in ranked_indices]
