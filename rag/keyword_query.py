import math
from pathlib import Path

import spacy

EPSILON = 1e-12
K1 = 1.5
B = 0.75
# Spacy handles
# 1. Tokenization: splits text into individual tokens
# 2. POS tagging: identifies part of speech (noun, verb, etc.)
# 3. Lemmatization:  reduces each token to its base form
# 4. NER: identifies named entities (people, orgs, places)
# 5. Stop word detection: marks common words like "the", "is", "who"
#
#
# # Step 1 — Compute IDF for each unique term (once, at index time):
# IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
# - N = total number of chunks
# - df = number of chunks containing the term

# Step 2 — For each chunk, compute BM25 score by summing over all query terms:
# BM25(query, chunk) = Σ IDF(term) * (TF(term, chunk) * (k1 + 1)) / (TF(term, chunk) + k1 * (1 - b + b * len(chunk) / avg_chunk_len))
# - TF(term, chunk) = count of term in that chunk
# - len(chunk) = number of tokens in that chunk
# - avg_chunk_len = average token count across all chunks
# - k1 = 1.5, b = 0.75

# Step 3 — Rank chunks by their BM25 score, return top-k.
#


nlp = spacy.load("en_core_web_sm")
from dotenv import load_dotenv

from rag.chunk import chunk, load_docs
from rag.embed import embed


def keyword_query(user_query: str, chunks: list[str], topk: int = 5):
    tokenized_chunks = [preprocess(chunk) for chunk in chunks]

    tokenized_query = preprocess(user_query)

    unique_terms = list(set(tokenized_query))

    chunk_score = []

    terms_idf_scores = {}
    for t in unique_terms:
        terms_idf_scores[t] = idf(t, tokenized_chunks)

    avg_chunk_length = sum(
        len(tokenized_chunk) for tokenized_chunk in tokenized_chunks
    ) / len(tokenized_chunks)

    for tokenized_chunk in tokenized_chunks:
        total_score = bm25_score(
            unique_terms, tokenized_chunk, terms_idf_scores, avg_chunk_length
        )
        chunk_score.append(total_score)

    ranked_indices = sorted(
        range(len(chunks)), key=lambda i: chunk_score[i], reverse=True
    )
    return ranked_indices[:topk]


def preprocess(text: str) -> list[str]:
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]


# term frequency: tF: how many times was the term in a certain chunk
def tf(term: str, tokenized_chunk: list[str]):
    return tokenized_chunk.count(term)


# BM25's IDF variant (robertson-sparck jones)
def idf(term: str, tokenized_chunks: list[list[str]]):
    N = len(tokenized_chunks)
    df_count = 0
    for tokenized_chunk in tokenized_chunks:
        if tf(term, tokenized_chunk):
            df_count += 1

    numerator = N - df_count + 0.5
    denominator = df_count + 0.5

    return max(EPSILON, math.log(numerator / denominator))


def bm25_score(
    tokenized_query: list[str],
    tokenized_chunk: list[str],
    idf_scores: dict[str, float],
    avg_chunk_len: float,
) -> float:
    chunk_len = len(tokenized_chunk)
    score = 0.0
    for term in tokenized_query:
        term_tf = tf(term, tokenized_chunk)
        if term_tf == 0:
            continue
        numerator = term_tf * (K1 + 1)
        denominator = term_tf + K1 * (1 - B + B * chunk_len / avg_chunk_len)
        score += idf_scores[term] * (numerator / denominator)
    return score


def main():
    docs = load_docs(Path("data"))
    chunks = chunk(docs)
    print(top_docs_keyword(chunks, "who is this, liam cannon")[0])


if __name__ == "__main__":
    main()
