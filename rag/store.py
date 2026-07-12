import json
import os
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "naive-rag"
DIMENSION = 384  # all-MiniLM-L6-v2
CACHE_FILE = Path("data/chunks_cache.json")
_UPSERT_BATCH = 100


# Pinecone is an ANN (Approximate Nearest Neighbor) index. Rather than looping over all
# stored vectors and computing cosine similarity in Python, Pinecone uses HNSW to find
# the closest vectors without checking every one — much faster at scale. The similarity
# score is computed server-side at query time, not stored.
def get_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(INDEX_NAME)


def is_populated(index) -> bool:
    stats = index.describe_index_stats()
    return stats.total_vector_count > 0 and CACHE_FILE.exists()


def save(index, chunks: list[str], embeddings: list[list[float]]):
    vectors = [
        {"id": f"chunk-{i}", "values": emb, "metadata": {"text": chunks[i]}}
        for i, emb in enumerate(embeddings)
    ]
    for i in range(0, len(vectors), _UPSERT_BATCH):
        index.upsert(vectors=vectors[i : i + _UPSERT_BATCH])
    CACHE_FILE.write_text(json.dumps(chunks))
    print(f"[Store] Upserted {len(chunks)} chunks to Pinecone.")


def load_chunks() -> list[str]:
    return json.loads(CACHE_FILE.read_text())
