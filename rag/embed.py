from rag.constants import MODEL


def embed(chunks: list[str]):
    return MODEL.encode(chunks).tolist()
