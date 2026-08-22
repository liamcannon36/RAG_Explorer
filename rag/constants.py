from sentence_transformers import CrossEncoder, SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

