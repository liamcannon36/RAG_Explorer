from rag.constants import CROSS_ENCODER


def rerank(query: str, chunks: list[str]):
    scores = CROSS_ENCODER.predict([(query, chunk) for chunk in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked]


def main():
    query = "Who is Liam Cannon"
    chunks = [
        "Liam Cannon is a student at the university of california. He is a full-time writer. He is 24 and works in Providence, Rhode Island",
        "Liam Cannon's linkedin profile: https:/liamcannon/linkedin/01",
        "hi how can i help you today",
        "can you repeat the question",
    ]

    print(rerank(query, chunks))


if __name__ == "__main__":
    main()
