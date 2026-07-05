from pathlib import Path


# loads docs from data folder
def load_docs(folder: Path) -> list[str]:

    docs = []
    for file in folder.glob("*.txt"):
        text: str = file.read_text()
        docs.append(text)

    return docs


# chunk size is in words.
def chunk(docs: list[str], chunk_size: int = 500) -> list[str]:
    chunks = []
    for doc in docs:
        words = doc.split(" ")
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            # print(chunk)
            chunks.append(chunk)

    return chunks


# runs chunking
def main():
    docs = load_docs(Path("data"))

    chunks = chunk(docs)
    print("number of chunks: ", len(chunks))


if __name__ == "__main__":
    main()
