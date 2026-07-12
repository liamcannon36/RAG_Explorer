from pathlib import Path

import spacy

nlp = spacy.load("en_core_web_sm")


def load_docs(folder: Path) -> list[str]:
    docs = []
    for file in folder.glob("*.txt"):
        text: str = file.read_text()
        docs.append(text)
    return docs


# Chunk by sentences, but limited roughly by word size of the chunk. overlap = 2 sentences, in order to avoid cutoff in semantic search.
def chunk(docs: list[str], chunk_size: int = 500, overlap: int = 2) -> list[str]:
    chunks = []
    for doc in docs:
        sentences = [sent.text.strip() for sent in nlp(doc).sents if sent.text.strip()]

        current_chunk: list[str] = []
        current_word_count = 0

        # iterate through sentences
        for sent in sentences:
            sent_word_count = len(sent.split())
            if current_word_count + sent_word_count > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sent)
            current_word_count += sent_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks


def main():
    docs = load_docs(Path("data"))
    chunks = chunk(docs)
    print(f"{len(chunks)} chunks")


if __name__ == "__main__":
    main()
