from pathlib import Path

from dotenv import load_dotenv

from agent.graph import build_graph
from rag.chunk import chunk, load_docs
from rag.embed import embed
from rag.store import get_index, is_populated, load_chunks, save

load_dotenv()


def main():
    index = get_index()

    if is_populated(index):
        chunks = load_chunks()
        print(f"[Store] Loaded {len(chunks)} chunks from cache.")
    else:
        docs = load_docs(Path("data"))
        chunks = chunk(docs)
        embeddings = embed(chunks)
        save(index, chunks, embeddings)

    graph = build_graph(chunks, index)

    with open("graph.md", "w") as f:
        f.write("```mermaid\n")
        f.write(graph.get_graph().draw_mermaid())
        f.write("\n```")

    while True:
        user_input = input("you: ").strip()
        if user_input.lower() in ("quit", "q"):
            break

        result = graph.invoke(
            {
                "user_question": user_input,
                "returned_docs": None,
                "current_agent_response": None,
                "routing_decision": None,
                "rag_retry_count": 0,
                "answer_sufficient": None,
            }
        )

        print(f"agent: {result['current_agent_response']}\n")


if __name__ == "__main__":
    main()
