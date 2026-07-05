import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rag.chunk import chunk, load_docs
from rag.constants import MODEL
from rag.embed import embed
from rag.query import query

# Load variables from the .env file
load_dotenv()


def main():

    client = anthropic.Anthropic()

    # rag pipeline
    docs = load_docs(Path("data"))
    chunks = chunk(docs)
    embedded_chunks = embed(chunks)
    topk = 5

    while True:
        user_input = input("you: ").strip()
        if user_input.lower() in ("quit", "q"):
            break

        top_docs = query(user_input, embedded_chunks, chunks, topk)

        print("Most relevant chunk: ", top_docs[0])
        context = "\n\n".join(top_docs)

        enriched_prompt = [
            {
                "role": "user",
                "content": f"Context: {context} \n Query: {user_input}",
            }
        ]

        message = client.messages.create(
            model="claude-sonnet-4-5",  # Use your target model
            max_tokens=300,
            messages=enriched_prompt,
            system="Use the Context to answer the user query if applicable.",
        )

        print(message.content[0].text)


if __name__ == "__main__":
    main()
