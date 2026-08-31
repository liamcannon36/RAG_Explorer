import anthropic

from agent.state import State
from rag.combine_ranked_docs import rrf
from rag.constants import MODEL
from rag.keyword_query import keyword_query
from rag.mmr import mmr
from rag.rerank import rerank
from rag.semantic_query import semantic_query

# router -> decides rag or general (changes state to match that)

# if RAG:
#     Retrieval


# answer: if came from RAG, uses returned docs + user question. If from general, just uses question

# check_answer:


def router(state: State):
    print(f"\n[Router] Question: {state['user_question']}")
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"Given this user query: {state['user_question']} \n Can this be answered just using your general knowledge, or does it need additional context? If it can be answered from your general knowledge, reply with only 'General'. Else, reply with 'RAG' ",
            }
        ],
    )

    # Makes it more likely to say just RAG or general if you grab the first word of the response.
    decision = message.content[0].text.strip().split()[0]
    print(f"[Router] Decision: {decision}")
    return {"routing_decision": decision}


def answer(state: State):
    print(
        f"\n[Answer] Generating response via {'RAG' if state['routing_decision'] == 'RAG' else 'general knowledge'}"
    )
    if state["routing_decision"] == "RAG":
        content = (
            f"Context: {state['returned_docs']} \n User Query: {state['user_question']}"
        )
        system = "Use the Context to answer the user query if applicable."
    else:
        content = state["user_question"]
        system = None

    client = anthropic.Anthropic()
    if system:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": content}],
            system=system,
        )
    else:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": content}],
        )
    return {"current_agent_response": message.content[0].text.strip()}


def make_retrieve(chunks: list[str], index):
    def retrieve(state: State):
        print(
            f"\n[Retrieve] Attempt #{state['rag_retry_count'] + 1} for: {state['user_question']}"
        )
        if state["rephrased_question"]:
            semantic_indices = semantic_query(
                state["rephrased_question"], index, topk=5
            )
            keyword_indices = keyword_query(state["rephrased_question"], chunks, topk=5)
        else:
            semantic_indices = semantic_query(state["user_question"], index, topk=5)
            keyword_indices = keyword_query(state["user_question"], chunks, topk=5)

        print("\n[Retrieve] Keyword ranking (BM25):")
        for rank, idx in enumerate(keyword_indices):
            preview = chunks[idx][:80].replace("\n", " ")
            print(f"  {rank + 1}. chunk[{idx}]: {preview!r}")

        print("\n[Retrieve] Semantic ranking (cosine similarity):")
        for rank, idx in enumerate(semantic_indices):
            preview = chunks[idx][:80].replace("\n", " ")
            print(f"  {rank + 1}. chunk[{idx}]: {preview!r}")

        top_combined_docs = rrf(keyword_indices, semantic_indices, chunks=chunks, k=60)

        reranked_docs = rerank(state["user_question"], top_combined_docs)[:5]

        reranked_embedded_docs = MODEL.encode(reranked_docs).tolist()

        docs = mmr(
            state["user_question"],
            reranked_embedded_docs,
            reranked_docs,
            top_k=5,
            lambda_var=0.5,
        )

        print("\n[Retrieve] Combined ranking (RRF):")
        for rank, doc in enumerate(docs):
            preview = doc[:80].replace("\n", " ")
            print(f"  {rank + 1}. {preview!r}")

        return {
            "returned_docs": docs,
            "rag_retry_count": state["rag_retry_count"] + 1,
        }

    return retrieve


def check_answer(state: State):
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"Determine if this answer: {state['current_agent_response']} sufficiently answers the user question: {state['user_question']}. If so, return only the word 'True'. If not, return only the word 'False'.",
            }
        ],
    )

    is_sufficient = message.content[0].text.strip() == "True"
    print(
        f"[Check Answer] Sufficient: {is_sufficient} (retry count: {state['rag_retry_count']})"
    )
    return {"answer_sufficient": is_sufficient}


def rephrase_question(state: State):
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"""The following question was asked: {state["user_question"]}.
                 The following answer was retrieved but was insufficient: {state["current_agent_response"]}.
                 Rephrase the original question to be more specific or approach it from a different angle, so that a document retrieval system can
                 find better information to answer it. Return only the rephrased question, nothing else.""",
            }
        ],
    )
    return {"rephrased_question": message.content[0].text.strip()}
