import sys

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, LLM_MODEL, TOP_K

PROMPT_TEMPLATE = """You are an assistant for answering questions about bank documents.
Answer ONLY using the context below. If the answer is not in the context,
say that you don't know. Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""


def retrieve(question, model, collection):
    q_emb = model.encode([f"query: {question}"], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=TOP_K)
    return result["documents"][0], result["metadatas"][0]


def answer(question, model, collection):
    docs, metas = retrieve(question, model, collection)
    context = "\n\n---\n\n".join(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"], metas


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = None

    while True:
        q = questions[0] if questions else input("\nسؤال (exit برای خروج): ")
        if q.strip().lower() == "exit":
            break
        text, metas = answer(q, model, collection)
        print("\nپاسخ:", text)
        print("منابع:", sorted({m["source"] for m in metas}))
        if questions:
            break


if __name__ == "__main__":
    main()