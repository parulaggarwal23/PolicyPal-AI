import json
import os
import requests
import math
import re

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_URL = os.getenv("EMBEDDING_URL") or (OLLAMA_BASE if OLLAMA_BASE.endswith("/api/embeddings") else f"{OLLAMA_BASE.rstrip('/')}/api/embeddings")
GENERATE_URL = os.getenv("GENERATE_URL") or (OLLAMA_BASE if OLLAMA_BASE.endswith("/api/generate") else f"{OLLAMA_BASE.rstrip('/')}/api/generate")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "codellama:latest")

with open("embeddings.json", "r", encoding="utf-8") as f:
    documents = json.load(f)


def get_embedding(text):
    try:
        response = requests.post(
            EMBEDDING_URL,
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception:
        return None


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def _keyword_similarity(query: str, doc_text: str, doc_source: str = "") -> float:
    stop = {"what", "which", "when", "where", "who", "whom", "this", "that", "these", "those", "have", "has", "had", "does", "from", "with", "about"}
    words = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in stop]
    if not words:
        return 0.2
    target = f"{doc_source} {doc_text}".lower()
    matches = sum(1 for w in words if w in target)
    return round(0.40 + 0.45 * (matches / len(words)), 6) if matches > 0 else 0.20


def retrieve(question, top_k=3):
    query_embedding = get_embedding(question)

    scored_documents = []

    if query_embedding:
        for document in documents:
            score = cosine_similarity(
                query_embedding,
                document["embedding"]
            )
            scored_documents.append({
                "score": score,
                "source": document["source"],
                "text": document["text"]
            })
    else:
        for document in documents:
            score = _keyword_similarity(question, document["text"], document.get("source", ""))
            scored_documents.append({
                "score": score,
                "source": document["source"],
                "text": document["text"]
            })

    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_documents[:top_k]


def generate_answer(question, results):
    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are PolicyPal AI, a policy information assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:
"I could not find this information in the provided policy documents."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            GENERATE_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"[PolicyPal Notice] Unable to connect to Ollama at {GENERATE_URL}: {e}.\nTop context was successfully retrieved."


if __name__ == "__main__":
    question = input("Ask PolicyPal: ")
    results = retrieve(question)

    print("\nRetrieved Context:")
    print("------------------")
    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['source']} "
            f"(similarity: {result['score']:.4f})"
        )

    answer = generate_answer(question, results)
    print("\nPolicyPal AI Response:")
    print("----------------------")
    print(answer)