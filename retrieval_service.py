import os
import json
import requests
import math
import re
from fastapi import FastAPI

app = FastAPI(title="PolicyPal Retrieval Service")

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_URL = os.getenv("EMBEDDING_URL") or (OLLAMA_BASE if OLLAMA_BASE.endswith("/api/embeddings") else f"{OLLAMA_BASE.rstrip('/')}/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

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
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def _keyword_similarity(query: str, doc_text: str, doc_source: str = "") -> float:
    stop = {"what", "which", "when", "where", "who", "whom", "this", "that", "these", "those", "have", "has", "had", "does", "from", "with", "about"}
    words = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in stop]
    if not words:
        return 0.2
    target = f"{doc_source} {doc_text}".lower()
    matches = sum(1 for w in words if w in target)
    return round(0.40 + 0.45 * (matches / len(words)), 6) if matches > 0 else 0.20


@app.get("/")
def home():
    return {"service": "PolicyPal Retrieval Service", "status": "running", "document_count": len(documents)}


@app.post("/retrieve")
def retrieve(question: str):
    query_embedding = get_embedding(question)

    results = []

    if query_embedding:
        for document in documents:
            score = cosine_similarity(
                query_embedding,
                document["embedding"]
            )
            results.append({
                "score": score,
                "source": document["source"],
                "text": document["text"]
            })
    else:
        for document in documents:
            score = _keyword_similarity(question, document["text"], document.get("source", ""))
            results.append({
                "score": score,
                "source": document["source"],
                "text": document["text"]
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "question": question,
        "results": results[:3]
    }