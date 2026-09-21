import os
import requests
from fastapi import FastAPI

app = FastAPI(title="PolicyPal LLM Service")

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_URL = os.getenv("GENERATE_URL") or (OLLAMA_BASE if OLLAMA_BASE.endswith("/api/generate") else f"{OLLAMA_BASE.rstrip('/')}/api/generate")
MODEL = os.getenv("LLM_MODEL", "codellama:latest")


@app.get("/")
def home():
    return {
        "service": "PolicyPal LLM Service",
        "status": "running",
        "model": MODEL
    }


@app.post("/generate")
def generate(prompt: str, model: str = None):
    use_model = model if model else MODEL
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": use_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return {
            "response": data.get("response", "")
        }
    except Exception as e:
        return {
            "response": f"[LLM Service Notice] Ollama is currently unreachable at {OLLAMA_URL}. Ensure Ollama is running ('ollama serve') with '{MODEL}'.",
            "error": str(e)
        }

