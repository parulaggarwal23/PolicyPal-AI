from fastapi import FastAPI
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "codellama:latest"


@app.get("/")
def home():
    return {
        "service": "PolicyPal LLM Service",
        "status": "running"
    }


@app.post("/generate")
def generate(prompt: str):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return {
        "response": data["response"]
    }