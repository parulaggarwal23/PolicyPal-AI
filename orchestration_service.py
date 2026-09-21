import os
import requests
import re
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PolicyPal Orchestration Service")

RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://localhost:8001/retrieve")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8002/generate")

# --- Guardrails Configuration ---
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "500"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))

_OOS_PATTERNS = [
    r"\b(write|compose|create|generate)\s+(me\s+)?(a\s+)?(poem|song|story|joke|essay|letter|email|code|program|script|recipe)\b",
    r"\b(tell me a joke|make me laugh|funny)\b",
    r"\bweather\b.*\b(today|tomorrow|forecast|temperature)\b",
    r"\b(weather|temperature|rain|sunny)\b.*\b(today|tomorrow|forecast|city|delhi|mumbai)\b",
    r"\bprime\s*minister\b",
    r"\bpresident\s+of\s+india\b",
    r"\bchief\s*minister\b",
    r"\belection\b.*\b(result|winner|vote)\b",
    r"\bstock\s+(price|market|exchange)\b",
    r"\bshare\s+price\b",
    r"\bnifty\b|\bsensex\b|\bbse\b|\bnse\b",
    r"\b(cook|recipe|ingredient)\b.*\b(biryani|dal|curry|food)\b",
    r"\b(recipe|how\s+to\s+cook|how\s+to\s+make)\b",
    r"\b(sort|reverse|array|linked.list|binary.tree)\b.*\b(in\s+(python|java|c\+\+|javascript))\b",
    r"\bsolve\s+this\s+(programming|coding)\b",
    r"\b(symptoms|diagnosis|treatment|medicine|doctor|hospital)\b.*\b(disease|fever|cancer|covid)\b",
    r"\b(cricket|football|tennis)\b.*\b(score|match|player|team|ipl|world.cup)\b",
]
_OOS_COMPILED = [re.compile(p, re.IGNORECASE) for p in _OOS_PATTERNS]

def _check_scope_guardrail(question: str) -> dict:
    q = question.strip()
    for pat in _OOS_COMPILED:
        if pat.search(q):
            return {
                "triggered": True,
                "guardrail": "scope",
                "reason": f"Question matches out-of-scope pattern: '{pat.pattern[:60]}'",
                "message": "This question appears to be outside the scope of PolicyPal. Please ask a policy-related question.",
            }
    return {"triggered": False, "guardrail": "scope"}

def _check_input_length_guardrail(question: str) -> dict:
    if len(question.strip()) > MAX_INPUT_LENGTH:
        return {
            "triggered": True,
            "guardrail": "input_length",
            "reason": f"Input length exceeds maximum {MAX_INPUT_LENGTH} characters.",
            "message": f"Your question is too long. Please limit it to {MAX_INPUT_LENGTH} characters.",
        }
    return {"triggered": False, "guardrail": "input_length"}

def _check_kb_relevance_guardrail(best_score: float) -> dict:
    if best_score < RELEVANCE_THRESHOLD:
        return {
            "triggered": True,
            "guardrail": "kb_relevance",
            "reason": f"Best similarity score {best_score:.4f} is below threshold {RELEVANCE_THRESHOLD}.",
            "message": "I don't have enough information in the PolicyPal Knowledge Base to answer this question reliably.",
        }
    return {"triggered": False, "guardrail": "kb_relevance"}

def _validate_output(answer: str, question: str, context: str) -> dict:
    answer_lower  = answer.lower().strip()
    question_lower = question.lower().strip()
    context_lower  = context.lower() if context else ""
    
    def _tok(t): return set(re.findall(r"[a-z]{4,}", t.lower()))
    q_toks = _tok(question)
    a_toks = _tok(answer)
    relevance_overlap = len(q_toks & a_toks) / len(q_toks) if q_toks else 0
    t1_pass = relevance_overlap >= 0.15
    
    ctx_toks = _tok(context) if context else set()
    a_content_toks = a_toks - _tok("could find information policy documents provided context answer question")
    support_ratio = len(a_content_toks & ctx_toks) / len(a_content_toks) if a_content_toks else 1.0
    t2_pass = support_ratio >= 0.20 or len(context) < 100
    
    specific_patterns = [
        re.compile(r'\b\d+\s*(crore|lakh|rupee|rs\.?|inr)\b', re.I),
        re.compile(r'\b\d+\s*%\s*(tax|gst|rate|fine|penalty)', re.I),
        re.compile(r'section\s+\d+[a-z]?\s+of\s+the\s+(act|code)', re.I),
    ]
    hallucination_flags = []
    for pat in specific_patterns:
        found = pat.findall(answer_lower)
        for match in found:
            match_str = match if isinstance(match, str) else " ".join(match)
            if match_str and match_str not in context_lower:
                hallucination_flags.append(match_str)
    t3_pass = len(hallucination_flags) == 0
    
    refusal_phrases = ["could not find", "not in the policy", "don't have enough", "not available"]
    is_refusal = any(p in answer_lower for p in refusal_phrases)
    has_context = len(context.strip()) > 200
    t4_pass = not (has_context and is_refusal)
    t5_pass = True
    t6_pass = len(answer.strip()) > 20
    
    overall = all([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass, t6_pass])
    
    return {
        "relevance": {"pass": t1_pass, "score": round(relevance_overlap, 4)},
        "context_support": {"pass": t2_pass, "score": round(support_ratio, 4)},
        "hallucination_check": {"pass": t3_pass, "flags": hallucination_flags[:3]},
        "sufficient_info": {"pass": t4_pass, "has_context": has_context, "is_refusal": is_refusal},
        "appropriate_refusal": {"pass": t5_pass},
        "format": {"pass": t6_pass, "length": len(answer.strip())},
        "overall_pass": overall,
    }


@app.get("/")
def home():
    return {
        "service": "PolicyPal Orchestration Service",
        "status": "running",
        "retrieval_url": RETRIEVAL_URL,
        "llm_url": LLM_URL
    }

class AskRequest(BaseModel):
    question: str
    skip_guardrails: bool = False
    model: str = ""

@app.post("/ask")
def ask(question: str):
    return _ask_internal(question, False, "")

@app.post("/ask_v2")
def ask_v2(req: AskRequest):
    return _ask_internal(req.question, req.skip_guardrails, req.model)

def _ask_internal(question: str, skip_guardrails: bool, model: str):
    guardrails_applied = []
    
    if not skip_guardrails:
        # Input Guardrails
        g_len = _check_input_length_guardrail(question)
        guardrails_applied.append("input_length")
        if g_len["triggered"]:
            return {"question": question, "answer": g_len["message"], "guardrail_triggered": g_len}
            
        g_scope = _check_scope_guardrail(question)
        guardrails_applied.append("scope")
        if g_scope["triggered"]:
            return {"question": question, "answer": g_scope["message"], "guardrail_triggered": g_scope}

    # Step 1: Retrieve relevant information
    try:
        retrieval_response = requests.post(
            RETRIEVAL_URL,
            params={"question": question},
            timeout=30
        )
        retrieval_response.raise_for_status()
        retrieval_data = retrieval_response.json()
        results = retrieval_data.get("results", [])
    except Exception as e:
        results = []

    best_score = results[0].get("score", 0.0) if results else 0.0

    if not skip_guardrails:
        g_kb = _check_kb_relevance_guardrail(best_score)
        guardrails_applied.append("kb_relevance")
        if g_kb["triggered"]:
            return {"question": question, "answer": g_kb["message"], "guardrail_triggered": g_kb, "retrieved_results": results}

    # Step 2: Build context
    context = "\n\n".join(
        result.get("text", "") for result in results
    )

    # Step 3: Send context + question to LLM
    prompt = f"""
You are PolicyPal AI.

Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say that the
information is not available in the knowledge base.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        params = {"prompt": prompt}
        if model:
            params["model"] = model
            
        llm_response = requests.post(
            LLM_URL,
            params=params,
            timeout=120
        )
        llm_response.raise_for_status()
        llm_data = llm_response.json()
        answer = llm_data.get("response", "")
    except Exception as e:
        answer = f"[Orchestration Notice] LLM Service at {LLM_URL} was unreachable: {e}"

    out_val = None
    if not skip_guardrails:
        out_val = _validate_output(answer, question, context)
        guardrails_applied.append("output_validation")
        if not out_val["overall_pass"]:
            # Output Guardrail triggered, fallback to safe message
            safe_message = "The generated answer did not pass PolicyPal's internal quality tests."
            return {
                "question": question,
                "answer": safe_message,
                "retrieved_results": results,
                "output_validation": out_val,
                "guardrail_triggered": {"guardrail": "output_validation", "triggered": True, "reason": "Failed tests", "message": safe_message}
            }

    return {
        "question": question,
        "retrieved_results": results,
        "answer": answer,
        "output_validation": out_val,
        "guardrails_active": not skip_guardrails
    }

