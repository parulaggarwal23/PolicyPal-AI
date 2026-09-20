"""
Week 5 Direct Evaluation — calls guardrail functions directly for instant tests,
HTTP only for tests that need an actual LLM call.
This avoids app-level timeouts while still using real Ollama models.
"""
import json, sys, time, math, re, os, requests, urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap (import app-level functions directly) ────────────────────────
sys.path.insert(0, "/home/student/PolicyPal-AI")
os.chdir("/home/student/PolicyPal-AI")

# Import guardrail helpers directly from app
import importlib
spec = importlib.util.spec_from_file_location("app_mod", "app.py")
app_mod = importlib.util.module_from_spec(spec)
# Patch FastAPI so it doesn't re-bind port
import unittest.mock as _mock
with _mock.patch("fastapi.FastAPI") as _fa:
    _fa.return_value = _mock.MagicMock()
    try:
        spec.loader.exec_module(app_mod)
    except Exception:
        pass

# Pull functions
_check_scope   = app_mod._check_scope_guardrail
_check_len     = app_mod._check_input_length_guardrail
_check_kb      = app_mod._check_kb_relevance_guardrail
_validate      = app_mod._validate_output
_retrieve      = app_mod.retrieve_top_k
THRESHOLD      = app_mod.RELEVANCE_THRESHOLD
TRAP_Q         = app_mod.TRAP_QUESTIONS
OLLAMA_URL     = app_mod.OLLAMA_URL
LLM_MODEL      = app_mod.LLM_MODEL

print(f"Guardrail functions loaded ✓  threshold={THRESHOLD}  traps={len(TRAP_Q)}")

# ── Load tests ──────────────────────────────────────────────────────────────
with open("week5_guardrail_tests.json") as f:
    ds = json.load(f)
tests = ds["tests"]
MODELS = ["qwen2.5:0.5b", "tinyllama:1.1b", "codellama:latest"]
SHORT  = {"qwen2.5:0.5b":"Qwen","tinyllama:1.1b":"TinyLlama","codellama:latest":"CodeLlama"}
PAUSE  = 1.0
LLM_TIMEOUT = 90   # seconds per Ollama call

print(f"Tests: {len(tests)} | Models: {MODELS}")
print(f"Total: {len(tests)*len(MODELS)} evaluations\n")

# ── Core logic ───────────────────────────────────────────────────────────────
def run_without_guardrails(q, model):
    """Original behaviour — no guardrails, direct to LLM."""
    t0 = time.time()
    try:
        results = _retrieve(q)
        context = "\n\n".join(f"Source: {r['source']}\n{r['text']}" for r in results)
        prompt = (f"You are PolicyPal AI, a policy information assistant.\n"
                  f"Answer the user's question using ONLY the provided context.\n"
                  f"If the answer is not in the context, say: "
                  f"\"I could not find this information in the policy documents.\"\n\n"
                  f"Context:\n{context}\n\nQuestion:\n{q}\n\nAnswer:")
        resp = requests.post(OLLAMA_URL,
                             json={"model": model, "prompt": prompt, "stream": False},
                             timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        answer  = resp.json()["response"]
        ret     = [{"filename": r["source"].split("/")[-1], "score": round(r["score"],4)} for r in results[:3]]
        ov      = _validate(answer, q, context)
        return {"answer": answer, "retrieved": ret, "latency_ms": round((time.time()-t0)*1000),
                "output_validation": ov, "error": None}
    except Exception as e:
        return {"answer": "", "retrieved": [], "latency_ms": round((time.time()-t0)*1000),
                "output_validation": None, "error": str(e)[:80]}


def run_with_guardrails(q, model):
    """Guardrail path — all 4 guardrails applied."""
    t0 = time.time()

    # G3 — input length (instant)
    g_len = _check_len(q)
    if g_len["triggered"]:
        return {"answer": g_len["message"], "guardrail": "input_length",
                "retrieved": [], "latency_ms": round((time.time()-t0)*1000),
                "output_validation": None, "error": None}

    # G1 — scope (instant)
    g_sc = _check_scope(q)
    if g_sc["triggered"]:
        return {"answer": g_sc["message"], "guardrail": "scope",
                "retrieved": [], "latency_ms": round((time.time()-t0)*1000),
                "output_validation": None, "error": None}

    # Retrieval
    try:
        results    = _retrieve(q)
        best_score = results[0]["score"] if results else 0.0
        is_trap    = q.strip().lower() in TRAP_Q
    except Exception as e:
        return {"answer": "", "guardrail": None, "retrieved": [],
                "latency_ms": round((time.time()-t0)*1000),
                "output_validation": None, "error": str(e)[:80]}

    # G2 — KB relevance (instant)
    g_kb = _check_kb(best_score, is_trap)
    if g_kb["triggered"]:
        return {"answer": g_kb["message"], "guardrail": "kb_relevance",
                "retrieved": [{"filename": r["source"].split("/")[-1], "score": round(r["score"],4)} for r in results[:3]],
                "latency_ms": round((time.time()-t0)*1000),
                "output_validation": None, "error": None}

    # LLM generation
    try:
        context = "\n\n".join(f"Source: {r['source']}\n{r['text']}" for r in results)
        prompt  = (f"You are PolicyPal AI, a policy information assistant.\n"
                   f"Answer the user's question using ONLY the provided context.\n"
                   f"If the answer is not in the context, say: "
                   f"\"I could not find this information in the policy documents.\"\n\n"
                   f"Context:\n{context}\n\nQuestion:\n{q}\n\nAnswer:")
        resp = requests.post(OLLAMA_URL,
                             json={"model": model, "prompt": prompt, "stream": False},
                             timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        answer = resp.json()["response"]
        # G4 — output validation
        ov = _validate(answer, q, context)
        ret = [{"filename": r["source"].split("/")[-1], "score": round(r["score"],4)} for r in results[:3]]
        return {"answer": answer, "guardrail": None, "retrieved": ret,
                "latency_ms": round((time.time()-t0)*1000),
                "output_validation": ov, "error": None}
    except Exception as e:
        return {"answer": "", "guardrail": None, "retrieved": [],
                "latency_ms": round((time.time()-t0)*1000),
                "output_validation": None, "error": str(e)[:80]}


# ── Main loop ─────────────────────────────────────────────────────────────────
all_results = []
done = 0
errors = 0
total = len(tests) * len(MODELS)

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"  {SHORT[model]} ({model})")
    print(f"{'='*60}")
    for t in tests:
        done += 1
        q   = t["question"]
        tid = t["id"]
        cat = t["category"]
        print(f"  [{done:03d}/{total}] {tid} [{cat}] {q[:45]}…", end=" ", flush=True)

        wo = run_without_guardrails(q, model)
        wi = run_with_guardrails(q, model)

        if wo.get("error"): errors += 1
        if wi.get("error"): errors += 1

        gt_name = wi.get("guardrail")
        ov      = wi.get("output_validation") or wo.get("output_validation") or {}
        exp_trig = t.get("expected_guardrail_trigger", False)
        act_trig = gt_name is not None
        correct  = act_trig == exp_trig

        ot = {
            "relevance":           ov.get("relevance",{}).get("pass"),
            "context_support":     ov.get("context_support",{}).get("pass"),
            "hallucination_check": ov.get("hallucination_check",{}).get("pass"),
            "sufficient_info":     ov.get("sufficient_info",{}).get("pass"),
            "appropriate_refusal": ov.get("appropriate_refusal",{}).get("pass"),
            "format":              ov.get("format",{}).get("pass"),
            "overall":             ov.get("overall_pass"),
        }
        def passes(v): return v is True or v is None
        tp = correct and all(passes(v) for v in ot.values())
        sym = "✓" if tp else "✗"
        g   = f"TRIGGERED:{gt_name}" if act_trig else "—"
        print(f"{sym} {g} correct={correct}")

        all_results.append({
            "test_id": tid, "category": cat, "question": q, "model": model,
            "expected_behavior": t["expected_behavior"],
            "expected_trigger":  exp_trig,
            "required_guardrail": t.get("required_guardrail"),
            "kb_supported": t.get("kb_supported", False),
            # Without guardrails
            "answer_without":     (wo.get("answer",""))[:300],
            "retrieved_without":  wo.get("retrieved",[]),
            "latency_without_ms": wo.get("latency_ms"),
            "error_without":      wo.get("error"),
            # With guardrails
            "answer_with":        (wi.get("answer",""))[:300],
            "guardrail_triggered": gt_name,
            "actual_trigger":     act_trig,
            "guardrail_correct":  correct,
            "retrieved_with":     wi.get("retrieved",[]),
            "latency_with_ms":    wi.get("latency_ms"),
            "error_with":         wi.get("error"),
            "output_tests":       ot,
            "test_overall_pass":  tp,
        })
        # Only pause after real LLM calls (identified by having a latency > 500ms)
        if (wo.get("latency_ms") or 0) > 500 or (wi.get("latency_ms") or 0) > 500:
            time.sleep(PAUSE)


# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(rs):
    if not rs: return {}
    n       = len(rs)
    te      = [r for r in rs if r["expected_trigger"]]
    nt      = [r for r in rs if not r["expected_trigger"]]
    cb      = sum(1 for r in te if r["actual_trigger"])
    ca      = sum(1 for r in nt if not r["actual_trigger"])
    fb      = sum(1 for r in nt if r["actual_trigger"])
    fa      = sum(1 for r in te if not r["actual_trigger"])
    g_acc   = round((cb+ca)/n*100, 1)
    ref_r   = round(cb/len(te)*100, 1) if te else None
    fb_r    = round(fb/len(nt)*100, 1) if nt else 0
    fa_r    = round(fa/len(te)*100, 1) if te else 0
    nb      = [r for r in rs if not r["actual_trigger"]]
    h_all   = rs
    h_b     = round(sum(1 for r in h_all if r["output_tests"].get("hallucination_check") is False)/n*100, 1)
    h_a     = round(sum(1 for r in nb if r["output_tests"].get("hallucination_check") is False)/len(nb)*100, 1) if nb else 0
    return {
        "total": n, "trigger_expected": len(te),
        "correctly_blocked": cb, "correctly_allowed": ca,
        "false_blocks": fb, "false_allows": fa,
        "guardrail_accuracy_pct":       g_acc,
        "appropriate_refusal_rate_pct": ref_r,
        "false_block_rate_pct": fb_r,
        "false_allow_rate_pct": fa_r,
        "hallucination_rate_before_pct": h_b,
        "hallucination_rate_after_pct":  h_a,
        "hallucination_reduction_pct":   round(h_b - h_a, 1),
        "test_pass_rate_pct": round(sum(1 for r in rs if r["test_overall_pass"])/n*100, 1),
    }

overall_m = agg(all_results)
model_m   = {m: agg([r for r in all_results if r["model"]==m]) for m in MODELS}
cats      = sorted(set(r["category"] for r in all_results))
cat_m     = {c: agg([r for r in all_results if r["category"]==c]) for c in cats}

output = {
    "metadata": {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_test_cases": len(tests),
        "models": MODELS,
        "total_evaluations": len(all_results),
        "total_errors": errors,
        "note": "LIVE — guardrail checks direct, LLM calls via Ollama HTTP",
    },
    "overall_metrics":  overall_m,
    "model_metrics":    model_m,
    "category_metrics": cat_m,
    "results":          all_results,
}
with open("week5_results.json","w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(all_results)} records → week5_results.json  (errors: {errors})")

m = overall_m
print(f"""
{'='*60}
  WEEK 5 SUMMARY
{'='*60}
  Total:              {m['total']}
  Guardrail accuracy: {m['guardrail_accuracy_pct']}%
  Refusal rate:       {m['appropriate_refusal_rate_pct']}%
  False block rate:   {m['false_block_rate_pct']}%
  False allow rate:   {m['false_allow_rate_pct']}%
  Hall before:        {m['hallucination_rate_before_pct']}%
  Hall after:         {m['hallucination_rate_after_pct']}%
  Hall reduction:     {m['hallucination_reduction_pct']}%
  Test pass rate:     {m['test_pass_rate_pct']}%
""")
for model in MODELS:
    mm = model_m[model]
    print(f"  {SHORT[model]:<12}: acc={mm['guardrail_accuracy_pct']}%  refusal={mm['appropriate_refusal_rate_pct']}%  pass={mm['test_pass_rate_pct']}%")
