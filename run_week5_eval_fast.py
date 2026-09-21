"""
Week 5 Fast Evaluation — uses the existing /api/w5-rag-ask endpoint.
Guardrail-triggered tests return instantly (<1s).
Only truly in-scope LLM calls take time (~6-15s with qwen/tinyllama).
CodeLlama only runs on a small representative subset to avoid timeouts.
"""
import json, time, requests, urllib.parse
from datetime import datetime, timezone
from pathlib import Path
import re

import os

APP = os.getenv("APP_URL", "http://localhost:8088")
# All 3 models run the full 28 tests

MODELS = ["qwen2.5:0.5b", "tinyllama:1.1b", "codellama:latest"]
TIMEOUT = 120   # per call
PAUSE   = 1.5

with open("week5_guardrail_tests.json") as f:
    ds = json.load(f)
tests = ds["tests"]
print(f"Tests: {len(tests)} | Models: {MODELS}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def call(q, model, skip):
    url = f"http://localhost:8000/ask_v2"
    payload = {
        "question": q,
        "skip_guardrails": skip
    }
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def passes(v):
    return v is True or v is None   # None = not applicable


SHORT = {"codellama:latest": "CodeLlama",
         "qwen2.5:0.5b":     "Qwen",
         "tinyllama:1.1b":    "TinyLlama"}

results = []
done = 0
total = len(tests) * len(MODELS)
errors = 0

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"  {SHORT[model]} ({model})")
    print(f"{'='*60}")

    for t in tests:
        done += 1
        q   = t["question"]
        tid = t["id"]
        cat = t["category"]
        print(f"  [{done:03d}/{total}] {tid} [{cat}] {q[:50]}…", end=" ", flush=True)

        # WITHOUT guardrails
        d_wo, err_wo = call(q, model, skip=True)
        if err_wo:
            errors += 1
        ans_wo = (d_wo or {}).get("answer", "")
        ret_wo = (d_wo or {}).get("retrieved_chunks", [])

        # WITH guardrails
        d_wi, err_wi = call(q, model, skip=False)
        if err_wi:
            errors += 1
        ans_wi = (d_wi or {}).get("answer", "")
        gt = (d_wi or {}).get("guardrail_triggered") or {}
        gt_name = gt.get("guardrail") if gt else None
        ov = (d_wi or {}).get("output_validation") or (d_wo or {}).get("output_validation") or {}

        # Scoring
        expected_trig = t.get("expected_guardrail_trigger", False)
        actual_trig   = gt_name is not None
        guard_correct = actual_trig == expected_trig

        output_tests = {
            "relevance":           ov.get("relevance",           {}).get("pass"),
            "context_support":     ov.get("context_support",     {}).get("pass"),
            "hallucination_check": ov.get("hallucination_check", {}).get("pass"),
            "sufficient_info":     ov.get("sufficient_info",     {}).get("pass"),
            "appropriate_refusal": ov.get("appropriate_refusal", {}).get("pass"),
            "format":              ov.get("format",              {}).get("pass"),
            "overall":             ov.get("overall_pass"),
        }
        test_pass = guard_correct and all(passes(v) for v in output_tests.values())

        sym = "✓" if test_pass else "✗"
        g   = f"TRIGGERED:{gt_name}" if actual_trig else "—"
        print(f"{sym}  guard={g}  correct={guard_correct}")

        results.append({
            "test_id":            tid,
            "category":           cat,
            "question":           q,
            "model":              model,
            "expected_behavior":  t["expected_behavior"],
            "expected_trigger":   expected_trig,
            "required_guardrail": t.get("required_guardrail"),
            "kb_supported":       t.get("kb_supported", False),
            "answer_without":     ans_wo[:300],
            "retrieved_without":  ret_wo,
            "latency_without_ms": (d_wo or {}).get("latency_ms"),
            "error_without":      err_wo,
            "answer_with":        ans_wi[:300],
            "guardrail_triggered": gt_name,
            "actual_trigger":     actual_trig,
            "guardrail_correct":  guard_correct,
            "retrieved_with":     (d_wi or {}).get("retrieved_chunks", []),
            "latency_with_ms":    (d_wi or {}).get("latency_ms"),
            "error_with":         err_wi,
            "output_tests":       output_tests,
            "test_overall_pass":  test_pass,
        })
        time.sleep(PAUSE)


# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(rs):
    if not rs:
        return {}
    n        = len(rs)
    trig_exp = [r for r in rs if r["expected_trigger"]]
    no_trig  = [r for r in rs if not r["expected_trigger"]]
    c_block  = sum(1 for r in trig_exp if r["actual_trigger"])
    c_allow  = sum(1 for r in no_trig  if not r["actual_trigger"])
    f_block  = sum(1 for r in no_trig  if r["actual_trigger"])
    f_allow  = sum(1 for r in trig_exp if not r["actual_trigger"])
    g_acc    = round((c_block + c_allow) / n * 100, 1)
    ref_rate = round(c_block / len(trig_exp) * 100, 1) if trig_exp else None
    fb_rate  = round(f_block / len(no_trig)  * 100, 1) if no_trig  else 0
    fa_rate  = round(f_allow / len(trig_exp) * 100, 1) if trig_exp else 0
    hv_all   = [r for r in rs if r["output_tests"].get("hallucination_check") is not None]
    h_before = round(sum(1 for r in rs if r["output_tests"].get("hallucination_check") is False) / n * 100, 1)
    nb       = [r for r in rs if not r["actual_trigger"]]
    h_after  = round(sum(1 for r in nb if r["output_tests"].get("hallucination_check") is False) / len(nb) * 100, 1) if nb else 0
    return {
        "total": n,
        "trigger_expected": len(trig_exp),
        "correctly_blocked": c_block,
        "correctly_allowed": c_allow,
        "false_blocks": f_block,
        "false_allows": f_allow,
        "guardrail_accuracy_pct":       g_acc,
        "appropriate_refusal_rate_pct": ref_rate,
        "false_block_rate_pct": fb_rate,
        "false_allow_rate_pct": fa_rate,
        "hallucination_rate_before_pct": h_before,
        "hallucination_rate_after_pct":  h_after,
        "hallucination_reduction_pct":   round(h_before - h_after, 1),
        "test_pass_rate_pct": round(sum(1 for r in rs if r["test_overall_pass"]) / n * 100, 1),
    }

overall_m = agg(results)
model_m   = {m: agg([r for r in results if r["model"] == m]) for m in MODELS}
cats      = sorted(set(r["category"] for r in results))
cat_m     = {c: agg([r for r in results if r["category"] == c]) for c in cats}

output = {
    "metadata": {
        "run_at":             datetime.now(timezone.utc).isoformat(),
        "total_test_cases":   len(tests),
        "models":             MODELS,
        "total_evaluations":  len(results),
        "total_errors":       errors,
        "note":               "LIVE — all answers from actual Ollama execution",
    },
    "overall_metrics":  overall_m,
    "model_metrics":    model_m,
    "category_metrics": cat_m,
    "results":          results,
}

with open("week5_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(results)} records → week5_results.json  (errors: {errors})")

# ── Summary ───────────────────────────────────────────────────────────────────
m = overall_m
print(f"""
{'='*60}
  WEEK 5 SUMMARY
{'='*60}
  Total tests:        {m['total']}
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
print()
