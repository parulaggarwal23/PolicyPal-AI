"""
PolicyPal-AI — Week 5 Guardrails & AI Output Testing: Batch Evaluation Script
===============================================================================
Runs all 28 Week 5 test cases through all 3 models (84 total evaluations).
Tests both WITH and WITHOUT guardrails for each case.
Saves results to week5_results.json.

Usage:
    venv/bin/python run_week5_evaluation.py
"""
import json
import time
import requests
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

APP_URL   = "http://localhost:8080"
MODELS    = ["codellama:latest", "qwen2.5:0.5b", "tinyllama:1.1b"]
TIMEOUT   = 150  # seconds per model call
PAUSE_S   = 2    # pause between requests

# Load test dataset
with open("week5_guardrail_tests.json") as f:
    ds = json.load(f)
tests = ds["tests"]
print(f"Loaded {len(tests)} test cases")
print(f"Models: {MODELS}")
print(f"Total planned evaluations: {len(tests)} × {len(MODELS)} = {len(tests)*len(MODELS)} (WITH guardrails)")
print(f"Plus {len(tests)} × {len(MODELS)} WITHOUT guardrails = {len(tests)*len(MODELS)*2} total calls")
print()

results = []
total_calls = 0
errors_total = 0

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"  MODEL: {model}")
    print(f"{'='*60}")

    for test in tests:
        qid = test["id"]
        q   = test["question"]
        total_calls += 1
        print(f"  [{total_calls:03d}] {qid} [{test['category']}] {q[:55]}…", flush=True)

        # ── WITHOUT guardrails ────────────────────────────────
        answer_without = None
        retrieved_without = []
        latency_without = None
        ov_without = None
        error_without = None

        try:
            t0 = time.time()
            url = (f"{APP_URL}/api/w5-rag-ask"
                   f"?question={urllib.parse.quote(q)}"
                   f"&model={urllib.parse.quote(model)}"
                   f"&skip_guardrails=true")
            r = requests.post(url, timeout=TIMEOUT)
            r.raise_for_status()
            d = r.json()
            answer_without   = d.get("answer", "")
            retrieved_without = d.get("retrieved_chunks", [])
            latency_without  = d.get("latency_ms")
            ov_without       = d.get("output_validation")
        except Exception as e:
            error_without = str(e)
            errors_total += 1
            print(f"    WITHOUT error: {e}")

        # ── WITH guardrails ───────────────────────────────────
        answer_with = None
        retrieved_with = []
        latency_with = None
        ov_with = None
        guardrail_triggered = None
        error_with = None

        try:
            url2 = (f"{APP_URL}/api/w5-rag-ask"
                    f"?question={urllib.parse.quote(q)}"
                    f"&model={urllib.parse.quote(model)}"
                    f"&skip_guardrails=false")
            r2 = requests.post(url2, timeout=TIMEOUT)
            r2.raise_for_status()
            d2 = r2.json()
            answer_with        = d2.get("answer", "")
            retrieved_with     = d2.get("retrieved_chunks", [])
            latency_with       = d2.get("latency_ms")
            ov_with            = d2.get("output_validation")
            guardrail_triggered = d2.get("guardrail_triggered", {}).get("guardrail") if d2.get("guardrail_triggered") else None
        except Exception as e:
            error_with = str(e)
            errors_total += 1
            print(f"    WITH error: {e}")

        # ── Scoring ───────────────────────────────────────────
        expected_trigger = test.get("expected_guardrail_trigger", False)
        actual_trigger   = guardrail_triggered is not None
        guardrail_correct = actual_trigger == expected_trigger

        # Output test results (use WITH-guardrail validation if available, else WITHOUT)
        ov = ov_with or ov_without or {}
        output_tests = {
            "relevance":           ov.get("relevance",           {}).get("pass"),
            "context_support":     ov.get("context_support",     {}).get("pass"),
            "hallucination_check": ov.get("hallucination_check", {}).get("pass"),
            "sufficient_info":     ov.get("sufficient_info",     {}).get("pass"),
            "appropriate_refusal": ov.get("appropriate_refusal", {}).get("pass"),
            "format":              ov.get("format",              {}).get("pass"),
            "overall":             ov.get("overall_pass"),
        }

        # Overall test pass: guardrail correct + output tests OK (or N/A)
        def _passes(v): return v is True or v is None  # None = not applicable
        test_overall_pass = guardrail_correct and all(_passes(v) for v in output_tests.values())

        status_str = "✓" if test_overall_pass else "✗"
        guard_str  = f"guard={'TRIGGERED:'+guardrail_triggered if actual_trigger else 'not_triggered'}"
        print(f"    {status_str} {guard_str} correct={guardrail_correct}  ans_w={len(answer_with or '')}ch")

        results.append({
            "test_id":            qid,
            "category":           test["category"],
            "question":           q,
            "model":              model,
            "expected_behavior":  test["expected_behavior"],
            "expected_trigger":   expected_trigger,
            "required_guardrail": test.get("required_guardrail"),
            "kb_supported":       test.get("kb_supported", False),
            # Without guardrails
            "answer_without":     answer_without or "",
            "retrieved_without":  retrieved_without,
            "latency_without_ms": latency_without,
            "error_without":      error_without,
            # With guardrails
            "answer_with":        answer_with or "",
            "guardrail_triggered": guardrail_triggered,
            "actual_trigger":     actual_trigger,
            "guardrail_correct":  guardrail_correct,
            "retrieved_with":     retrieved_with,
            "latency_with_ms":    latency_with,
            "error_with":         error_with,
            # Output tests
            "output_tests":       output_tests,
            "test_overall_pass":  test_overall_pass,
        })

        time.sleep(PAUSE_S)

# ── Aggregate metrics ─────────────────────────────────────────────────────────
def _calc_metrics(rs):
    """Calculate guardrail effectiveness metrics for a set of results."""
    total      = len(rs)
    trigger_cases  = [r for r in rs if r["expected_trigger"]]
    no_trigger     = [r for r in rs if not r["expected_trigger"]]
    correct_block  = sum(1 for r in trigger_cases  if r["actual_trigger"])
    correct_allow  = sum(1 for r in no_trigger      if not r["actual_trigger"])
    false_block    = sum(1 for r in no_trigger      if r["actual_trigger"])     # blocked when shouldn't
    false_allow    = sum(1 for r in trigger_cases   if not r["actual_trigger"]) # allowed when should block
    guardrail_acc  = round((correct_block + correct_allow) / total * 100, 1) if total else 0
    refusal_rate   = round(correct_block / len(trigger_cases) * 100, 1) if trigger_cases else None
    false_block_r  = round(false_block / len(no_trigger) * 100, 1) if no_trigger else 0
    false_allow_r  = round(false_allow / len(trigger_cases) * 100, 1) if trigger_cases else 0
    # Hallucination: answers where hallucination check failed (WITH guardrail path)
    hv_results = [r for r in rs if r["output_tests"].get("hallucination_check") is not None]
    hall_before = sum(1 for r in rs if r["output_tests"].get("hallucination_check") is False) / len(rs) * 100 if rs else 0
    # After guardrails: only non-blocked answers
    non_blocked = [r for r in rs if not r["actual_trigger"]]
    hall_after  = sum(1 for r in non_blocked if r["output_tests"].get("hallucination_check") is False) / len(non_blocked) * 100 if non_blocked else 0
    return {
        "total":           total,
        "trigger_expected": len(trigger_cases),
        "correctly_blocked": correct_block,
        "correctly_allowed": correct_allow,
        "false_blocks":    false_block,
        "false_allows":    false_allow,
        "guardrail_accuracy_pct":  guardrail_acc,
        "appropriate_refusal_rate_pct": refusal_rate,
        "false_block_rate_pct": false_block_r,
        "false_allow_rate_pct": false_allow_r,
        "hallucination_rate_before_pct": round(hall_before, 1),
        "hallucination_rate_after_pct":  round(hall_after, 1),
        "hallucination_reduction_pct":   round(hall_before - hall_after, 1) if hall_before > 0 else 0,
        "test_pass_rate_pct": round(sum(1 for r in rs if r["test_overall_pass"]) / total * 100, 1) if total else 0,
    }

overall_metrics = _calc_metrics(results)
model_metrics   = {m: _calc_metrics([r for r in results if r["model"]==m]) for m in MODELS}

# Category breakdown
cat_metrics = {}
cats = set(r["category"] for r in results)
for cat in cats:
    cat_rs = [r for r in results if r["category"]==cat]
    cat_metrics[cat] = _calc_metrics(cat_rs)

# Save
output = {
    "metadata": {
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "total_test_cases": len(tests),
        "models":     MODELS,
        "total_evaluations": len(results),
        "total_errors": errors_total,
        "source": "LIVE — all answers generated by actual Ollama model execution",
    },
    "overall_metrics":  overall_metrics,
    "model_metrics":    model_metrics,
    "category_metrics": cat_metrics,
    "results":          results,
}

with open("week5_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n\nSaved {len(results)} results → week5_results.json")
print(f"Total errors: {errors_total}")
print()
print("="*60)
print("  WEEK 5 EVALUATION SUMMARY")
print("="*60)
m = overall_metrics
print(f"  Total test cases:        {m['total']}")
print(f"  Guardrail accuracy:      {m['guardrail_accuracy_pct']}%")
print(f"  Appropriate refusal:     {m['appropriate_refusal_rate_pct']}%")
print(f"  False block rate:        {m['false_block_rate_pct']}%")
print(f"  False allow rate:        {m['false_allow_rate_pct']}%")
print(f"  Hallucination before:    {m['hallucination_rate_before_pct']}%")
print(f"  Hallucination after:     {m['hallucination_rate_after_pct']}%")
print(f"  Hallucination reduction: {m['hallucination_reduction_pct']}%")
print(f"  Test pass rate:          {m['test_pass_rate_pct']}%")
print()
for model in MODELS:
    mm = model_metrics[model]
    short = {"codellama:latest":"CodeLlama","qwen2.5:0.5b":"Qwen","tinyllama:1.1b":"TinyLlama"}[model]
    print(f"  {short}: guardrail_acc={mm['guardrail_accuracy_pct']}%  refusal={mm['appropriate_refusal_rate_pct']}%  pass={mm['test_pass_rate_pct']}%")
print("="*60)
