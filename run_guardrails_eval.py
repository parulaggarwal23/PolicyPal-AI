import json
import requests
import time
from pathlib import Path

DATASET_FILE = "data/week5_guardrails_dataset.json"
RESULTS_FILE = "static/guardrails_eval_results.json"
API_URL = "http://localhost:8088/api/rag-ask-pipeline"

def run_eval():
    print("Loading dataset...")
    with open(DATASET_FILE, "r") as f:
        cases = json.load(f)

    results = []
    
    # Initialize metrics structure
    metrics = {
        "input": {"total": 0, "passed": 0, "failed": 0, "correct_block": 0, "false_block": 0, "false_allow": 0},
        "scope": {"total": 0, "passed": 0, "failed": 0, "correct_block": 0, "false_block": 0, "false_allow": 0},
        "retrieval": {"total": 0, "passed": 0, "failed": 0, "correct_block": 0, "false_block": 0, "false_allow": 0},
        "grounding": {"total": 0, "passed": 0, "failed": 0, "correct_block": 0, "false_block": 0, "false_allow": 0, "supported": 0, "unsupported": 0, "hallucinated": 0},
        "output": {"total": 0, "passed": 0, "failed": 0, "correct_block": 0, "false_block": 0, "false_allow": 0},
    }

    for idx, case in enumerate(cases):
        q = case["question"]
        expected_guardrail = case["guardrail"]
        
        print(f"[{idx+1}/{len(cases)}] Testing: {q[:50]}...")
        
        try:
            resp = requests.post(API_URL, params={"question": q, "model": "tinyllama:1.1b"})
            data = resp.json()
        except Exception as e:
            print(f"Error calling API: {e}")
            continue
            
        triggered_guardrail = "none"
        if data.get("guardrail_input", {}).get("triggered"):
            triggered_guardrail = "input"
        elif data.get("guardrail_scope", {}).get("triggered"):
            triggered_guardrail = "scope"
        elif data.get("guardrail_retrieval", {}).get("triggered"):
            triggered_guardrail = "retrieval"
        elif data.get("guardrail_grounding", {}).get("triggered"):
            triggered_guardrail = "grounding"
        elif data.get("guardrail_output", {}).get("triggered"):
            triggered_guardrail = "output"

        # Record specific metrics
        # For grounding metrics
        if "guardrail_grounding" in data:
            if not data["guardrail_grounding"].get("triggered"):
                metrics["grounding"]["supported"] += 1
            else:
                metrics["grounding"]["unsupported"] += 1
                if data["guardrail_grounding"].get("details", {}).get("hallucination_check", {}).get("pass") is False:
                    metrics["grounding"]["hallucinated"] += 1

        is_correct = (triggered_guardrail == expected_guardrail)
        
        # Calculate false blocks / allows for each guardrail type
        for g_type in metrics.keys():
            metrics[g_type]["total"] += 1
            
            # If the test case was EXPECTING this guardrail to trigger
            if expected_guardrail == g_type:
                if triggered_guardrail == g_type:
                    metrics[g_type]["correct_block"] += 1
                    metrics[g_type]["passed"] += 1
                elif triggered_guardrail == "none" or triggered_guardrail != g_type:
                    metrics[g_type]["false_allow"] += 1
                    metrics[g_type]["failed"] += 1
            # If the test case was NOT expecting this guardrail
            else:
                if triggered_guardrail == g_type:
                    metrics[g_type]["false_block"] += 1
                    metrics[g_type]["failed"] += 1
                else:
                    metrics[g_type]["passed"] += 1

        results.append({
            "id": case["id"],
            "question": q,
            "expected_behavior": case["expected_behavior"],
            "expected_guardrail": expected_guardrail,
            "actual_guardrail_triggered": triggered_guardrail,
            "is_correct": is_correct,
            "raw_response": data
        })
        time.sleep(1) # Be nice to Ollama

    # Calculate accuracy
    for g_type, mets in metrics.items():
        if mets["total"] > 0:
            mets["accuracy"] = round((mets["passed"] / mets["total"]) * 100, 1)
        else:
            mets["accuracy"] = 0.0

    final_output = {
        "metrics": metrics,
        "results": results
    }

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Saved evaluation results to {RESULTS_FILE}")

if __name__ == "__main__":
    run_eval()
