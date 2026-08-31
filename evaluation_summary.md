# PolicyPal AI — Evaluation Summary
## Week 4 Exercise 3: Quantitative Evaluation

**Run date:** 2026-08-26 19:01  
**Total evaluations:** 75 (25 questions × 3 models)  
**Embedding model (fixed):** nomic-embed-text  
**Retrieval:** Cosine similarity · Top-3 chunks · 590 total chunks  
**Scoring:** Keyword overlap vs ground truth (KB questions); refusal detection (hallucination traps)  

---

## Model Comparison — Overall

| Metric | codellama:latest | qwen2.5:0.5b | tinyllama:1.1b |
|---|---|---|---|
| Valid responses (of 25) | 19 | 24 | 25 |
| Errors / timeouts | 6 | 1 | 0 |
| Total correctness score | 27 | 32 | 34 |
| Max possible score | 38 | 48 | 50 |
| Accuracy (% of max) | 71.1% | 66.7% | 68.0% |
| Avg correctness (0–2) | 1.421 | 1.333 | 1.36 |
| Avg relevance (0–1) | 0.748 | 0.784 | 0.712 |
| Hallucinations | 2 | 2 | 1 |
| Hallucination rate | 10.5% | 8.3% | 4.0% |
| Retrieval hit rate | 100.0% | 100.0% | 100.0% |
| Avg total latency | 37721 ms | 5932 ms | 9290 ms |
| Median total latency | 38327 ms | 5994 ms | 8537 ms |
| Latency std dev | 6480.0 ms | 1191.8 ms | 4397.2 ms |
| Avg LLM-only latency | 37205 ms | 5854 ms | 9209 ms |

---

## Score Distribution

| Score | Meaning | codellama:latest | qwen2.5:0.5b | tinyllama:1.1b |
|---|---|---|---|---|
| 2 | ✅ Correct & grounded | 11 | 13 | 12 |
| 1 | ⚠️ Partially correct | 5 | 6 | 10 |
| 0 | ❌ Incorrect / hallucinated | 3 | 5 | 3 |

---

## Avg Correctness by Difficulty

| Difficulty | codellama:latest | qwen2.5:0.5b | tinyllama:1.1b |
|---|---|---|---|
| Easy | 1.57 (n=7) | 1.57 (n=7) | 1.5 (n=8) |
| Medium | 1.25 (n=8) | 1.2 (n=10) | 1.1 (n=10) |
| Hard | 1.5 (n=4) | 1.29 (n=7) | 1.57 (n=7) |

---

## Avg Correctness by Category

| Category | codellama:latest | qwen2.5:0.5b | tinyllama:1.1b |
|---|---|---|---|
| GST | 1.43 | 1.25 | 1.33 |
| Consumer Protection | 1.0 | 1.33 | 1.5 |
| Information Technology | 1.75 | 1.4 | 1.6 |
| Essential Commodities | 2.0 | 1.75 | 1.25 |
| Out-of-scope | 0.0 | 0.0 | 0.0 |

---

## Hallucination Analysis

Hallucination-trap questions: **Q09** (GST export rate — not in KB) and **Q25** (Income tax slabs — not in KB).  
Correct behaviour: model says *'I could not find this information in the policy documents.'*

| ID | Question | codellama | qwen2.5 | tinyllama |
|---|---|---|---|---|
| Q09 | What is the GST rate applicable to export of goods from… | ❌ Hallucinated | ❌ Hallucinated | ⚠️ Partial |
| Q25 | What is the income tax slab rate for individuals earnin… | ❌ Hallucinated | ❌ Hallucinated | ❌ Hallucinated |

---

## Retrieval Quality

Score = 1 if expected source document appears in the top-3 retrieved chunks, 0 otherwise.  
Q09 and Q25 excluded (no expected source).

| Model | Questions tested | Hits | Hit rate |
|---|---|---|---|
| codellama:latest | 18 | 18 | 100.0% |
| qwen2.5:0.5b | 23 | 23 | 100.0% |
| tinyllama:1.1b | 24 | 24 | 100.0% |

---

## Per-Question Results

| ID | Diff | KB | codellama | qwen2.5 | tinyllama | codellama lat | qwen lat | tinyllama lat |
|---|---|---|---|---|---|---|---|---|
| Q01 | Eas | Y | 2 | — | 2 | 37449ms | — | 7136ms |
| Q02 | Eas | Y | 2 | 2 | 2 | 39625ms | 10113ms | 8537ms |
| Q03 | Med | Y | 1 | 1 | 0 | 34984ms | 4228ms | 11842ms |
| Q04 | Har | Y | — | 0 | 2 | — | 6355ms | 14428ms |
| Q05 | Med | Y | 2 | 2 | 2 | 40049ms | 5576ms | 12626ms |
| Q06 | Har | Y | 1 | 1 | 1 | 39920ms | 5700ms | 24862ms |
| Q07 | Med | Y | 2 | 2 | 1 | 26296ms | 5824ms | 5036ms |
| Q08 | Eas | Y | — | 2 | 1 | — | 6004ms | 4914ms |
| Q09 | Med | N | 0 | 0 | 1 | 32938ms | 4768ms | 5770ms |
| Q10 | Eas | Y | 1 | 1 | 1 | 32927ms | 6393ms | 6489ms |
| Q11 | Med | Y | 1 | 2 | 1 | 41965ms | 6309ms | 9792ms |
| Q12 | Med | Y | — | 2 | 2 | — | 6569ms | 9111ms |
| Q13 | Med | Y | 0 | 0 | 1 | 40020ms | 5994ms | 8672ms |
| Q14 | Har | Y | 2 | 2 | 2 | 55602ms | 4521ms | 9573ms |
| Q15 | Har | Y | — | 1 | 2 | — | 4861ms | 16363ms |
| Q16 | Eas | Y | 2 | 2 | 2 | 39806ms | 5367ms | 8526ms |
| Q17 | Med | Y | 2 | 0 | 1 | 29483ms | 6434ms | 6575ms |
| Q18 | Har | Y | — | 2 | 2 | — | 5889ms | 6844ms |
| Q19 | Har | Y | 1 | 1 | 1 | 37765ms | 6804ms | 6941ms |
| Q20 | Eas | Y | 2 | 2 | 2 | 30326ms | 5636ms | 9110ms |
| Q21 | Eas | Y | 2 | 2 | 2 | 38327ms | 7208ms | 11145ms |
| Q22 | Med | Y | — | 1 | 0 | — | 6178ms | 10790ms |
| Q23 | Med | Y | 2 | 2 | 2 | 40563ms | 6025ms | 5923ms |
| Q24 | Har | Y | 2 | 2 | 1 | 45431ms | 4102ms | 5033ms |
| Q25 | Eas | N | 0 | 0 | 0 | 33229ms | 5503ms | 6219ms |

---

## Latency Analysis

| Model | Avg total (ms) | Median (ms) | Std dev (ms) | Avg LLM only (ms) |
|---|---|---|---|---|
| codellama:latest | 37721 | 38327 | 6480.0 | 37205 |
| qwen2.5:0.5b | 5932 | 5994 | 1191.8 | 5854 |
| tinyllama:1.1b | 9290 | 8537 | 4397.2 | 9209 |

---

## Unavailable Metrics

| Metric | Status | Reason |
|---|---|---|
| Token usage | **Unavailable** | Ollama /api/generate does not return token counts via pipeline endpoint |
| CPU usage % | **Unavailable** | Not measured at request level; requires OS-level sampling |
| Memory (MB) | **Unavailable** | Not measured at request level |

---

## Metric Definitions

### Correctness Score (0–2)
- **KB-supported questions:** Keyword overlap between response and ground truth answer.
  - **2** → overlap ≥ 0.40 (correct)
  - **1** → overlap ≥ 0.20 (partially correct)
  - **0** → overlap < 0.20 (incorrect)
- **Hallucination-trap questions (Q09, Q25):** Refusal phrase detection.
  - **2** → response contains 'not found / not in documents' with no fabricated number
  - **1** → vague response, no fabricated value
  - **0** → specific fabricated answer provided

### Accuracy (%)
Sum of correctness scores ÷ (valid responses × 2) × 100. Maximum = 100%.

### Relevance Score (0–1)
Keyword overlap (tokens ≥4 chars) between question text and response text.

### Hallucination Rate (%)
Count of responses where model fabricated a specific out-of-scope answer ÷ valid responses × 100.

### Retrieval Hit Rate (%)
Count of KB-supported questions where expected source document appeared in top-3 chunks ÷ KB-supported questions evaluated × 100.

### Latency
Wall-clock time from HTTP POST sent to full response received (embedding + similarity search + LLM generation). In milliseconds.

---

*Generated by run_evaluation.py — PolicyPal AI Week 4 Exercise 3*
