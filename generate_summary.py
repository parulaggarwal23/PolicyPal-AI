import json

with open("evaluation_results_live.json") as f:
    data = json.load(f)
results = data["results"]

models = list(set(r["model"] for r in results))
categories = list(set(r["category"] for r in results))

print("# PolicyPal AI — Evaluation Summary")
print("## Week 4 Exercise 4: Category-Wise Analysis\n")

print("| Category | " + " | ".join(models) + " |")
print("|---" + "|---" * len(models) + "|")

for cat in sorted(categories):
    row = [cat]
    for model in models:
        cat_results = [r for r in results if r["model"] == model and r["category"] == cat]
        if not cat_results:
            row.append("N/A")
            continue
        scores = [r["correctness_score"] for r in cat_results if r.get("correctness_score") is not None]
        avg = round(sum(scores)/len(scores), 2) if scores else 0
        row.append(str(avg))
    print("| " + " | ".join(row) + " |")

