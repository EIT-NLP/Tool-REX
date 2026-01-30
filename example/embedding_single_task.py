# evaluate embedding models

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_de.eval import eval_retrieval
from tool_de.config import _MODEL, _TASK, _QUERY_REPO, _TOOL_REPO, _CATEGORY
from collections import defaultdict



model_path = "Lux1997/Tool-Embed-0.6B"
model_name = os.path.basename(model_path)
print(model_name)

# New: control is_inst in one place and use it for naming
is_inst = False
inst_suffix = "inst" if is_inst else "noinst"

# Results directory
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# Naming: include model name + inst/noinst in filenames
output_file = os.path.join(results_dir, f"output_{model_name}_{inst_suffix}_single.json")
results_file = os.path.join(results_dir, f"results_{model_name}_{inst_suffix}_single.json")

# Print parquet filenames here
print("📂 Data sources for this evaluation:")
print(f"Query dataset: {_QUERY_REPO}")
print(f"Tool dataset: {_TOOL_REPO}")
print("Per-task parquet files:")
for task_name in _TASK:
    query_path = f"{_QUERY_REPO}/{task_name}"
    print(f"  - {task_name} (queries): {query_path}")
for category in _CATEGORY:
    tool_path = f"{_TOOL_REPO}/{category}"
    print(f"  - {category} (tools): {tool_path}")
        
results = eval_retrieval(
    model_name=model_path,
    tasks="all",
    category="all",
    batch_size=8,
    output_file=output_file,
    top_k=100,
    is_inst=is_inst,          # Keep original parameter, use the is_inst variable above
    is_print=True
)

# Save per-task scores
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Compute and save average metrics across tasks
metric_sums = {}
metric_counts = {}
for task_metrics in results.values():
    if not isinstance(task_metrics, dict):
        continue
    for metric, value in task_metrics.items():
        if isinstance(value, (int, float)):
            metric_sums[metric] = metric_sums.get(metric, 0.0) + float(value)
            metric_counts[metric] = metric_counts.get(metric, 0) + 1

avg_results = {m: round(metric_sums[m] / metric_counts[m], 6) for m in metric_sums if metric_counts.get(m, 0) > 0}

avg_results_file = os.path.join(results_dir, f"results_avg_{model_name}_{inst_suffix}_single.json")
with open(avg_results_file, 'w', encoding='utf-8') as f:
    json.dump(avg_results, f, indent=2, ensure_ascii=False)

print(f"✅ Per-task results saved to: {results_file}")
print(f"✅ Retrieval outputs saved to: {output_file}")
print(f"✅ Average metrics saved to: {avg_results_file}")
print(results)
