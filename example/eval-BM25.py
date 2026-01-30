# evaluate BM25 only (dataset from config)

import sys
import os
import json
from typing import Dict, Tuple, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import bm25s
import pytrec_eval

# Import data sources and task/category from config
from tool_de.config import _TASK, _QUERY_REPO, _TOOL_REPO, _CATEGORY

# ----------------- trec-style metrics (aligned with other models) -----------------
def trec_eval_aligned(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    ndcg, _map, recall, prec, comp = {}, {}, {}, {}, {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        prec[f"Precision@{k}"] = 0.0
        comp[f"Comprehensiveness@{k}"] = 0.0

    map_string = "map_cut." + ",".join(map(str, k_values))
    ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
    recall_string = "recall." + ",".join(map(str, k_values))
    precision_string = "P." + ",".join(map(str, k_values))

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    if not scores:
        return {**ndcg, **_map, **recall, **prec, **comp}

    for qid, sc in scores.items():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += sc[f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"] += sc[f"map_cut_{k}"]
            recall[f"Recall@{k}"] += sc[f"recall_{k}"]
            prec[f"Precision@{k}"] += sc[f"P_{k}"]
            comp[f"Comprehensiveness@{k}"] += (sc[f"recall_{k}"] == 1)

    n = len(scores)
    def _norm(d): return {k: round(v / n, 5) for k, v in d.items()}

    out = {}
    out.update(_norm(ndcg))
    out.update(_norm(_map))
    out.update(_norm(recall))
    out.update(_norm(prec))
    out.update(_norm(comp))
    return out

def _safe_json_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

# ----------------- BM25 core -----------------
def _load_all_tools(category: str = "all", cache_dir: str = None):
    """
    Matches the semantics of load_tools('all'):
    - If category == 'all': merge tools from each split in _CATEGORY
    - Otherwise: load only the specified split
    Required fields: id + documentation (or compatible doc)
    """
    tools = []
    cats: List[str]
    if category.lower() == "all":
        cats = list(_CATEGORY)
    else:
        cats = [category]

    for c in cats:
        ds = load_dataset(_TOOL_REPO, c, cache_dir=cache_dir)
        if 'tools' not in ds:
            raise ValueError(f"{_TOOL_REPO}/{c} does not contain a 'tools' split")
        tools.extend(ds['tools'])
    return tools

def _load_queries_for_task(task: str, cache_dir: str = None):
    ds = load_dataset(_QUERY_REPO, task, cache_dir=cache_dir)
    if 'queries' not in ds:
        raise ValueError(f"{_QUERY_REPO}/{task} does not contain a 'queries' split")
    return ds['queries']

def eval_bm25_all_tasks(
    tasks: List[str],
    category: str = "all",
    rk_num: int = 100,
    instruct: bool = True,
    cache_dir: str = None,
    k_values: Tuple[int, ...] = (1, 5, 10, 20),
):
    # 1) Build tool corpus and BM25 index
    tools = _load_all_tools(category=category, cache_dir=cache_dir)

    def get_doc_text(t):
        if 'documentation' in t:
            return str(t['documentation'])
        if 'doc' in t:
            return str(t['doc'])
        raise KeyError("Tool data missing 'documentation' or 'doc' field")

    doc_texts = [get_doc_text(t) for t in tools]
    doc_ids = [str(t['id']) for t in tools]

    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # 2) Retrieve and evaluate per task
    runs: Dict[str, Dict[str, Dict[str, float]]] = {}
    metrics_by_task: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        queries = _load_queries_for_task(task, cache_dir=cache_dir)

        # Retrieval
        task_run: Dict[str, Dict[str, float]] = {}
        for q in queries:
            qid = str(q['id'])
            qtext = q['query']
            if instruct:
                inst = q.get('instruction') or ''
                if inst:
                    qtext = f"{inst} {qtext}"

            q_tokens = bm25s.tokenize(qtext)
            if len(q_tokens.vocab) == 0:
                q_tokens = bm25s.tokenize("NONE", stopwords=[])

            hits, scores = retriever.retrieve(
                q_tokens, corpus=doc_ids, k=min(rk_num, len(doc_ids))
            )
            q_res = {}
            for i in range(hits.shape[1]):
                q_res[str(hits[0, i])] = float(scores[0, i])
            task_run[qid] = q_res

        # qrels
        qrels = {}
        for q in queries:
            qid = str(q['id'])
            labels = _safe_json_loads(q.get('labels', []))
            if isinstance(labels, list):
                qrels[qid] = {str(x['id']): int(x['relevance']) for x in labels}
            else:
                qrels[qid] = {}

        # Evaluation
        metrics = trec_eval_aligned(qrels, task_run, k_values=k_values)
        runs[task] = task_run
        metrics_by_task[task] = metrics

    return runs, metrics_by_task

# ----------------- main: follow your current pattern -----------------
if __name__ == "__main__":
    # Model name & flags (keep current style)
    model_path = "BM25"
    model_name = os.path.basename(model_path)
    print(model_name)

    is_inst = True
    inst_suffix = "inst" if is_inst else "noinst"

    # Results directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, f"output_{model_name}_{inst_suffix}_single.json")
    results_file = os.path.join(results_dir, f"results_{model_name}_{inst_suffix}_single.json")

    # Print data sources (from config)
    print("📂 Data sources for this evaluation:")
    print(f"Query dataset: {_QUERY_REPO}")
    print(f"Tool dataset: {_TOOL_REPO}")
    print("Per-task parquet files:")
    for task_name in _TASK:
        print(f"  - {task_name} (queries): {_QUERY_REPO}/{task_name}")
    for category in _CATEGORY:
        print(f"  - {category} (tools): {_TOOL_REPO}/{category}")

    # Run BM25 evaluation (tasks="all" means iterate _TASK)
    tasks_list = list(_TASK)  # Keep the original call behavior
    runs, metrics_by_task = eval_bm25_all_tasks(
        tasks=tasks_list,
        category="all",
        rk_num=100,
        instruct=is_inst,
        cache_dir="./cache",  # Optional
        k_values=(1, 5, 10, 20),
    )

    # Save run (candidates and scores)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)

    # Save per-task metrics (aligned with other models)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics_by_task, f, indent=2, ensure_ascii=False)

    # Compute and save average metrics across tasks (keep existing logic)
    metric_sums = {}
    metric_counts = {}
    for task_metrics in metrics_by_task.values():
        if not isinstance(task_metrics, dict):
            continue
        for metric, value in task_metrics.items():
            if isinstance(value, (int, float)):
                metric_sums[metric] = metric_sums.get(metric, 0.0) + float(value)
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

    avg_results = {m: round(metric_sums[m] / metric_counts[m], 6)
                   for m in metric_sums if metric_counts.get(m, 0) > 0}

    avg_results_file = os.path.join(results_dir, f"results_avg_{model_name}_{inst_suffix}_single.json")
    with open(avg_results_file, "w", encoding="utf-8") as f:
        json.dump(avg_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Per-task results saved to: {results_file}")
    print(f"✅ Retrieval outputs saved to: {output_file}")
    print(f"✅ Average metrics saved to: {avg_results_file}")
    print(metrics_by_task)
