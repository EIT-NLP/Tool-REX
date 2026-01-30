# evaluate rerank models (ToolRank)

import sys
import os
import json
import logging
import time
import multiprocessing
from datetime import datetime

# Set multiprocessing start method (must be before other imports)
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional environment variables can be set externally if needed.

from tool_de.eval import eval_rerank, eval_toolrank
from tool_de.config import _TASK, _QUERY_REPO, _TOOL_REPO, _CATEGORY
from collections import defaultdict

# Logging setup
def setup_logger():
    """Configure logging."""
    # Create log directory
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rerank_evaluation_{timestamp}.log")
    
    # Configure log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create dedicated logger
    logger = logging.getLogger('rerank_evaluation')
    logger.setLevel(logging.INFO)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, log_file

def main():
    """Main entry point."""
    # Initialize logging
    logger, log_file = setup_logger()

    # Record start time
    script_start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 Reranking evaluation started")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Config
    model_path = "Lux1997/Tool-Rank-4B"
    model_name = os.path.basename(model_path)
    logger.info(f"Using model: {model_name}")

    # Retrieval results path
    retrieval_results_path = "./results/output/retrieval_results.json"

    # Control parameters
    is_inst = True
    inst_suffix = "inst" if is_inst else "noinst"
    from_top_k = 100  # Rerank from top-100 candidates

    # Results directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Output paths
    rerank_output_file = os.path.join(results_dir, f"rerank_output_{model_name}_{inst_suffix}_single.json")
    rerank_results_file = os.path.join(results_dir, f"rerank_results_{model_name}_{inst_suffix}_single.json")

    logger.info("📂 Data sources used for this reranking run:")
    logger.info(f"Query dataset: {_QUERY_REPO}")
    logger.info(f"Tool dataset: {_TOOL_REPO}")
    logger.info(f"Retrieval results file: {retrieval_results_path}")
    logger.info("Per-task parquet files:")
    for task_name in _TASK:
        query_path = f"{_QUERY_REPO}/{task_name}"
        logger.info(f"  - {task_name} (queries): {query_path}")
    for category in _CATEGORY:
        tool_path = f"{_TOOL_REPO}/{category}"
        logger.info(f"  - {category} (tools): {tool_path}")

    # Check required files
    logger.info("🔍 Checking required files...")
    if os.path.exists(retrieval_results_path):
        file_size = os.path.getsize(retrieval_results_path) / (1024 * 1024)  # MB
        logger.info(f"✅ Retrieval results file exists: {retrieval_results_path} ({file_size:.2f} MB)")
    else:
        logger.warning(f"⚠️ Retrieval results file not found: {retrieval_results_path}")

    if os.path.isabs(model_path) or model_path.startswith("."):
        if os.path.exists(model_path):
            logger.info(f"✅ Model path exists: {model_path}")
        else:
            logger.warning(f"⚠️ Model path not found: {model_path}")
    else:
        logger.info(f"✅ Using model from Hugging Face Hub: {model_path}")

    # Check output directory
    if os.path.exists(results_dir):
        logger.info(f"✅ Results directory exists: {results_dir}")
    else:
        logger.info(f"📁 Creating results directory: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

    logger.info("🚀 Starting ToolRank reranking...")
    logger.info(f"Reranking from top-{from_top_k} candidates")
    logger.info(f"Use instruction: {is_inst}")

    # Method 1: use eval_toolrank (recommended)
    evaluation_success = False
    try:
        logger.info("=== Using eval_toolrank ===")
        start_time = time.time()
        
        output, results = eval_toolrank(
            model_name=model_path,
            tasks="all",
            instruct=is_inst,
            from_top_k=from_top_k,
            batch_size=16,
            context_size=8192,
            num_gpus=2,
            force_rethink=0,
            retrieval_results_path=retrieval_results_path
        )
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"✅ ToolRank reranking finished in {duration:.2f}s")
        
        # Save rerank outputs
        logger.info("Saving rerank outputs...")
        with open(rerank_output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Rerank outputs saved to: {rerank_output_file}")
        
        # Save evaluation results
        logger.info("Saving evaluation results...")
        with open(rerank_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Evaluation results saved to: {rerank_results_file}")
        
        # Compute averages across tasks
        logger.info("Computing average metrics across tasks...")
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
        
        avg_results_file = os.path.join(results_dir, f"rerank_avg_{model_name}_{inst_suffix}_single.json")
        with open(avg_results_file, 'w', encoding='utf-8') as f:
            json.dump(avg_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Average metrics saved to: {avg_results_file}")
        
        # Verify rerank output is non-empty
        if not output or not results:
            logger.error("❌ Rerank output is empty!")
            logger.error("The script finished but produced no rerank results.")
            evaluation_success = False
        else:
            logger.info(f"✅ Rerank output verified with {len(output)} tasks")
            evaluation_success = True
        
        # Print results
        logger.info("📊 Reranking evaluation results:")
        logger.info("=" * 80)
        for task, metrics in results.items():
            logger.info(f"Task: {task}")
            logger.info("-" * 40)
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
        
        logger.info("📈 Average metrics:")
        logger.info("-" * 40)
        for metric, value in avg_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Show a few rerank examples
        logger.info("🔍 Rerank examples (first 5 queries):")
        logger.info("=" * 80)
        for task_name, task_results in output.items():
            logger.info(f"Task: {task_name}")
            query_count = 0
            for query_id, tool_scores in task_results.items():
                if query_count >= 5:  # Only show first 5 queries
                    break
                logger.info(f"Query {query_id}:")
                # Show top-5 tools sorted by score
                sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (tool_id, score) in enumerate(sorted_tools[:5]):
                    logger.info(f"  {i+1}. {tool_id}: {score:.4f}")
                query_count += 1
            break  # Only show the first task
        
    except Exception as e:
        logger.error(f"❌ eval_toolrank failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        logger.info("Trying fallback eval_rerank...")
        
        # Method 2: use eval_rerank as fallback
        try:
            logger.info("=== Using eval_rerank ===")
            start_time = time.time()
            
            results = eval_rerank(
                model_name=model_path,
                tasks="all",
                instruct=is_inst,
                from_top_k=from_top_k
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ Reranking finished in {duration:.2f}s")
            
            # Save results
            logger.info("Saving rerank results...")
            with open(rerank_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Evaluation results saved to: {rerank_results_file}")
            
            # Print results
            logger.info("📊 Reranking evaluation results:")
            logger.info("=" * 80)
            for task, metrics in results.items():
                logger.info(f"Task: {task}")
                logger.info("-" * 40)
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"{metric}: {value:.4f}")
            
            evaluation_success = True
        
        except Exception as e2:
            logger.error(f"❌ Fallback reranking also failed: {e2}")
            logger.error(f"Error type: {type(e2).__name__}")
            logger.error(f"Error details: {str(e2)}")
            logger.error("Please check model path and dependencies.")
            logger.error("Suggested checks:")
            logger.error("1. Model path correctness")
            logger.error("2. Dependency installation")
            logger.error("3. GPU memory availability")
            logger.error("4. Data files existence")
            
            # Mark as failed
            evaluation_success = False
            
            # Suggested fixes
            logger.error("=" * 60)
            logger.error("🔧 Suggested fixes:")
            logger.error("1. vLLM multiprocessing issues:")
            logger.error("   - Ensure the script runs under if __name__ == '__main__':")
            logger.error("   - Or set: export TOKENIZERS_PARALLELISM=false")
            logger.error("2. SamplingParams errors:")
            logger.error("   - Check vLLM version compatibility")
            logger.error("   - Try upgrading or downgrading vLLM")
            logger.error("3. BuilderConfig errors:")
            logger.error("   - Check task names")
            logger.error("   - Ensure all tasks exist in the dataset")
            logger.error("4. Suggested steps:")
            logger.error("   - Restart the Python environment")
            logger.error("   - Check dependency versions: pip list | grep vllm")
            logger.error("   - Try single-GPU mode: num_gpus=1")
            logger.error("=" * 60)

    # Total duration
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time

    logger.info("=" * 80)
    if not evaluation_success:
        logger.error("❌ Reranking evaluation failed!")
        logger.error("All reranking methods failed. Check the error logs.")
    else:
        logger.info("🎉 Reranking evaluation completed!")

    logger.info(f"Model: {model_name}")
    logger.info(f"Instruction: {is_inst}")
    logger.info(f"Reranked from top-{from_top_k}")
    logger.info(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    logger.info(f"Finish time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
