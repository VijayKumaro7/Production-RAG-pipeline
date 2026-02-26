"""
experiment_runner.py — Automated experiment runner for RAG hyperparameter search.

Loops through different configurations, runs RAGAS evaluation for each,
and logs all results to JSON files and optionally MLflow.

Experiments:
  - chunk_size: [256, 512, 1024]
  - chunk_overlap: [0, 50, 100]
  - top_k_retrieval: [2, 4, 6]
  - retrieval_method: [similarity, mmr]
  - embedding_model: [openai, huggingface]

Usage:
    python experiments/experiment_runner.py --experiments chunk_size top_k_retrieval
    python experiments/experiment_runner.py --all
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ─────────────────────────────────────────────────────
# Experiment Definitions
# ─────────────────────────────────────────────────────

EXPERIMENT_GRID = {
    "chunk_size": {
        "param": "chunk_size",
        "values": [256, 512, 1024],
        "description": "Effect of chunk size on retrieval and generation quality",
    },
    "chunk_overlap": {
        "param": "chunk_overlap",
        "values": [0, 50, 100],
        "description": "Effect of chunk overlap on boundary information retention",
    },
    "top_k_retrieval": {
        "param": "top_k",
        "values": [2, 4, 6],
        "description": "Effect of number of retrieved chunks on answer quality",
    },
    "retrieval_method": {
        "param": "retrieval_method",
        "values": ["similarity", "mmr"],
        "description": "Similarity search vs MMR for retrieval diversity",
    },
    "embedding_model": {
        "param": "embedding_provider",
        "values": ["huggingface", "openai"],
        "description": "Effect of embedding model quality on retrieval",
    },
}

# ─────────────────────────────────────────────────────
# MLflow Logging (optional)
# ─────────────────────────────────────────────────────

def init_mlflow(tracking_uri: str = "./mlruns"):
    """Initialize MLflow tracking."""
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("RAG_Pipeline_Experiments")
        logger.info(f"MLflow initialized at {tracking_uri}")
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed. Using JSON logging only.")
        return None


def log_to_mlflow(mlflow, run_name: str, params: Dict, metrics: Dict):
    """Log an experiment run to MLflow."""
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                if v is not None:
                    mlflow.log_metric(k, v)
        logger.info(f"Logged to MLflow: {run_name}")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


# ─────────────────────────────────────────────────────
# Single Experiment Run
# ─────────────────────────────────────────────────────

def run_single_experiment(
    experiment_name: str,
    param_name: str,
    param_value: Any,
    base_config: Dict,
    questions: List[Dict],
    mlflow=None,
    results_dir: str = "experiments/results",
) -> Dict[str, Any]:
    """
    Run one experiment configuration and evaluate with RAGAS.

    Args:
        experiment_name: Human-readable name (e.g., 'chunk_size_512').
        param_name: Config parameter being varied.
        param_value: Value for this run.
        base_config: Base pipeline config dict.
        questions: Test QA dataset.
        mlflow: MLflow instance (optional).
        results_dir: Output directory for result JSONs.

    Returns:
        Result dict with config, scores, and timing.
    """
    from src.pipeline import RAGConfig, RAGPipeline
    from evaluation.ragas_eval import run_pipeline_on_questions, evaluate_with_ragas

    run_name = f"{experiment_name}_{param_value}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {run_name}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Build config with override
    config = RAGConfig(
        chunk_strategy=base_config["ingestion"]["chunk_strategy"],
        chunk_size=base_config["ingestion"]["chunk_size"],
        chunk_overlap=base_config["ingestion"]["chunk_overlap"],
        embedding_provider=base_config["embeddings"]["provider"],
        vector_store_provider=base_config["vector_store"]["provider"],
        retrieval_method=base_config["retrieval"]["method"],
        top_k=base_config["retrieval"]["top_k"],
        llm_provider=base_config["llm"]["provider"],
    )

    # Apply the experiment override
    setattr(config, param_name, param_value)

    # Rebuild vector store if chunking params changed
    if param_name in ("chunk_size", "chunk_overlap", "chunk_strategy", "embedding_provider"):
        config.force_rebuild = True

    try:
        pipeline = RAGPipeline(config)
        pipeline.initialize()

        # Run pipeline on test questions
        eval_data = run_pipeline_on_questions(questions, pipeline)

        # RAGAS evaluation
        scores = evaluate_with_ragas(eval_data)

        total_time = time.time() - start_time

        result = {
            "experiment": experiment_name,
            "run_name": run_name,
            "param_name": param_name,
            "param_value": str(param_value),
            "timestamp": datetime.now().isoformat(),
            "total_time_s": round(total_time, 2),
            "num_samples": len(eval_data),
            "config": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "chunk_strategy": config.chunk_strategy,
                "embedding_provider": config.embedding_provider,
                "retrieval_method": config.retrieval_method,
                "top_k": config.top_k,
                "llm_provider": config.llm_provider,
            },
            "scores": scores,
            "mean_score": (
                sum(v for v in scores.values() if v is not None)
                / max(1, sum(1 for v in scores.values() if v is not None))
            ),
            "status": "success",
        }

        # Log to MLflow
        log_to_mlflow(mlflow, run_name, result["config"], scores)

    except Exception as e:
        logger.error(f"Experiment {run_name} failed: {e}")
        result = {
            "experiment": experiment_name,
            "run_name": run_name,
            "param_name": param_name,
            "param_value": str(param_value),
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
            "scores": {},
            "mean_score": 0.0,
        }

    # Save individual result
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{run_name}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved result to {result_path}")

    return result


# ─────────────────────────────────────────────────────
# Experiment Suite Runner
# ─────────────────────────────────────────────────────

def run_experiment_suite(
    experiment_names: List[str],
    config_path: str = "config.yaml",
    questions_path: str = "evaluation/test_questions.json",
    results_dir: str = "experiments/results",
    use_mlflow: bool = True,
) -> pd.DataFrame:
    """
    Run multiple experiment suites and compare results.

    Args:
        experiment_names: List of experiment keys from EXPERIMENT_GRID.
        config_path: Path to config.yaml.
        questions_path: Path to test_questions.json.
        results_dir: Output directory.
        use_mlflow: Whether to log to MLflow.

    Returns:
        DataFrame with all experiment results.
    """
    import json
    from evaluation.ragas_eval import load_test_questions

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed, returning raw list")
        pd = None

    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    questions = load_test_questions(questions_path)
    mlflow_client = init_mlflow() if use_mlflow else None

    all_results = []

    for exp_name in experiment_names:
        if exp_name not in EXPERIMENT_GRID:
            logger.warning(f"Unknown experiment: {exp_name}. Skipping.")
            continue

        exp_config = EXPERIMENT_GRID[exp_name]
        logger.info(f"\n🧪 Starting experiment suite: {exp_name}")
        logger.info(f"   {exp_config['description']}")

        for value in exp_config["values"]:
            result = run_single_experiment(
                experiment_name=exp_name,
                param_name=exp_config["param"],
                param_value=value,
                base_config=base_config,
                questions=questions,
                mlflow=mlflow_client,
                results_dir=results_dir,
            )
            all_results.append(result)

    # Save aggregated results
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n📊 All results saved to {summary_path}")

    # Print comparison table
    print_results_table(all_results)

    if pd is not None:
        df = pd.DataFrame([
            {
                "Run": r["run_name"],
                "Status": r["status"],
                "Faithfulness": r.get("scores", {}).get("faithfulness"),
                "Answer Relevancy": r.get("scores", {}).get("answer_relevancy"),
                "Context Precision": r.get("scores", {}).get("context_precision"),
                "Context Recall": r.get("scores", {}).get("context_recall"),
                "Mean Score": r.get("mean_score", 0),
            }
            for r in all_results
        ])
        return df

    return all_results


def print_results_table(results: List[Dict]):
    """Print a formatted comparison table of all experiment results."""
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    header = f"{'Run':<35} {'Faith':>8} {'AnsRel':>8} {'CtxPrec':>8} {'CtxRec':>8} {'Mean':>8}"
    print(header)
    print("-" * 80)

    for r in results:
        if r["status"] == "failed":
            print(f"{r['run_name']:<35} {'FAILED':>42}")
            continue

        scores = r.get("scores", {})
        faith = scores.get("faithfulness")
        ans_rel = scores.get("answer_relevancy")
        ctx_prec = scores.get("context_precision")
        ctx_rec = scores.get("context_recall")
        mean = r.get("mean_score", 0)

        def fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        print(
            f"{r['run_name']:<35} {fmt(faith):>8} {fmt(ans_rel):>8} "
            f"{fmt(ctx_prec):>8} {fmt(ctx_rec):>8} {fmt(mean):>8}"
        )

    print("=" * 80)


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        pd = None

    parser = argparse.ArgumentParser(description="Run RAG pipeline experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENT_GRID.keys()),
        help="Specific experiments to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--questions", default="evaluation/test_questions.json", help="Test questions path"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    experiments_to_run = (
        list(EXPERIMENT_GRID.keys()) if args.all
        else (args.experiments or ["chunk_size", "top_k_retrieval"])
    )

    results_df = run_experiment_suite(
        experiment_names=experiments_to_run,
        config_path=args.config,
        questions_path=args.questions,
        use_mlflow=not args.no_mlflow,
    )

    if pd is not None and hasattr(results_df, 'to_csv'):
        csv_path = "experiments/results/summary.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to {csv_path}")
