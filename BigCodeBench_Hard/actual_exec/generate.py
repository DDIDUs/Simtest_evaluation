import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from template import GENERATION_PROMPT_TMPL
from utils import extract_code_blocks, load_bigcodebench_hard, save_json


MAX_CONCURRENT_TASKS = 5
TIMEOUT_SECONDS = 120
MAX_RETRIES = 2


DEFAULT_MODELS: Dict[str, Dict] = {
    "qwen3-coder-30B-A3B-instruct": {
        "api_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "api_key": "EMPTY",
        "base_url": "http://129.254.222.36:8000/v1",
        "extra_body": {"repetition_penalty": 1.05},
    },
    "llama3.2:1b": { # for testing code in local
        "api_model": "llama3.2:1b",
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "extra_body": {"repetition_penalty": 1.05},
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=Path(__file__).parent / "log_code_generate.txt",
    filemode="a",
)


async def call_llm(
    prompt: str,
    llm: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Optional[Dict] = None,
) -> Optional[str]:
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                resp = await asyncio.wait_for(
                    llm.chat.completions.create(**kwargs),
                    timeout=TIMEOUT_SECONDS,
                )
                return resp.choices[0].message.content or ""
            except asyncio.TimeoutError:
                logging.warning("call_llm timeout (attempt %s)", attempt)
            except Exception as exc:
                logging.error("call_llm error (attempt %s): %s", attempt, exc)
        return None


def build_client(model_cfg: Dict) -> AsyncOpenAI:
    api_key = model_cfg.get("api_key")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=model_cfg.get("base_url"),
    )


async def generate_for_model(
    model_name: str,
    model_cfg: Dict,
    problems: List[Dict[str, str]],
    output_dir: Path,

    sampling_subset: Optional[List[str]] = None,
    n_samples: int = 1,
) -> None:
    client = build_client(model_cfg)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    strategies = {
        "greedy": {"temperature": 0.2, "top_p": 0.95},
        "nucleus": {"temperature": 0.7, "top_p": 0.95},
    }
    if sampling_subset:
        strategies = {k: v for k, v in strategies.items() if k in sampling_subset}

    for strategy, sampling in strategies.items():
        # Initialize results storage
        raw_dict: Dict[str, List[str]] = {}
        code_dict: Dict[str, List[str]] = {}
        
        # Output path
        out_path = output_dir / model_name / f"{strategy}_code_generate.json"
        
        # Load existing progress if file exists (optional, but good for resuming)
        if out_path.exists():
            try:
                import json
                with open(out_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    raw_dict = existing_data.get("raw", {})
                    code_dict = existing_data.get("code", {})
                    logging.info("Resuming from existing file with %d tasks completed", len(raw_dict))
            except Exception as e:
                logging.warning(f"Could not load existing file {out_path}: {e}")

        # Create tasks
        tasks = []
        for problem in problems:
            task_id = problem["task_id"]
            
            # Skip if already completed (all samples generated)
            if task_id in raw_dict and len(raw_dict[task_id]) >= n_samples:
                continue
                
            prompt = GENERATION_PROMPT_TMPL.format(prompt=problem["prompt"])
            
            # Create n_samples tasks for each problem
            # We need to know how many we need to generate to reach n_samples
            current_count = len(raw_dict.get(task_id, []))
            needed = n_samples - current_count
            
            for _ in range(needed):
                tasks.append(
                    asyncio.create_task(
                        process_problem(
                            task_id,
                            prompt,
                            client,
                            semaphore,
                            sampling["temperature"],
                            sampling["top_p"],
                            model_cfg,
                        )
                    )
                )
        
        if not tasks:
            logging.info("All tasks already completed for strategy %s", strategy)
            continue

        logging.info(f"Starting {len(tasks)} tasks for strategy {strategy}")
        
        # Process results as they complete
        completed_count = 0
        save_interval = 50
        
        for coro in asyncio.as_completed(tasks):
            task_id, raw_resp, extracted_codes = await coro
            
            if task_id not in raw_dict:
                raw_dict[task_id] = []
                code_dict[task_id] = []
            
            if raw_resp:
                raw_dict[task_id].append(raw_resp)
            if extracted_codes:
                code_dict[task_id].extend(extracted_codes)
                
            completed_count += 1
            
            # Incremental save
            if completed_count % save_interval == 0:
                save_json(str(out_path), {"raw": raw_dict, "code": code_dict})
                logging.info(f"Saved progress: {completed_count}/{len(tasks)} tasks completed")

        # Final save
        save_json(str(out_path), {"raw": raw_dict, "code": code_dict})
        logging.info("Saved final results to %s", out_path)


async def process_problem(
    task_id: str,
    prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    model_cfg: Dict,
) -> Tuple[str, Optional[str], List[str]]:
    raw_resp = await call_llm(
        prompt=prompt,
        llm=client,
        model_name=model_cfg["api_model"],
        semaphore=semaphore,
        temperature=temperature,
        top_p=top_p,
        max_tokens=model_cfg.get("max_tokens", 4096),
        extra_body=model_cfg.get("extra_body"),
    )



    extracted = extract_code_blocks(raw_resp or "")
    return task_id, raw_resp, extracted


async def main(
    dataset_path: Optional[str],
    output_root: str,
    limit: Optional[int],
    models: Dict[str, Dict],
    sampling_subset: Optional[List[str]] = None,
    n_samples: int = 1,
) -> None:
    problems = load_bigcodebench_hard(dataset_path)
    if not problems:
        raise RuntimeError("No problems found to generate code for.")
    if limit is not None:
        problems = problems[:limit]

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in models.items():
        await generate_for_model(model_name, cfg, problems, output_dir, sampling_subset, n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON/JSONL. If omitted, load from HuggingFace bigcodebench-hard.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).parent / "results"),
        help="Root output directory for generated code.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of problems to generate (e.g., 3 for dry run).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of models to run (default: all defaults).",
    )
    parser.add_argument(
        "--local_free",
        action="store_true",
        help="Use an additional local/free OpenAI-compatible endpoint for dry run.",
    )
    parser.add_argument(
        "--local_model_name",
        type=str,
        default="local-free",
        help="Name key for the local/free model.",
    )
    parser.add_argument(
        "--local_api_model",
        type=str,
        default="qwen2.5-coder-3b-instruct",
        help="Model ID served by the local/free endpoint.",
    )
    parser.add_argument(
        "--local_base_url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL of the local/free OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--local_api_key",
        type=str,
        default="EMPTY",
        help="API key for the local/free endpoint (if required).",
    )
    parser.add_argument(
        "--local_max_tokens",
        type=int,
        default=4096,
        help="max_tokens for the local/free endpoint.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        nargs="*",
        default=None,
        help="Subset of strategies to run (greedy, nucleus).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate per problem (default: 1).",
    )
    args = parser.parse_args()

    models = dict(DEFAULT_MODELS)
    # Optionally inject a local/free model for dry runs
    if args.local_free:
        models[args.local_model_name] = {
            "api_model": args.local_api_model,
            "api_key": args.local_api_key,
            "base_url": args.local_base_url,
            "max_tokens": args.local_max_tokens,
        }

    # Allow selecting subset
    if args.models:
        selected = {}
        for m in args.models:
            if m in models:
                selected[m] = models[m]
            else:
                raise ValueError(f"Unknown model '{m}'. Available: {list(models.keys())}")
        models = selected

    asyncio.run(main(args.dataset, args.out_dir, args.limit, models, args.sampling, args.n_samples))
