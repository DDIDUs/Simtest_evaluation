import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
        "base_url": "http://129.254.177.83:8085/v1",
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
) -> None:
    client = build_client(model_cfg)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    strategies = {
        "greedy": {"temperature": 0.2, "top_p": 0.95},
        "neuclus": {"temperature": 0.7, "top_p": 0.95},
    }
    if sampling_subset:
        strategies = {k: v for k, v in strategies.items() if k in sampling_subset}

    for strategy, sampling in strategies.items():
        raw_dict: Dict[str, List[str]] = {}
        code_dict: Dict[str, List[str]] = {}

        tasks = []
        for problem in problems:
            task_id = problem["task_id"]
            prompt = GENERATION_PROMPT_TMPL.format(prompt=problem["prompt"])
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
                        raw_dict,
                        code_dict,
                    )
                )
            )

        await asyncio.gather(*tasks)

        out_path = output_dir / model_name / f"{strategy}_code_generate.json"
        save_json(str(out_path), {"raw": raw_dict, "code": code_dict})
        logging.info("Saved %s", out_path)


async def process_problem(
    task_id: str,
    prompt: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    model_cfg: Dict,
    raw_store: Dict[str, List[str]],
    code_store: Dict[str, List[str]],
) -> None:
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

    raw_store[task_id] = [raw_resp or ""]
    code_store[task_id] = extract_code_blocks(raw_resp or "")


async def main(
    dataset_path: Optional[str],
    output_root: str,
    limit: Optional[int],
    models: Dict[str, Dict],
    sampling_subset: Optional[List[str]] = None,
) -> None:
    problems = load_bigcodebench_hard(dataset_path)
    if not problems:
        raise RuntimeError("No problems found to generate code for.")
    if limit is not None:
        problems = problems[:limit]

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in models.items():
        await generate_for_model(model_name, cfg, problems, output_dir, sampling_subset)


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
        help="Subset of strategies to run (greedy, neuclus).",
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

    asyncio.run(main(args.dataset, args.out_dir, args.limit, models, args.sampling))
