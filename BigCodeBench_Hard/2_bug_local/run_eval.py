import argparse
import asyncio
import json
import logging
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
import re

from dotenv import load_dotenv

# Load .env from parent directory (root of BigCodeBench_Hard)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from utils import load_generated_codes, split_test_cases, construct_prompt, load_bigcodebench_hard

# Configure Logging
log_file_path = Path(__file__).resolve().parent / "log_bug_local_eval.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(log_file_path)),
        logging.StreamHandler()
    ]
)

# Model Configurations
MODEL_CONFIGS = {
    "qwen3-coder-30B-A3B-instruct": {
        "api_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "api_key": "EMPTY",
        "base_url": "http://129.254.222.36:8000/v1",
        "extra_body": {"repetition_penalty": 1.05},
        "max_tokens": 2048,
        "temperature": 0.2, # As per template, maybe 0.2? Or low temp for deterministic eval. 1_pred used 0.2
    },
    "gpt-5-mini-2025-08-07": {
        "api_model": "gpt-5-mini-2025-08-07",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": None,  # Use default OpenAI URL
        "max_completion_tokens": 2048,
        "temperature": 1,
    },
    "claude-haiku-4-5-20251001": {
        "api_model": "claude-haiku-4-5-20251001",
        "api_key": os.getenv("CLAUDE_API_KEY"),
        "base_url": None,
        "max_tokens": 2048,
        "temperature": 0.2,
    }
}

MAX_CONCURRENT_REQUESTS = 3
TIMEOUT_SECONDS = 120

async def call_llm(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_config: Dict
) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                start_time = time.time()
                kwargs = {
                    "model": model_config["api_model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": model_config["temperature"],
                    "extra_body": model_config.get("extra_body"),
                }
                if "max_completion_tokens" in model_config:
                    kwargs["max_completion_tokens"] = model_config["max_completion_tokens"]
                else:
                    kwargs["max_tokens"] = model_config["max_tokens"]

                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=TIMEOUT_SECONDS
                )
                latency = time.time() - start_time
                content = response.choices[0].message.content
                logging.info(f"LLM call successful for prompt length {len(prompt)}")
                return content, latency
            except asyncio.TimeoutError:
                 logging.warning(f"Attempt {attempt + 1} timed out after {TIMEOUT_SECONDS}s")
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return None, 0.0

def parse_result(response: Optional[str]) -> str:
    if response is None:
        return "NULL"
    
    if not response.strip():
        return "NULL"

    # Pattern: [Result] followed by ```plaintext ... ``` containing PASS or FAIL
    # Case insensitive for robustness. 
    # [Result] tag might be missing, so we look for the code block anywhere.
    match = re.search(r"```plaintext\s*(PASS|FAIL|\[PASS\]|\[FAIL\])\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        result = match.group(1).upper()
        if "PASS" in result:
            return "PASS"
        if "FAIL" in result:
            return "FAIL"

    # Fallback: Look for [Result] section if code block missing
    if "[Result]" in response:
        part = response.split("[Result]")[1]
        # Look for the next section header e.g. [Bug Localization] or end of string
        end_idx = part.find("[Bug Localization]")
        if end_idx != -1:
            part = part[:end_idx]
        
        if "PASS" in part and "FAIL" not in part:
            return "PASS"
        if "FAIL" in part and "PASS" not in part:
            return "FAIL"
    
    # Ultimate fallback: if just "PASS" or "FAIL" is the entire content (trimmed)
    cleaned = response.strip().upper()
    if cleaned in ["PASS", "[PASS]"]:
        return "PASS"
    if cleaned in ["FAIL", "[FAIL]"]:
        return "FAIL"
            
    return "NULL"

async def evaluate_task(
    task_id: str,
    code: str,
    test_cases: List[tuple], # (name, code) tuples
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_config: Dict,
    code_index: int = 0
) -> Dict:
    
    tasks = []
    test_names = []
    
    for test_name, test_code in test_cases:
        prompt = construct_prompt(code, test_code)
        logging.info(f"Starting test case: {test_name} for task {task_id}")
        tasks.append(call_llm(client, prompt, semaphore, model_config))
        test_names.append(test_name)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    pass_fail_dict = {}
    latency_dict = {}
    raw_response_dict = {}

    for i, res in enumerate(results):
        name = test_names[i]
        if isinstance(res, Exception):
            logging.error(f"Test {name} failed with exception: {res}")
            content, latency = None, 0.0
        else:
             content, latency = res

        result = parse_result(content)
        pass_fail_dict[name] = result
        latency_dict[name] = latency
        raw_response_dict[name] = content if content is not None else None
        
    overall_pass = "PASS" if all(r == "PASS" for r in pass_fail_dict.values()) else "FAIL"
    
    return {
        "id": task_id,
        "num_testcases": len(test_cases),
        "num_codes": 1,
        "code_index": code_index,
        "overall_pass": overall_pass,
        "pass_fail_list": pass_fail_dict,
        "latency_list": latency_dict,
        "raw_responses": raw_response_dict,
        "code_results": [{ 
            "code_index": code_index,
            "overall_pass": overall_pass == "PASS", 
        }]
    }

async def main(
    generated_code_path: str,
    output_file: str,
    model_name: str,
    limit: Optional[int] = None
):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_name]
    
    logging.info("Loading generated codes...")
    gen_data = load_generated_codes(generated_code_path)
    # gen_data structure check: {"code": {task_id: [code1...]}, ...} or {"task_id": [code1...]}
    # The file we viewed `nucleus_code_generate.json` had a "raw" key at the top level?
    # View showed: {"raw": {"BigCodeBench/13": [...]}}
    # So we need to handle that.
    
    code_map = {}
    if "raw" in gen_data:
        code_map = gen_data["raw"]
    elif "code" in gen_data:
        code_map = gen_data["code"] # Fallback to 1_pred assumption
    else:
        # Check if root is correct
        code_map = gen_data

    # Ensure code_map is {id: [codes]}
    
    logging.info("Loading test cases (BigCodeBench Hard)...")
    problems_data = load_bigcodebench_hard()
    
    problems = []

    for item in problems_data:
        t_id = item['task_id']
        if t_id in code_map:
            codes = code_map[t_id]
            # Select 1st code as requested: "code" 중 1번째 코드를 가져오고
            selected_code = None
            selected_idx = 0
            
            if isinstance(codes, list) and len(codes) > 0:
                selected_code = codes[0]
            elif isinstance(codes, str):
                selected_code = codes
            
            if selected_code:
                 problems.append({
                    "task_id": t_id,
                    "code": selected_code, 
                    "code_index": selected_idx,
                    "test_code": item['test']
                })
    
    if limit:
        problems = problems[:limit]
        
    logging.info(f"Evaluating {len(problems)} problems...")
    
    # LLM Client
    api_key = model_config.get("api_key")
    if not api_key or api_key == "EMPTY":
        # Check env if not EMPTY (EMPTY is for vllm typically)
        if model_config["api_model"] != "Qwen/Qwen3-Coder-30B-A3B-Instruct": 
             env_key = os.getenv("OPENAI_API_KEY")
             if env_key:
                 api_key = env_key

    client = AsyncOpenAI(
        api_key=api_key if api_key else "EMPTY",
        base_url=model_config["base_url"]
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Output Setup
    results_root = Path(__file__).parent / "results"
    output_dir = results_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_output_path = output_dir / output_file
    
    raw_output_filename = output_file.replace(".jsonl", "_raw.jsonl")
    if raw_output_filename == output_file:
         raw_output_filename += "_raw.jsonl"
    final_raw_output_path = output_dir / raw_output_filename
    
    processed_ids = set()
    if final_output_path.exists():
        with open(final_output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue
    
    if processed_ids:
        logging.info(f"Found {len(processed_ids)} processed problems. Resuming...")
        
    problems_to_run = [p for p in problems if p['task_id'] not in processed_ids]
    
    if not problems_to_run:
        logging.info("All problems already processed. Exiting.")
        return

    # Open files for appending
    f_out = open(final_output_path, 'a', buffering=1, encoding='utf-8')
    f_raw = open(final_raw_output_path, 'a', buffering=1, encoding='utf-8')
    
    try:
        # Create all coroutines
        pending_tasks = []
        for prob in problems_to_run:
            split_tests = split_test_cases(prob['test_code'])
            
            coro = evaluate_task(
                prob['task_id'],
                prob['code'],
                split_tests,
                client,
                semaphore,
                model_config,
                code_index=prob['code_index']
            )
            # Wrap in a task to ensure it's scheduled correctly if needed, but simple list is fine for as_completed
            pending_tasks.append(coro)

        progress_bar = tqdm(total=len(pending_tasks), desc="Evaluating")
        
        # Process as they complete
        for coro in asyncio.as_completed(pending_tasks):
            result = await coro
            
            # Extract raw responses
            raw_data = {
                "id": result["id"],
                "raw_responses": result.pop("raw_responses")
            }
            
            # Write key results to main file
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            
            # Write raw responses to raw file
            f_raw.write(json.dumps(raw_data) + "\n")
            f_raw.flush()
            
            progress_bar.update(1)
            
        progress_bar.close()
        
    finally:
        f_out.close()
        f_raw.close()
    
    logging.info(f"Done. Results saved to {final_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_code", type=str, required=True, help="Path to generated code JSON")
    parser.add_argument("--output_file", type=str, default="test.jsonl", help="Filename for output (default: test.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems for testing")
    parser.add_argument("--model", type=str, default="qwen3-coder-30B-A3B-instruct", choices=list(MODEL_CONFIGS.keys()), help="Model to use for prediction")
    args = parser.parse_args()
    
    asyncio.run(main(args.generated_code, args.output_file, args.model, args.limit))
