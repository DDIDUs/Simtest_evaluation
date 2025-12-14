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
# Load .env from parent directory (root of BigCodeBench_Hard)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm

from utils import load_generated_codes, split_test_cases, construct_prompt, load_bigcodebench_hard

# Configure Logging
# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("log_test_eval.txt"),
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
        "max_tokens": 1024,
        "temperature": 0.2,
    },
    "gpt-4o-2024-08-06": {
        "api_model": "gpt-4o-2024-08-06",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": None,  # Use default OpenAI URL
        "max_tokens": 1024,
        "temperature": 0.2,
    }
}

MAX_CONCURRENT_REQUESTS = 5 # Reduced from 20 to 5 for stability
TIMEOUT_SECONDS = 60 # Increased from 30 to 60 for long generation

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
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_config["api_model"],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=model_config["temperature"],
                        max_tokens=model_config["max_tokens"],
                        extra_body=model_config.get("extra_body"),
                    ),
                    timeout=TIMEOUT_SECONDS
                )
                latency = time.time() - start_time
                content = response.choices[0].message.content
                return content, latency
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
    # Case insensitive for robustness
    match = re.search(r"\[Result\].*?```plaintext\s*(PASS|FAIL|\[PASS\]|\[FAIL\])\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        result = match.group(1).upper()
        if "PASS" in result:
            return "PASS"
        if "FAIL" in result:
            return "FAIL"

    # Fallback looser match within [Result] section?
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
        # logging.info(f"Starting test case: {test_name} for task {task_id}") 
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
             # call_llm in 1_pred returns (content, latency) or (None, 0.0) if it caught its own exception
             # But if call_llm crashed, res would be exception.

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
    # Load Data
    logging.info("Loading generated codes...")
    gen_data = load_generated_codes(generated_code_path)
    # gen_data is expected to be {"code": {task_id: [code1, code2...], ...}}
    if "code" in gen_data:
        code_map = gen_data["code"]
    else:
        # Maybe it's directly the dict?
        code_map = gen_data
    
    logging.info("Loading test cases (BigCodeBench Hard)...")
    problems_data = load_bigcodebench_hard()
    
    # Filter/Map problems
    problems = []

    for item in problems_data:
        t_id = item['task_id']
        if t_id in code_map:
            codes = code_map[t_id]
            # Always select the first code (index 0) as per user request
            if isinstance(codes, list) and len(codes) > 0:
                selected_idx = 0
                selected_code = codes[0]
            else:
                # Handle edge case where it might not be a list or list is empty
                selected_idx = 0
                selected_code = codes if isinstance(codes, str) else codes[0]

            problems.append({
                "task_id": t_id,
                "code": selected_code, 
                "code_index": selected_idx,
                "test_code": item['test']  # The full unittest class
            })
    
    if limit:
        problems = problems[:limit]
        
    logging.info(f"Evaluating {len(problems)} problems...")
    
    # LLM Client
    # LLM Client
    api_key = model_config.get("api_key")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") # Fallback to env var if config was None (though config uses getenv already)
    
    if not api_key:
        logging.error(f"API Key for model {model_name} is missing. Checked 'OPENAI_API_KEY' in env.")
        logging.error(f"Ensure .env file at {env_path} contains OPENAI_API_KEY.")
        raise ValueError("API Key missing")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=model_config["base_url"]
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Execution Loop
    output_lines = []
    
    # We can run problems in parallel too, but let's do sequential problems, parallel test cases within problem
    # or fully parallel? Fully parallel might be too much context switching, but let's try batching or simple gather.
    # Given the number of test cases could be high, let's process problems sequentially or in small chunks.
    
    progress_bar = tqdm(total=len(problems), desc="Evaluating")
    
    # Output Directory Setup
    # requested structure: results/<model_name>/test.jsonl
    # If output_path is provided as a full path, use it? 
    # User asked for auto-creation. Let's make output_path default to "test.jsonl" and use a root results dir.
    
    # We will treat output_path as the filename (test.jsonl)
    # And we need a results root dir.
    
    results_root = Path(__file__).parent / "results"
    output_dir = results_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_output_path = output_dir / output_file
    
    # Raw output filename: test.jsonl -> test_raw.jsonl
    raw_output_filename = output_file.replace(".jsonl", "_raw.jsonl")
    if raw_output_filename == output_file:
        raw_output_filename += "_raw.jsonl" # Fallback if no extension
    
    final_raw_output_path = output_dir / raw_output_filename
    
    # Check for existing progress
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
        
    # Filter problems
    problems_to_run = [p for p in problems if p['task_id'] not in processed_ids]
    
    if not problems_to_run:
        logging.info("All problems already processed. Exiting.")
        return

    logging.info(f"Evaluating {len(problems_to_run)} problems (Skipped {len(processed_ids)})...")
    
    # Open files for appending
    f_out = open(final_output_path, 'a', encoding='utf-8')
    f_raw = open(final_raw_output_path, 'a', encoding='utf-8')
    
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
            # Wrap in a task to ensure it's scheduled? 
            # Actually as_completed takes a list of awaitables.
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
            
            # Write to files
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            
            f_raw.write(json.dumps(raw_data) + "\n")
            f_raw.flush()
            
            progress_bar.update(1)
            
        progress_bar.close()
        
    finally:
        f_out.close()
        f_raw.close()
    logging.info(f"Done. Results saved to {final_output_path}")
    logging.info(f"Raw responses saved to {final_raw_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_code", type=str, required=True, help="Path to generated code JSON")
    parser.add_argument("--output_file", type=str, default="test.jsonl", help="Filename for output (default: test.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems for testing")
    parser.add_argument("--model", type=str, default="qwen3-coder-30B-A3B-instruct", choices=list(MODEL_CONFIGS.keys()), help="Model to use for prediction")
    args = parser.parse_args()
    
    asyncio.run(main(args.generated_code, args.output_file, args.model, args.limit))
