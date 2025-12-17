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
import sys

from dotenv import load_dotenv

# Load .env from parent directory (root of BigCodeBench_Hard)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

from utils import load_generated_codes, split_test_cases, load_bigcodebench_hard
from template import BUG_REPORT_PROMPT_TMPL

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "log_test_eval.txt"),
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
        "max_tokens": 8192,
        "temperature": 0.0,
    },
    "gpt-5-mini-2025-08-07": {
        "api_model": "gpt-5-mini-2025-08-07",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": None,  # Use default OpenAI URL
        "max_completion_tokens": 8192,
        "temperature": 1,
    },
    "claude-haiku-4-5-20251001": {
        "api_model": "claude-haiku-4-5-20251001",
        "api_key": os.getenv("CLAUDE_API_KEY"),
        "base_url": None,
        "max_tokens": 8192,
        "temperature": 0.0,
        "client_type": "anthropic"
    }
}

MAX_CONCURRENT_REQUESTS = 5
TIMEOUT_SECONDS = 60

async def call_llm(
    client: any,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model_config: Dict
) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                start_time = time.time()
                if model_config.get("client_type") == "anthropic":
                    # Claude: support top_p or temperature
                    kwargs = {
                        "model": model_config["api_model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": model_config.get("max_tokens", 1024),
                    }
                    if "top_p" in model_config and model_config["top_p"] is not None:
                        kwargs["top_p"] = model_config["top_p"]
                    
                    if "temperature" in model_config and model_config["temperature"] is not None:
                        kwargs["temperature"] = model_config["temperature"]

                    response = await asyncio.wait_for(
                        client.messages.create(**kwargs),
                        timeout=TIMEOUT_SECONDS
                    )
                    latency = time.time() - start_time

                    content = response.content[0].text if response.content else ""
                    if not content or not content.strip():
                        logging.warning(f"Anthropic response content is empty! Prompt length: {len(prompt)}")
                    logging.info(f"Anthropic call finished in {latency:.2f}s")
                    return content, latency
                else:
                    # OpenAI style
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
                    if not content or not content.strip():
                        logging.warning(f"OpenAI response content is empty! Prompt length: {len(prompt)}")
                        if hasattr(response.choices[0], 'finish_reason'):
                             logging.warning(f"Finish reason: {response.choices[0].finish_reason}")
                    return content, latency
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return None, 0.0

def construct_bug_report_prompt(code, test_case):
    return BUG_REPORT_PROMPT_TMPL.format(code=code, testcase=test_case)

def parse_result(response: Optional[str]) -> str:
    if response is None:
        return "NULL"

    if not response.strip():
        return "NULL"

    # Normalize
    response_upper = response.upper()

    # 1. Targeted Extraction: [Result] ... [Bug Localization] or end
    # Regex captures content starting from [Result] until [Bug Localization], [Explanation], or End
    targeted_match = re.search(r"\[RESULTS?\]([\s\S]*?)(?:\[BUG LOCALIZATION\]|\[EXPLANATION\]|$)", response_upper)
    if targeted_match:
        section_content = targeted_match.group(1)
        # Check for explicit tags first in this section
        if "[PASS]" in section_content: return "PASS"
        if "[FAIL]" in section_content: return "FAIL"
        
        # Check for independent PASS/FAIL words
        if re.search(r"\bPASS\b", section_content): return "PASS"
        if re.search(r"\bFAIL\b", section_content): return "FAIL"

    # 2. Code Block fallback
    match = re.search(r"```(?:plaintext|text)?\s*[\r\n]+(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
    if match:
        block_content = match.group(1).upper()
        if "[PASS]" in block_content or "PASS" in block_content.split():
            return "PASS"
        if "[FAIL]" in block_content or "FAIL" in block_content.split():
            return "FAIL"

    # 3. Global explicit tags (high confidence)
    if "[PASS]" in response_upper:
        return "PASS"
    if "[FAIL]" in response_upper:
        return "FAIL"

    # 4. Fallback: last occurrence of PASS/FAIL
    candidates = re.findall(r"\b(PASS|FAIL)\b", response_upper)
    if candidates:
        return candidates[-1]

    return "NULL"

async def evaluate_task(
    task_id: str,
    code: str,
    test_cases: List[tuple], # (name, code) tuples
    client: any,
    semaphore: asyncio.Semaphore,
    model_config: Dict,
    code_index: int = 0
) -> Dict:
    pass_fail_list = []
    
    tasks = []
    test_names = []
    
    for test_name, test_code in test_cases:
        prompt = construct_bug_report_prompt(code, test_code)
        tasks.append(call_llm(client, prompt, semaphore, model_config))
        test_names.append(test_name)
    
    results = await asyncio.gather(*tasks)
    
    pass_fail_dict = {}
    latency_dict = {}
    raw_response_dict = {}

    for i, (content, latency) in enumerate(results):
        name = test_names[i]
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
    if "code" in gen_data:
        code_map = gen_data["code"]
    else:
        code_map = gen_data
    
    logging.info("Loading test cases (BigCodeBench Hard)...")
    problems_data = load_bigcodebench_hard()
    
    problems = []

    for item in problems_data:
        t_id = item['task_id']
        if t_id in code_map:
            codes = code_map[t_id]
            # Use 1st code (index 0)
            if isinstance(codes, list) and len(codes) > 0:
                selected_idx = 0
                selected_code = codes[0]
            else:
                selected_idx = 0
                selected_code = codes if isinstance(codes, str) else codes[0]

            problems.append({
                "task_id": t_id,
                "code": selected_code, 
                "code_index": selected_idx,
                "test_code": item['test']
            })
    
    if limit:
        problems = problems[:limit]
        
    logging.info(f"Evaluating {len(problems)} problems...")
    
    client_type = model_config.get("client_type", "openai")
    
    if client_type == "anthropic":
        api_key = model_config.get("api_key")
        if not api_key:
             logging.error(f"API Key for Anthropic model {model_name} is missing. Checked 'CLAUDE_API_KEY' in env.")
             raise ValueError("API Key missing")
        client = AsyncAnthropic(
            api_key=api_key,
        )
    else:
        api_key = model_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("API Key missing.")
            raise ValueError("API Key missing")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=model_config["base_url"]
        )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Output Directory Setup
    results_root = Path(__file__).resolve().parent.parent / "results" # Using shared results dir logic from 1_pred? 
    # User said: "results/{모델이름}/ 폴더에 ... BigCodeBench_Hard/1_pred/results/qwen3-coder-30B-A3B-instruct/test.jsonl 를 참고해서"
    # Wait, the user instruction implies the results should be under 3_bug_report/results/... ?
    # "BigCodeBench_Hard/1_pred/results/... 참고해서 동일한 형태로 저장되게 해줘. ... 과정을 log하는 파일은 3_bug_report 아래에 저장"
    # Usually results are local to the folder (1_pred/results, 3_bug_report/results).
    # Let's save to 3_bug_report/results/{model_name}/
    
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

    progress_bar = tqdm(total=len(problems_to_run), desc="Evaluating")

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
