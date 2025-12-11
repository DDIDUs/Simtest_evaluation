
import asyncio
import base64
import json
import logging
import pickle
import re
import zlib
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiofiles
from openai import AsyncOpenAI
from tqdm import tqdm
from datasets import load_dataset

from template import CODECONTEST_PROMPT_TMPL
from utils import *

# ========= 설정 =========
MAX_CONCURRENT_TASKS = 5
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
DEFAULT_OUT_PATH = "./test.jsonl"
DEFAULT_PROBLEM_JSON = "/home/ysy/icst2026/Simtest_evaluation/LiveCodeBench/BaseLine/results/Qwen3-Coder-30B-A3B-Instruct/Qwen3-Coder-30B-A3B-Instruct/Scenario.codegeneration_1_0.2.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="log_code_eval.txt",
    filemode="a",
)

# ========= LLM 호출 =========
async def gen_code(prompt: str, llm: AsyncOpenAI, sem: asyncio.Semaphore) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = await asyncio.wait_for(
                llm.chat.completions.create(
                    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    top_p=0.8,
                    max_tokens=60000,
                    extra_body={"repetition_penalty": 1.05},
                ),
                timeout=TIMEOUT_SECONDS,
            )
            content = resp.choices[0].message.content or ""
            return extract_block(content)
        except asyncio.TimeoutError:
            logging.warning(f"gen_code timeout (attempt {attempt})")
        except Exception as e:
            logging.error(f"gen_code exception (attempt {attempt}): {e}")
    return None


# ========= Code-Contests-Plus 로드 및 인덱싱 =========
def load_codecontests_index() -> Dict[str, dict]:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_latest",
    )
    dataset = ds["train"] if "train" in ds else ds

    index: Dict[str, dict] = {}
    for row in dataset:
        qid = row.get("question_id") or row.get("id") or row.get("problem_id")
        if qid is None:
            continue
        qid = str(qid)
        index[qid] = row

    logging.info(f"Indexed livecodebench rows: {len(index)}")
    return index


# ========= 개별 문제 + code_list 처리 =========
async def handle_problem_item(
    problem: dict,
    qid_to_row: Dict[str, dict],
    llm: AsyncOpenAI,
    sem: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    out_path: str,
) -> bool:
    async with sem:
        try:
            question_id = str(problem.get("question_id"))
            if not question_id:
                return False

            row = qid_to_row.get(question_id)
            if not row:
                logging.warning(f"No matching Code-Contests row for question_id={question_id}")
                return False

            # 1) 테스트케이스 문자열 리스트 생성
            eval_sample = build_evaluation_sample(row)
            testcase_list: List[str] = eval_sample["testcases"]
            if not testcase_list:
                logging.warning(f"No testcases for question_id={question_id}")
                return False

            code_list: List[str] = problem.get("code_list") or []
            if not code_list:
                # 코드가 없으면 평가 불가
                return False

            code_results: List[dict] = []

            # 2) code_list의 각 코드에 대해 모든 테스트케이스 평가
            for code_idx, code in enumerate(code_list):
                per_test_results: List[dict] = []

                for tc_idx, tc in enumerate(testcase_list):
                    prompt = CODECONTEST_PROMPT_TMPL.format(
                        code=code,
                        testcase=tc,
                    )
                    llm_output = await gen_code(prompt, llm, sem)
                    if not llm_output:
                        result = None
                    else:
                        result = extract_pass_fail(llm_output)

                    per_test_results.append(
                        {
                            "testcase_index": tc_idx,
                            "pass_fail": result,   # "PASS" / "FAIL" / None
                            "raw_response": llm_output,
                        }
                    )

                overall_pass: Optional[bool]
                if any(r["pass_fail"] == "FAIL" for r in per_test_results):
                    overall_pass = False
                elif all(r["pass_fail"] == "PASS" for r in per_test_results):
                    overall_pass = True
                else:
                    overall_pass = None  # 일부는 PASS/FAIL, 일부는 None 등 애매한 경우

                code_results.append(
                    {
                        "code_index": code_idx,
                        "overall_pass": overall_pass,
                        "pass_fail_list": [r["pass_fail"] for r in per_test_results],
                    }
                )

            # 3) 문제 단위 결과를 JSONL에 기록
            out = {
                "question_id": question_id,
                "question_title": problem.get("question_title"),
                "platform": problem.get("platform"),
                "contest_id": problem.get("contest_id"),
                "difficulty": problem.get("difficulty"),
                "num_testcases": len(testcase_list),
                "num_codes": len(code_list),
                "code_results": code_results,
            }

            async with file_lock:
                async with aiofiles.open(out_path, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(out, ensure_ascii=False) + "\n")

            return True

        except Exception as e:
            logging.error(f"handle_problem_item exception: {e}")
            return False


# ========= 메인 파이프라인 =========
async def eval_codes_from_json(
    llm: AsyncOpenAI,
    problem_json_path: str = DEFAULT_PROBLEM_JSON,
    out_path: str = DEFAULT_OUT_PATH,
) -> int:
    # 0. 문제 JSON 로드 (리스트 형태 가정)
    with open(problem_json_path, "r", encoding="utf-8") as f:
        problems: List[dict] = json.load(f)

    if not isinstance(problems, list):
        raise ValueError("problem_json 파일은 JSON 리스트 형식이어야 합니다.")

    # 1. Code-Contests-Plus 인덱스 생성 (question_id -> row)
    qid_to_row = load_codecontests_index()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    file_lock = asyncio.Lock()

    coros = [
        handle_problem_item(problem, qid_to_row, llm, sem, file_lock, out_path)
        for problem in problems
    ]

    ok_cnt = 0
    pbar = tqdm(total=len(coros), desc="Evaluate codes from code_list", unit="problem")
    try:
        for fut in asyncio.as_completed(coros):
            if await fut:
                ok_cnt += 1
            pbar.update(1)
    finally:
        pbar.close()

    logging.info(f"Completed problems: {len(coros)}, Saved: {ok_cnt}")
    return ok_cnt


# ========= 실행 =========
llm = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8086/v1")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problems",
        type=str,
        default=DEFAULT_PROBLEM_JSON,
        help="Input JSON path (list of problems with code_list)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_PATH,
        help="Output JSONL path",
    )
    args = parser.parse_args()

    saved = asyncio.run(
        eval_codes_from_json(
            llm,
            problem_json_path=args.problems,
            out_path=args.out,
        )
    )
    print(f"Saved problem results: {saved}")
