import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiofiles
from datasets import load_dataset
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tqdm import tqdm

from template import *
from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="log_code_eval1.txt",
    filemode="w",
)

MAX_CONCURRENT_TASKS = 5
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2

from json import JSONDecoder

async def load_done_qids_from_outfile(out_path: str) -> Set[str]:
    """
    out_path에 여러 줄로 pretty-printed 된 JSON 객체들이 연속 저장된 경우까지 지원.
    (현재 코드처럼 indent=4로 write하면 이 형태가 됨)
    """
    done: Set[str] = set()
    p = Path(out_path)
    if not p.exists():
        return done

    async with aiofiles.open(out_path, "r", encoding="utf-8") as f:
        text = await f.read()

    dec = JSONDecoder()
    i, n = 0, len(text)

    while True:
        # whitespace skip
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            # 파일 끝에 부분 write로 깨진 JSON이 남아있을 수 있음 -> 무시하고 종료
            break

        qid = obj.get("question_id")
        if qid is not None and str(qid).strip():
            done.add(str(qid).strip())

        i = j  # 다음 객체로 이동

    return done


async def gen_code(
    prompt: str,
    llm,
    platform: str,
    model: str,
    temperature: float,
    top_p: float,
) -> Optional[str]:
    if len(prompt) > 80000:
        logging.warning(f"Prompt too long ({len(prompt)} chars), skipping this testcase.")
        return None, None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            if platform == "anthropic":
                resp = await llm.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=60000,
                        temperature=0,
                        stream=True,
                    )
                full_text = []

                async for event in resp:
                    if event.type == "content_block_delta":
                        if event.delta.text:
                            full_text.append(event.delta.text)

                content = "".join(full_text)
                return extract_block(content), content
            elif platform == "openai":
                resp = await asyncio.wait_for(
                    llm.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_completion_tokens=60000,
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
                content = resp.choices[0].message.content or ""
                return extract_block(content), content
            else:
                resp = await asyncio.wait_for(
                    llm.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=60000,
                        extra_body={"repetition_penalty": 1.05},
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
                content = resp.choices[0].message.content or ""
                return extract_block(content), content
        except asyncio.TimeoutError:
            logging.warning(f"gen_code timeout (attempt {attempt})")
        except Exception as e:
            logging.error(f"gen_code exception (attempt {attempt}): {e}")

    return None, None


def list_question_ids_from_problems(problems: List[dict]) -> Set[str]:
    """JSON problems에서 question_id만 뽑아 집합으로."""
    qids: Set[str] = set()
    for p in problems:
        qid = p.get("question_id")
        if qid is not None and str(qid).strip():
            qids.add(str(qid).strip())
    return qids


def load_livecodebench_index_for_qids(target_qids: Set[str]) -> Dict[str, dict]:
    """
    LiveCodeBench test split을 로드한 뒤,
    target_qids에 해당하는 row만 index로 만든다.
    """
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_latest",
    )

    index: Dict[str, dict] = {}
    for row in ds:
        qid = row.get("question_id") or row.get("id") or row.get("problem_id")
        if not qid:
            continue
        qid = str(qid)
        if qid in target_qids:
            index[qid] = row

    missing = len(target_qids - set(index.keys()))
    logging.info(
        f"Target qids: {len(target_qids)}, Matched rows: {len(index)}, Missing: {missing}"
    )
    return index


async def handle_problem_item(
    problem: dict,
    task: str,
    qid_to_row: Dict[str, dict],
    llm: AsyncOpenAI,
    sem: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    out_path: str,
    model: str,
    platform: str,
    temperature: float,
    top_p: float,
) -> bool:
    async with sem:
        try:
            qid = str(problem.get("question_id")).strip()
            if not qid:
                return False

            row = qid_to_row.get(qid)
            if not row:
                logging.warning(f"No matching LiveCodeBench row for question_id={qid}")
                return False

            eval_sample = build_evaluation_sample(row)
            testcase_list = eval_sample.get("testcases", [])
            if not testcase_list:
                logging.warning(f"No testcases for question_id={qid}")
                return False

            code_list = problem.get("code_list") or []
            if not code_list:
                return False
                
            code_list = code_list[:1]
            code_results = []

            for code_idx, code in enumerate(code_list):
                per_test_results = []

                for tc_idx, tc in enumerate(testcase_list):
                    prompt = get_template(task).format(code=code, testcase=tc)
                    llm_output, raw_output = await gen_code(
                        prompt=prompt,
                        llm=llm,
                        platform=platform,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    pass_fail = extract_pass_fail(llm_output) if llm_output else None

                    per_test_results.append(
                        {
                            "testcase_index": tc_idx,
                            "pass_fail": pass_fail,
                            "raw_response": raw_output,
                        }
                    )

                if any(r["pass_fail"] == "FAIL" for r in per_test_results):
                    overall = False
                elif all(r["pass_fail"] == "PASS" for r in per_test_results):
                    overall = True
                else:
                    overall = None

                code_results.append(
                    {
                        "code_index": code_idx,
                        "overall_pass": overall,
                        "pass_fail_list": [r["pass_fail"] for r in per_test_results],
                        "raw_output": [r["raw_response"] for r in per_test_results],
                    }
                )

            out = {
                "question_id": qid,
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


async def eval_codes_from_json(
    llm: AsyncOpenAI,
    problem_json_path: str,
    task: str,
    out_path: str,
    model: str,
    platform: str,
    temperature: float,
    top_p: float,
) -> int:
    with open(problem_json_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    if not isinstance(problems, list):
        raise ValueError("problem_json must be a list.")

    target_qids = list_question_ids_from_problems(problems)
    if not target_qids:
        logging.warning("No question_id found in problem_json.")
        return 0

    qid_to_row = load_livecodebench_index_for_qids(target_qids)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done_qids = await load_done_qids_from_outfile(out_path)
    logging.info(f"Already done in out_path: {len(done_qids)}")

    sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    file_lock = asyncio.Lock()

    problems_to_eval = [
        p for p in problems
        if (qid := str(p.get("question_id", "")).strip())
        and qid in qid_to_row
        and qid not in done_qids
    ]

    logging.info(f"To eval (after skip done): {len(problems_to_eval)}")

    coros = [
        handle_problem_item(
            problem=p,
            task=task,
            qid_to_row=qid_to_row,
            llm=llm,
            sem=sem,
            file_lock=file_lock,
            out_path=out_path,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        for p in problems_to_eval
    ]

    ok_cnt = 0
    pbar = tqdm(total=len(coros), desc="Evaluating", unit="problem")
    try:
        for fut in asyncio.as_completed(coros):
            if await fut:
                ok_cnt += 1
            pbar.update(1)
    finally:
        pbar.close()

    logging.info(f"Completed: {len(coros)}, Saved: {ok_cnt}")
    return ok_cnt
