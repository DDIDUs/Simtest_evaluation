import asyncio
import base64
import json
import logging
import pickle
import re
import zlib
from typing import Dict, List, Any, Optional

def extract_block(s: str) -> str:
    """첫 번째 ```plaintext 블록 안의 내용만 추출, 없으면 전체."""
    if not s:
        return ""
    m = re.search(r"```plaintext[a-zA-Z0-9_-]*\n(.*?)```", s, re.DOTALL)
    return (m.group(1) if m else s).strip()


def extract_pass_fail(s: str) -> Optional[str]:
    """LLM 응답에서 PASS 또는 FAIL만 추출."""
    if not s:
        return None
    candidates = re.findall(r"\b(PASS|FAIL)\b", s)
    if not candidates:
        return None
    return candidates[-1]

# ========= 테스트케이스 준비 유틸 =========
def decode_private_test_cases(s: str):
    """private_test_cases 문자열을 json 리스트로 디코딩."""
    if not s:
        return []
    return json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(s.encode("utf-8"))
            )
        )
    )  # [{"input": ..., "output": ...}, ...]


def build_evaluation_sample(row: dict) -> dict:
    # public_test_cases: JSON 문자열 or 이미 리스트일 수 있음
    public_raw = row["public_test_cases"]
    if isinstance(public_raw, str):
        public_cases = json.loads(public_raw)
    else:
        public_cases = public_raw

    # private_test_cases: base64 + zlib + pickle + json
    private_cases = decode_private_test_cases(row["private_test_cases"])

    all_cases = public_cases + private_cases

    testcase_strings: List[str] = []
    for case in all_cases:
        # case: {"input": ..., "output": ...}
        tc = {
            "input": case.get("input"),
            "output": case.get("output"),
        }
        testcase_strings.append(json.dumps(tc, ensure_ascii=False))

    eval_sample = {
        "testcases": testcase_strings,
        "question_id": row.get("question_id", None),
    }
    return eval_sample