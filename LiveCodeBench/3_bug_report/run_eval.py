import argparse
import asyncio
from openai import AsyncOpenAI

from generation import eval_codes_from_json


DEFAULT_OUT_PATH = "./test.jsonl"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problems", type=str, required=True)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)

    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default="http://localhost:8003/v1")

    args = parser.parse_args()

    llm = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    saved = asyncio.run(
        eval_codes_from_json(
            llm=llm,
            problem_json_path=args.problems,
            out_path=args.out,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )

    print(f"Saved problem results: {saved}")


if __name__ == "__main__":
    main()
