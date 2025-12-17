import argparse
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from generation import eval_codes_from_json

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problems", type=str, required=True)
    parser.add_argument("--out", type=str, default="result/results.json")
    parser.add_argument("--task", type=str, default="pred", choices=['pred', 'bug_local', 'bug_report'])

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--platform", type=str, default="vllm")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)

    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default="http://localhost:8001/v1")

    args = parser.parse_args()

    #llm = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    llm = AsyncAnthropic(api_key=args.api_key)

    saved = asyncio.run(
        eval_codes_from_json(
            llm=llm,
            problem_json_path=args.problems,
            task=args.task,
            out_path=args.out,
            model=args.model,
            platform=args.platform,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )

    print(f"Saved problem results: {saved}")


if __name__ == "__main__":
    main()
