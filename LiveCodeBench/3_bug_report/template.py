CODECONTEST_PROMPT_TMPL = '''
You are a skilled Python programmer and code reviewer.
Carefully evaluate whether the given code can pass the given single test case, and present your reasoning and final judgment.

[You are given]
- Python code
- ONE test case (input, output, and optional function name)

[Your task]
- Analyze the functionality of the code and the provided test case,
  and determine whether the code will pass this specific test case.

[Requirements]
- Wrap the entire response in a ```plaintext ... ``` code block.
- In [Explanation], provide only the essential reasoning.
- In [Result], only generate either PASS or FAIL (no other expressions allowed).
- Do not modify the code or propose any new solution; perform only “review and judgment.”

Output format 
EXAMPLE:
[Explanation]
...
[Result]
```plaintext
[PASS/FAIL]
```
EXAMPLE_END:

Code:
```python
{code}
```

Testcase:
```test
{testcase}
```
'''