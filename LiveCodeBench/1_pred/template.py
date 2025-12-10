CODECONTEST_PROMPT_TMPL = '''
You are a skilled Python programmer and code reviewer.
Carefully evaluate the given code passes all provided public and hidden test cases.

[You are given]
- Python code
- Test cases

[Your task]
- Determine whether the given Python code produces the correct result for the provided test input.

Requirements:	
- Wrap the entire response in a single ```plaintext ... ``` code block.
- In [Result], only generate either PASS or FAIL (no other expressions allowed).

Output format:
EXAMPLE:
[Results]
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