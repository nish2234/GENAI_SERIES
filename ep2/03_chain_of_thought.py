"""Pattern 3: Chain-of-thought — force step-by-step reasoning for math/logic.

Demonstrates how adding "think step by step" to a prompt dramatically
improves accuracy on multi-step problems. Compares naive (direct answer)
vs chain-of-thought (show reasoning) approaches.
"""

from gemini_client import generate

problem = """A store sells notebooks for $4 each. If you buy 5 or more, 
you get a 20% discount on the entire purchase. Sarah buys 7 notebooks 
and pays with a $50 bill. How much change does she receive?"""

naive_prompt = f"""Solve this problem:
{problem}

Give only the final numerical answer."""

cot_prompt = f"""Solve this problem step by step. Show your reasoning clearly 
before giving the final answer.

Problem: {problem}

Let's think step by step:"""


if __name__ == "__main__":
    print("=== Chain-of-Thought Reasoning ===\n")
    print(f"Problem: {problem.strip()}\n")

    print("--- Without Chain-of-Thought ---")
    print(generate(naive_prompt, temperature=0.0))

    print("\n--- With Chain-of-Thought ---")
    print(generate(cot_prompt, temperature=0.0))


