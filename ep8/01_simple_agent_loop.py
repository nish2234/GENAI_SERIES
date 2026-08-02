"""
Episode 08 — File 1: A Simple Agent Loop
=========================================
Run: python 01_simple_agent_loop.py

What this file does:
  - Builds the simplest possible agent loop in ~30 lines
  - A while loop that sends conversation history to the LLM
  - The model decides when it has a final answer
  - No tools, no ReAct — just the core loop concept

This is the foundation: an agent is just a loop.
"""

from gemini_client import generate

SYSTEM_PROMPT = """\
You are a helpful assistant. When the user asks a question, think step by step.

If you need more information to answer well, say "CONTINUE" at the end of your
response and keep reasoning. When you have a complete answer, say "FINAL ANSWER:"
followed by your answer.

Always end with either CONTINUE or FINAL ANSWER: — never leave it ambiguous."""

def simple_agent(question: str, max_steps: int = 5) -> str:
    """A minimal agent loop: reason until FINAL ANSWER is reached."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    conversation = f"User question: {question}\n\nThink step by step."
    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")
        response = generate(conversation, temperature=0.3, system=SYSTEM_PROMPT)
        print(response)

        if "FINAL ANSWER:" in response:
            answer = response.split("FINAL ANSWER:")[-1].strip()
            print(f"\n{'='*60}")
            print(f"Agent finished in {step} step(s)")
            print(f"Final Answer: {answer}")
            print(f"{'='*60}")
            return answer

        conversation += f"\n\nAssistant (step {step}):\n{response}\n\nContinue reasoning."

    print(f"\nAgent hit max steps ({max_steps}) without a final answer.")
    return response


if __name__ == "__main__":
    simple_agent("What is 15% of 340, and is the result greater than 50?")
    print("\n")
    simple_agent("If I have 3 boxes with 12 apples each, and I give away 7 apples, how many do I have left?")
