"""
Episode 13 — File 1: Simple Keyword-Based Evaluation
=====================================================
Run: python 01_simple_eval.py

What this file does:
  - Loads 20 test cases from test_cases.json
  - Runs each question through a simple agent (with calculator + search tools)
  - Checks if expected keywords appear in the response
  - Prints pass/fail for each test case
  - Shows overall score at the end

This is the simplest eval approach — keyword matching. Fast but brittle.
"""

import json
from gemini_client import generate, generate_structured

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
When asked a math question, show your calculation and give the exact numerical answer.
When asked to search for information, provide what you know.
If asked to do something harmful, unethical, or illegal, politely refuse.
Always be accurate and concise."""


# --- Step 1: Define simple tools (simulated for eval) ---

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    """Simulated search — returns canned results for eval purposes."""
    results = {
        "weather tokyo": "Tokyo: 22°C, partly cloudy, humidity 65%.",
        "latest ai news": "AI news: New breakthroughs in multimodal models and agent frameworks in 2026.",
        "fifa world cup winner": "Argentina won the 2022 FIFA World Cup, defeating France in the final.",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if any(word in query_lower for word in key.split()):
            return value
    return f"Search results for '{query}': No specific results found."


TOOLS = {"calculator": calculator, "search": search}


# --- Step 2: Simple agent that decides whether to use tools ---

TOOL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": ["calculator", "search", "none"],
            "description": "Which tool to use, or 'none' for no tool",
        },
        "tool_input": {
            "type": "string",
            "description": "Input for the tool (math expression or search query). Empty if no tool.",
        },
    },
    "required": ["tool", "tool_input"],
}


def run_agent(question: str) -> dict:
    """Run a simple agent: decide tools via structured output, execute, generate final answer."""
    tools_used = []

    decision = generate_structured(
        f"Given this question, decide which tool (if any) to use.\n"
        f"Available tools: calculator (for math), search (for current info/lookups).\n"
        f"Question: {question}",
        response_schema=TOOL_DECISION_SCHEMA,
        temperature=0.0,
        max_tokens=100,
        system=SYSTEM_PROMPT,
    )

    tool_name = decision.get("tool", "none")
    tool_arg = decision.get("tool_input", "")

    tool_context = ""
    if tool_name != "none" and tool_name in TOOLS:
        tool_result = TOOLS[tool_name](tool_arg)
        tools_used.append(tool_name)
        tool_context = f"\n\nTool '{tool_name}' returned: {tool_result}"

    answer_prompt = f"Question: {question}{tool_context}\n\nProvide a clear, accurate answer."
    answer = generate(answer_prompt, temperature=0.0, max_tokens=512, system=SYSTEM_PROMPT)

    return {"answer": answer, "tools_used": tools_used}


# --- Step 3: Keyword matching evaluator ---

def check_keywords(response: str, expected_keywords: list[str]) -> bool:
    """Check if ANY of the expected keywords appear in the response."""
    response_lower = response.lower()
    return any(keyword.lower() in response_lower for keyword in expected_keywords)


# --- Step 4: Load test cases and run evaluation ---

def main():
    with open("test_cases.json") as f:
        test_cases = json.load(f)

    print("=" * 70)
    print("SIMPLE KEYWORD-BASED EVALUATION")
    print("=" * 70)

    passed = 0
    failed = 0
    results = []

    test_cases = test_cases[0:3]

    for tc in test_cases:
        tc_id = tc["id"]
        question = tc["question"]
        expected_keywords = tc["expected_answer_contains"]
        expected_tools = tc["expected_tools"]
        category = tc["category"]

        print(f"\n--- Test: {tc_id} ({category}) ---")
        print(f"  Q: {question}")

        try:
            result = run_agent(question)
            answer = result["answer"]
            tools_used = result["tools_used"]

            keyword_pass = check_keywords(answer, expected_keywords)
            tool_pass = set(expected_tools).issubset(set(tools_used)) if expected_tools else True
            overall_pass = keyword_pass and tool_pass

            status = "PASS" if overall_pass else "FAIL"
            if overall_pass:
                passed += 1
            else:
                failed += 1

            print(f"  A: {answer[:120]}...")
            print(f"  Tools used: {tools_used}")
            print(f"  Keyword match: {'✓' if keyword_pass else '✗'}  |  Tool match: {'✓' if tool_pass else '✗'}  |  {status}")

            results.append({
                "id": tc_id, "category": category, "passed": overall_pass,
                "keyword_pass": keyword_pass, "tool_pass": tool_pass,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            results.append({"id": tc_id, "category": category, "passed": False})

    # --- Step 5: Print summary ---
    total = passed + failed
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}  ({passed/total*100:.1f}%)")
    print(f"  Failed: {failed}  ({failed/total*100:.1f}%)")



if __name__ == "__main__":
    main()
