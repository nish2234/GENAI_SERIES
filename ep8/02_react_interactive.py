import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"

# ─── Structured output schema ────────────────────────────────────────────────

REACT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your reasoning about what to do next",
        },
        "action": {
            "type": "string",
            "enum": ["calculate", "search", "lookup", "none"],
            "description": "Tool to call, or 'none' if giving a final answer",
        },
        "action_input": {
            "type": "string",
            "description": "Input to pass to the tool (empty string if action is 'none')",
        },
        "answer": {
            "type": "string",
            "description": "Final answer (empty string if still working, filled when done)",
        },
    },
    "required": ["thought", "action", "action_input", "answer"],
}

# ─── Tools ──────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    """Simulated web search — returns hardcoded results for demo."""
    results = {
        "population of france": "France has a population of approximately 68.4 million people (2024 estimate).",
        "tallest building in the world": "The Burj Khalifa in Dubai is the tallest building at 828 meters (2,717 ft).",
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "distance earth to moon": "The average distance from Earth to the Moon is about 384,400 kilometers.",
        "python programming language": "Python is a high-level programming language created by Guido van Rossum, first released in 1991.",
        "who invented the telephone": "Alexander Graham Bell is credited with inventing the first practical telephone in 1876.",
        "gdp of japan": "Japan's GDP is approximately $4.2 trillion USD (2024), making it the 4th largest economy.",
        "boiling point of water": "Water boils at 100 degrees C (212 degrees F) at standard atmospheric pressure (1 atm).",
        "capital of australia": "The capital of Australia is Canberra, not Sydney as commonly thought.",
        "largest ocean": "The Pacific Ocean is the largest and deepest ocean, covering about 165.25 million km squared.",
        "mars distance from sun": "Mars is approximately 227.9 million km from the Sun on average.",
        "world population": "The world population is approximately 8.1 billion people (2024 estimate).",
    }
    query_lower = query.lower().strip()
    for key, value in results.items():
        if key in query_lower or query_lower in key:
            return value
    return f"Search results for '{query}': No specific results found. Try rephrasing."


def lookup(topic: str) -> str:
    """Simulated Wikipedia lookup — returns hardcoded info for demo."""
    articles = {
        "react pattern": (
            "ReAct (Reason + Act) is a prompting paradigm introduced by Yao et al. (2022) "
            "that interleaves reasoning traces and task-specific actions."
        ),
        "eiffel tower": (
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. Constructed "
            "1887-1889 for the World's Fair. Height: 330 meters. Named after Gustave Eiffel. "
            "Receives about 7 million visitors annually."
        ),
        "photosynthesis": (
            "Photosynthesis converts light energy into chemical energy stored in glucose. "
            "Equation: 6CO2 + 6H2O + light -> C6H12O6 + 6O2."
        ),
        "alan turing": (
            "Alan Turing (1912-1954) was a British mathematician and computer scientist. "
            "Father of theoretical CS and AI. Broke the Enigma code in WWII. "
            "Proposed the Turing Test in 1950."
        ),
        "gravity": (
            "Gravity is a fundamental force. On Earth, g is approximately 9.8 m/s squared. "
            "Described by Newton (1687) and refined by Einstein's General Relativity (1915)."
        ),
        "python": (
            "Python is a high-level, interpreted programming language created by Guido van "
            "Rossum. First released in 1991. Known for readability and versatility. "
            "Used in web dev, data science, AI/ML, and scripting."
        ),
        "artificial intelligence": (
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
            "Subfields include machine learning, NLP, computer vision, and robotics. "
            "The field was founded at the Dartmouth Conference in 1956."
        ),
    }
    topic_lower = topic.lower().strip()
    for key, value in articles.items():
        if key in topic_lower or topic_lower in key:
            return value
    return f"Wikipedia article for '{topic}': No article found. Try a different topic."


# ─── Tool dispatch & system prompt ───────────────────────────────────────────

TOOLS = {
    "calculate": calculate,
    "search": search,
    "lookup": lookup,
}

TOOL_DESCRIPTIONS = """You have access to these tools:

1. calculate(expression) — Evaluate a math expression. Example input: "15 * 340 / 100"
2. search(query) — Search the web for current information. Example input: "population of France"
available keywords: population of france, tallest building, speed of light, distance earth to moon, python programming language, who invented the telephone, gdp of japan, boiling point of water, capital of australia, largest ocean, mars distance from sun, world population
3. lookup(topic) — Look up a Wikipedia article on a topic. Example input: "Alan Turing"
available keywords: react pattern, eiffel tower, photosynthesis, alan turing, gravity, python, artificial intelligence

When you want to use a tool, set action to the tool name and action_input to the argument.
When you have the final answer, set action to "none" and fill in the answer field."""

# [UPDATED] Added explicit rules for failure handling to prevent prompt drift
REACT_SYSTEM_PROMPT = f"""\
You are a ReAct agent. You solve problems step-by-step using Thought and Action.

{TOOL_DESCRIPTIONS}

RULES:
- Always provide a thought explaining your reasoning.
- Use exactly ONE tool per step (set action to the tool name).
- When you have enough information to answer, set action to "none" and fill the answer field.
- Never make up information — use tools to find facts.
- Remember thoughts or answers should not exceed 200 words. Be concise and clear.
- Keep your final answer clear, concise, and directly addressing the original question.
- If a tool returns "No specific results found" or "No article found", DO NOT repeat the exact same search. Try a completely different keyword or set action to "none" and state you cannot find the answer.
- If you set action to "none", you MUST provide a final summary in the answer field."""


def react_agent(question: str, max_steps: int = 10) -> str:
    """Run the ReAct agent loop until an answer is found or max steps reached."""
    print(f"\n{'='*70}")
    print(f"  QUESTION: {question}")
    print(f"{'='*70}")

    conversation = f"Question: {question}"
    
    # [ADDED] Memory to track what the agent has already tried
    past_actions = set()

    for step in range(1, max_steps + 1):
        print(f"\n{'─'*70}")
        print(f"  STEP {step}")
        print(f"{'─'*70}")

        response = client.models.generate_content(
            model=MODEL,
            contents=conversation,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=REACT_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=REACT_SCHEMA,
            ),
        )
        print(response.text)
        parsed = json.loads(response.text)

        thought = parsed.get("thought", "")
        action = parsed.get("action", "none")
        action_input = parsed.get("action_input", "")
        answer = parsed.get("answer", "")

        if thought:
            print(f"  Thought: {thought}")

        # [ADDED] Repetition Circuit Breaker
        current_move = f"{action}:{action_input}"
        if current_move in past_actions and action != "none":
            print(f"  [System] Agent is repeating itself. Forcing stop to prevent infinite loop.")
            final_ans = "I got stuck repeating the same action and could not find the answer."
            print(f"  Answer:  {final_ans}")
            return final_ans
        past_actions.add(current_move)

        # [UPDATED] Robust exit condition that doesn't rely on the 'answer' string being populated
        if action == "none":
            final_ans = answer if answer else "I could not find the answer based on the available tools."
            print(f"  Answer:  {final_ans}")
            print(f"\n{'='*70}")
            print(f"  Agent finished in {step} step(s)")
            print(f"{'='*70}")
            return final_ans

        if action in TOOLS:
            print(f"  Action:  {action}({action_input})")
            observation = TOOLS[action](action_input)
            print(f"  Observe: {observation}")
            conversation += (
                f"\n\nThought: {thought}\n"
                f"Action: {action}({action_input})\n"
                f"Observation: {observation}"
            )

        elif action != "none":
            error_msg = f"Unknown tool: {action}. Available: {', '.join(TOOLS.keys())}"
            print(f"  Error:   {error_msg}")
            conversation += f"\n\nThought: {thought}\nObservation: {error_msg}"

        else:
            print(f"  Warning: No action or answer. Nudging agent...")
            conversation += (
                f"\n\nThought: {thought}\n"
                "Observation: You must either use a tool or provide a final answer."
            )

    print(f"\nMax steps ({max_steps}) reached.")
    return "Unable to determine the answer within the allowed steps."


# ─── Interactive loop ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  Interactive ReAct Agent (Structured Output)")
    print("  Pure Python, No Frameworks, No Regex")
    print("="*70)
    print("\n  Tools available: calculate, search, lookup")
    print("  Type 'quit' or 'exit' to stop.\n")
    print("  Try asking:")
    print("    - What is 25% of Japan's GDP?")
    print("    - Who is Alan Turing and what did he propose in 1950?")
    print("    - How many seconds does it take light to reach the Moon?")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        react_agent(question)
        print()


if __name__ == "__main__":
    main()