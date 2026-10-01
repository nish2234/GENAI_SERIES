"""
Episode 13 — File 3: Trajectory Logger
=======================================
Run: python 03_trajectory_logger.py

What this file does:
  - Wraps an agent so every step is logged with timestamps
  - Records: Thought, Action (tool call), Observation (tool result), Answer
  - Exports the full trajectory to CSV
  - Analyses trajectories: step counts, tool frequency, loop detection
  - Shows how trajectory data helps debug agent behaviour

Trajectory = the complete path an agent takes from question to answer.
"""

import csv
import json
import time
from datetime import datetime
from gemini_client import generate, generate_structured

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
When asked a math question, show your calculation and give the exact numerical answer.
When asked to search for information, provide what you know.
If asked to do something harmful, unethical, or illegal, politely refuse.
Always be accurate and concise."""


# --- Tools ---

def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
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


# --- Step 1: Trajectory Logger class ---

class TrajectoryLogger:
    """Records every step an agent takes."""

    def __init__(self):
        self.trajectories: list[dict] = []
        self._current: list[dict] = []
        self._current_id: str = ""
        self._start_time: float = 0

    def start(self, test_id: str, question: str):
        """Begin logging a new trajectory."""
        self._current = []
        self._current_id = test_id
        self._start_time = time.time()
        self._log_step("INPUT", question)

    def log_thought(self, thought: str):
        self._log_step("THOUGHT", thought)

    def log_action(self, tool_name: str, tool_input: str):
        self._log_step("ACTION", f"{tool_name}({tool_input})")

    def log_observation(self, result: str):
        self._log_step("OBSERVATION", result)

    def log_answer(self, answer: str):
        self._log_step("ANSWER", answer)

    def finish(self) -> dict:
        """Finalise the current trajectory and return summary."""
        elapsed = time.time() - self._start_time
        trajectory = {
            "test_id": self._current_id,
            "steps": list(self._current),
            "num_steps": len(self._current),
            "elapsed_seconds": round(elapsed, 2),
            "tools_used": [
                s["content"].split("(")[0]
                for s in self._current if s["type"] == "ACTION"
            ],
        }
        self.trajectories.append(trajectory)
        return trajectory

    def _log_step(self, step_type: str, content: str):
        self._current.append({
            "type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": round((time.time() - self._start_time) * 1000),
        })

    def export_csv(self, filename: str = "trajectories.csv"):
        """Export all trajectories to CSV — one row per step."""
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_id", "step_number", "step_type", "content",
                "timestamp", "elapsed_ms",
            ])
            for traj in self.trajectories:
                for i, step in enumerate(traj["steps"], 1):
                    writer.writerow([
                        traj["test_id"], i, step["type"],
                        step["content"][:500],
                        step["timestamp"], step["elapsed_ms"],
                    ])
        print(f"\nTrajectories exported to {filename}")
        return filename


# --- Step 2: Agent with trajectory logging ---

def run_agent_with_logging(question: str, logger: TrajectoryLogger, test_id: str) -> dict:
    """Run agent and log every step to the trajectory."""
    logger.start(test_id, question)

    decision = generate_structured(
        f"Given this question, decide which tool (if any) to use.\n"
        f"Available tools: calculator (for math), search (for current info/lookups).\n"
        f"Question: {question}",
        response_schema=TOOL_DECISION_SCHEMA,
        temperature=0.0, max_tokens=100, system=SYSTEM_PROMPT,
    )

    tool_name = decision.get("tool", "none")
    tool_arg = decision.get("tool_input", "")
    logger.log_thought(f"Tool decision: {tool_name} | {tool_arg}")

    tool_context = ""
    tools_used = []

    if tool_name != "none" and tool_name in TOOLS:
        logger.log_action(tool_name, tool_arg)
        tool_result = TOOLS[tool_name](tool_arg)
        logger.log_observation(tool_result)
        tools_used.append(tool_name)
        tool_context = f"\n\nTool '{tool_name}' returned: {tool_result}"

    answer = generate(
        f"Question: {question}{tool_context}\n\nProvide a clear, accurate answer.",
        temperature=0.0, max_tokens=512, system=SYSTEM_PROMPT,
    )
    logger.log_answer(answer)

    trajectory = logger.finish()
    return {"answer": answer, "tools_used": tools_used, "trajectory": trajectory}


# --- Step 3: Trajectory analysis ---

def analyse_trajectories(logger: TrajectoryLogger):
    """Print analysis of all recorded trajectories."""
    print("\n" + "=" * 70)
    print("TRAJECTORY ANALYSIS")
    print("=" * 70)

    total = len(logger.trajectories)
    all_steps = [t["num_steps"] for t in logger.trajectories]
    all_times = [t["elapsed_seconds"] for t in logger.trajectories]

    print(f"\n  Total trajectories: {total}")
    print(f"  Avg steps per trajectory: {sum(all_steps)/total:.1f}")
    print(f"  Min/Max steps: {min(all_steps)} / {max(all_steps)}")
    print(f"  Avg time per trajectory: {sum(all_times)/total:.2f}s")
    print(f"  Total eval time: {sum(all_times):.2f}s")

    tool_counts: dict[str, int] = {}
    for t in logger.trajectories:
        for tool in t["tools_used"]:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"\n  Tool usage:")
    if tool_counts:
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"    {tool}: {count} calls")
    else:
        print("    No tools were used.")

    no_tool = sum(1 for t in logger.trajectories if not t["tools_used"])
    print(f"\n  Questions answered without tools: {no_tool}/{total}")

    long_runs = [t for t in logger.trajectories if t["num_steps"] > 5]
    if long_runs:
        print(f"\n  ⚠ Long trajectories (>5 steps): {len(long_runs)}")
        for t in long_runs:
            print(f"    {t['test_id']}: {t['num_steps']} steps, {t['elapsed_seconds']}s")


# --- Step 4: Run demo ---

def main():
    with open("test_cases.json") as f:
        test_cases = json.load(f)

    sample = test_cases[:6]

    print("=" * 70)
    print("TRAJECTORY LOGGING DEMO")
    print(f"Running {len(sample)} test cases with full trajectory recording...")
    print("=" * 70)

    logger = TrajectoryLogger()

    for tc in sample:
        print(f"\n--- {tc['id']} ---")
        print(f"  Q: {tc['question']}")

        try:
            result = run_agent_with_logging(tc["question"], logger, tc["id"])
            print(f"  A: {result['answer'][:100]}...")
            print(f"  Steps: {result['trajectory']['num_steps']}  |  Time: {result['trajectory']['elapsed_seconds']}s")
            print(f"  Tools: {result['tools_used'] or 'none'}")
        except Exception as e:
            print(f"  ERROR: {e}")

    analyse_trajectories(logger)

    csv_file = logger.export_csv("trajectories.csv")

    print(f"\n--- Sample CSV output ({csv_file}) ---")
    with open(csv_file) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 10:
                print(f"  {','.join(row[:4])}...")
            else:
                print(f"  ... ({sum(1 for _ in f) + i} total rows)")
                break


if __name__ == "__main__":
    main()
