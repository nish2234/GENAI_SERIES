"""
04 — Three Agent Team
======================
Extended pipeline with 3 agents: Researcher → Writer → Editor
The Editor reviews and improves the Writer's output before final delivery.

Pipeline: research → write → edit → final output
"""

import importlib
import time
from gemini_client import generate

specialist_agents = importlib.import_module("02_specialist_agents")
ResearcherAgent = specialist_agents.ResearcherAgent
WriterAgent = specialist_agents.WriterAgent


# ---------------------------------------------------------------------------
# New agent: Editor
# ---------------------------------------------------------------------------

class EditorAgent:
    """Specialist agent focused on reviewing and improving written content."""

    SYSTEM_PROMPT = """You are a world-class editor at a top publication. Your ONLY job is to improve existing writing.

Rules:
- Take the blog post provided and produce an IMPROVED version
- Fix any factual inaccuracies or vague claims
- Strengthen weak transitions and tighten loose paragraphs
- Improve the hook/introduction to be more compelling
- Ensure the conclusion has a clear takeaway
- Make the writing more vivid and engaging while keeping the author's voice
- Remove filler words and redundant sentences
- Ensure proper structure with clear section headers
- Output ONLY the improved blog post — do NOT add meta-commentary or change notes"""

    def __init__(self, name: str = "Editor"):
        self.name = name

    def run(self, blog_post: str) -> str:
        prompt = f"""Improve the following blog post. Make it sharper, more engaging, and more polished.
Return ONLY the improved version.

BLOG POST TO EDIT:
{blog_post}"""
        return generate(prompt, system=self.SYSTEM_PROMPT, max_tokens=2048, temperature=0.6)



# ---------------------------------------------------------------------------
# Three-Agent Orchestrator
# ---------------------------------------------------------------------------

class ThreeAgentOrchestrator:
    """Coordinates Researcher → Writer → Editor pipeline."""

    def __init__(self):
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.trace: list[dict] = []

    def _log(self, agent: str, action: str, content: str):
        self.trace.append({
            "agent": agent,
            "action": action,
            "content": content,
            "timestamp": time.strftime("%H:%M:%S"),
        })

    def _print_header(self, agent_name: str, task_description: str):
        print(f"\n{'─' * 60}")
        print(f"  [{agent_name}] {task_description}")
        print(f"{'─' * 60}")

    def run(self, topic: str) -> dict:
        """Full 3-agent pipeline: research → write → edit."""

        print("\n" + "=" * 60)
        print("  THREE-AGENT TEAM — Starting Pipeline")
        print(f"  Topic: {topic}")
        print("=" * 60)

        # Step 1: Research
        self._print_header("Researcher", f"Gathering facts about: {topic}")
        research_notes = self.researcher.run(topic)
        self._log("Researcher", "RESEARCH", research_notes)
        print(research_notes)

        # Step 2: Write (receives research)
        self._print_header("Writer", "Composing blog post from research notes...")
        first_draft = self.writer.run(research_notes)
        self._log("Writer", "WRITE", first_draft)
        print(first_draft)

        # Step 3: Edit (receives first draft)
        self._print_header("Editor", "Reviewing and improving the draft...")
        final_post = self.editor.run(first_draft)
        self._log("Editor", "EDIT", final_post)
        print(final_post)



        return {
            "first_draft": first_draft,
            "final_post": final_post,
            
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo 1: Three-agent pipeline
    print("#" * 60)
    print("  DEMO: Three-Agent Team")
    print("#" * 60)

    team = ThreeAgentOrchestrator()
    result = team.run("the rise of small language models and why they matter")

    # Show the difference
    print("\n" + "#" * 60)
    print("  BEFORE vs AFTER EDITING")
    print("#" * 60)

    print("\n--- FIRST DRAFT (Writer output) ---")
    print(result["first_draft"][:500] + "...\n")

    print("--- EDITED VERSION (Editor output) ---")
    print(result["final_post"][:500] + "...\n")


