"""
02 — Specialist Agents
=======================
Two specialist agents, each with a focused system prompt:
  - ResearcherAgent: gathers facts, key points, thorough bullet-point notes
  - WriterAgent: takes research notes, produces a polished blog post

Each agent is a simple Python class wrapping a system prompt + generate().
"""

from gemini_client import generate


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

class ResearcherAgent:
    """Specialist agent focused purely on deep research."""

    SYSTEM_PROMPT = """You are a world-class research analyst. Your ONLY job is research.

Rules:
- Gather comprehensive facts, statistics, and key concepts about the given topic
- Identify recent breakthroughs, trends, and important developments
- Organize findings as clear, detailed bullet points
- Include specific names, dates, and numbers when possible
- Cover multiple angles: what it is, why it matters, current state, future outlook
- Be thorough — the writer who receives your research depends on its quality
- Output ONLY research notes in bullet-point format — do NOT write prose or blog posts"""

    def __init__(self, name: str = "Researcher"):
        self.name = name

    def run(self, topic: str) -> str:
        prompt = f"Research the following topic thoroughly and return detailed bullet-point notes:\n\nTopic: {topic}"
        return generate(prompt, system=self.SYSTEM_PROMPT, max_tokens=1500, temperature=0.4)



class WriterAgent:
    """Specialist agent focused purely on writing polished content."""

    SYSTEM_PROMPT = """You are a world-class blog writer and content creator. Your ONLY job is writing.

Rules:
- Take the research notes provided and transform them into an engaging, well-structured blog post
- Use clear, conversational language that any reader can understand
- Structure with a compelling introduction, organized sections with headers, and a strong conclusion
- Make complex topics accessible without dumbing them down
- Add transitions between sections for smooth reading flow
- Include a hook in the opening that grabs attention
- Output ONLY the finished blog post — do NOT add research or meta-commentary"""

    def __init__(self, name: str = "Writer"):
        self.name = name

    def run(self, research_notes: str) -> str:
        prompt = f"""Using the research notes below, write an engaging, well-structured blog post.

RESEARCH NOTES:
{research_notes}

Write the blog post now."""
        return generate(prompt, system=self.SYSTEM_PROMPT, max_tokens=2048, temperature=0.7)



# ---------------------------------------------------------------------------
# Demo: run each agent independently
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    topic = "quantum computing"

    # --- Demo the Researcher ---
    print("=" * 60)
    print(f"  RESEARCHER AGENT — Topic: '{topic}'")
    print("=" * 60)

    researcher = ResearcherAgent()
    research_notes = researcher.run(topic)
    print(research_notes)

    print("\n" + "=" * 60)
    print(f"  WRITER AGENT — Using research notes from above")
    print("=" * 60)

    writer = WriterAgent()
    blog_post = writer.run(research_notes)
    print(blog_post)


