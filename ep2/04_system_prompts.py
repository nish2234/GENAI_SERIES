"""Pattern 4: System prompts — same question answered with different personas.

Shows how system instructions control the model's identity, tone, and
output style. The same user question produces completely different
responses depending on the persona set in the system prompt.
"""

from gemini_client import generate

question = "Explain what an API is."

personas = {
    "Kindergarten Teacher": (
        "You are a kindergarten teacher. Explain everything using simple words, "
        "fun analogies, and short sentences. A 5-year-old must understand. "
        "Keep your response under 80 words."
    ),
    "Pirate Captain": (
        "You are a pirate captain. Answer all questions in pirate speak. "
        "Use nautical metaphors. Be dramatic and entertaining. "
        "Keep it under 80 words."
    ),
    "Senior Software Engineer": (
        "You are a senior software engineer with 15 years of experience. "
        "Be precise, technical, and concise. Include a practical example. "
        "No fluff. Max 80 words."
    ),
}


if __name__ == "__main__":
    print("=== System Prompts: Same Question, Different Personas ===")
    print(f'\nQuestion: "{question}"\n')

    for name, system_instruction in personas.items():
        print(f"--- {name} ---")
        result = generate(question, system=system_instruction, temperature=0.7)
        print(result.strip())
        print()
