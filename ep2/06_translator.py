"""Pattern 6: Multi-language translator with tone options (formal vs casual).

Shows how to control translation register/tone by specifying language-specific
politeness conventions. Demonstrates that prompt engineering applies to
any language task, not just English.
"""

from gemini_client import generate

text = "Hey, I won't be able to make it to the meeting tomorrow. Can we reschedule to Thursday?"

languages = ["Spanish", "French", "Japanese"]
tones = ["formal", "casual"]

prompt_template = """Translate the following text to {language}.
Tone: {tone}

Rules:
- If formal: use polite/respectful forms (usted, vous, keigo, etc.)
- If casual: use informal/friendly forms (tú, tu, casual Japanese, etc.)
- Provide ONLY the translation. No explanations or alternatives.

Text: "{text}"

Translation:"""


if __name__ == "__main__":
    print("=== Multi-Language Translator with Tone ===\n")
    print(f'Original: "{text}"\n')

    for language in languages:
        print(f"--- {language} ---")
        for tone in tones:
            prompt = prompt_template.format(
                language=language,
                tone=tone,
                text=text,
            )
            result = generate(prompt, temperature=0.3)
            print(f"  [{tone:6s}]  {result.strip()}")
        print()
