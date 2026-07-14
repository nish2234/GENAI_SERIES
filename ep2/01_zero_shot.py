"""Pattern 1: Zero-shot classification — classify text into categories with no examples.

Demonstrates that a well-constrained prompt can classify text accurately
without providing any examples. The key is specifying exact categories
and constraining the output format.
"""

from gemini_client import generate

categories = ["Technology", "Sports", "Politics", "Entertainment", "Science"]

texts = [
    "Apple released the new M4 MacBook Pro with improved neural engine.",
    "Manchester United signed a new striker for 80 million euros.",
    "The Senate passed the new climate regulation bill today.",
]

prompt_template = """Classify the following text into exactly ONE of these categories:
{categories}

Text: "{text}"

Reply with ONLY the category name. Nothing else."""


if __name__ == "__main__":
    print("=== Zero-Shot Classification ===\n")

    for text in texts:
        prompt = prompt_template.format(
            categories=", ".join(categories),
            text=text,
        )
        result = generate(prompt, temperature=0.0)
        print(f"  [{result.strip():15s}]  {text[:60]}")
