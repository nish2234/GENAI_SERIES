"""Pattern 2: Few-shot labeling — provide 3-5 examples then classify new text.

Shows how adding examples to a prompt dramatically improves accuracy
for ambiguous or nuanced classification tasks. The model learns your
specific labeling style from the examples.
"""

from gemini_client import generate

prompt_template = """Classify customer feedback as Positive, Negative, or Neutral.

Examples:
Feedback: "This product is amazing, best purchase ever!"
Sentiment: Positive

Feedback: "Terrible quality. Broke after one day. Want a refund."
Sentiment: Negative

Feedback: "It arrived on time. Does what it says."
Sentiment: Neutral

Feedback: "Works okay but the color is slightly different from the picture."
Sentiment: Neutral

Feedback: "Absolutely love it! Already ordered two more for gifts."
Sentiment: Positive

Now classify this new feedback:
Feedback: "{text}"
Sentiment:"""

test_cases = [
    "Not bad, but shipping took forever and the box was damaged.",
    "This changed my life. Five stars is not enough!",
    "It's fine. Nothing special.",
    "DO NOT BUY. Scam product. Customer service ghosted me.",
    "Pretty good for the price, minor scratches on arrival though.",
]


if __name__ == "__main__":
    print("=== Few-Shot Sentiment Classification ===\n")

    for text in test_cases:
        prompt = prompt_template.format(text=text)
        result = generate(prompt, temperature=0.0)
        label = result.strip().split("\n")[0]
        print(f"  [{label:8s}]  {text[:60]}")
