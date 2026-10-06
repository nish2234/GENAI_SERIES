"""
Episode 14 — File 2: Image Q&A
===============================
Run: python 02_image_qa.py

What this file does:
  - Loads an image (URL or local path)
  - Enters an interactive loop where the user asks questions about the image
  - The agent sees the image + question each time and answers
  - Great for exploring what Gemini can "see" in an image

Try questions like:
  - "What colors are in this image?"
  - "How many people are there?"
  - "What text is visible?"
  - "What is the mood of this image?"
"""

from gemini_client import generate_with_image

# Default demo image — a busy street scene with lots to ask about
DEFAULT_IMAGE = "sample_images/test-image.jpeg"

print("=" * 60)
print("IMAGE Q&A — Ask Questions About Any Image")
print("=" * 60)

# Let the user provide their own image or use the default
user_image = input("\nImage path or URL (press Enter for demo image): ").strip()
image_path = user_image if user_image else DEFAULT_IMAGE

print(f"\nUsing image: {image_path}")
print("\nLet me first describe what I see...\n")

overview = generate_with_image(
    prompt="Briefly describe what you see in this image in 2-3 sentences.",
    image_path=image_path,
    temperature=0.5,
)
print(f"Overview: {overview}\n")

print("-" * 60)
print("Now ask me anything about this image!")
print("Type 'quit' to exit, 'new' to load a different image.")
print("-" * 60)

while True:
    question = input("\nYour question: ").strip()

    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if question.lower() == "new":
        new_image = input("New image path or URL: ").strip()
        if new_image:
            image_path = new_image
            print(f"\nSwitched to: {image_path}")
            overview = generate_with_image(
                prompt="Briefly describe what you see in this image in 2-3 sentences.",
                image_path=image_path,
                temperature=0.5,
            )
            print(f"Overview: {overview}")
        continue

    answer = generate_with_image(
        prompt=question,
        image_path=image_path,
        temperature=0.5,
    )
    print(f"\nAnswer: {answer}")
