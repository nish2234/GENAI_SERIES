"""
Episode 14 — File 1: Image Description
=======================================
Run: python 01_image_description.py

What this file does:
  - Loads an image (from a URL or local file path)
  - Sends it to Gemini with "Describe this image in detail"
  - Prints the AI's description
  - Shows how easy multimodal input is with the Gemini SDK

This is the simplest possible vision task — one image, one prompt, one response.
"""

from gemini_client import generate_with_image


# --- Try with a local file (if it exists) ---
import os

local_image = "sample_images/test-image.jpeg"
if os.path.exists(local_image):
    print(f"\n\nAlso found local image: {local_image}")
    print("Describing local image...\n")

    local_description = generate_with_image(
        prompt="Describe this image in detail.",
        image_path=local_image,
        temperature=0.5,
    )
    print("DESCRIPTION (local):")
    print("-" * 40)
    print(local_description)
    print("-" * 40)
else:
    print(f"\nTip: Place an image at '{local_image}' to test with your own photos!")

print("\nDone!")
