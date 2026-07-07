"""
Reusable Gemini Client — Used in Every Episode of This Series
==============================================================

This module provides two clean utility functions:
  - generate()      → Send a prompt, get a text response
  - count_tokens()  → Count tokens without generating a response

Usage (in any future episode file):
  from gemini_client import generate, count_tokens

  answer = generate("What is a transformer?")
  tokens = count_tokens("How long is this sentence?")
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Singleton client — created once, reused for all calls
_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Returns the shared Gemini client, creating it on first call."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found.\n"
                "Make sure your .env file exists and contains: GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str | None = None,
) -> str:
    """
    Send a prompt to Gemini and return the response text.

    Args:
        prompt:      The user message / question.
        model:       Gemini model to use. Default: gemini-2.5-flash-lite
        temperature: 0.0 = deterministic, 1.5 = creative. Default: 0.7
        max_tokens:  Maximum tokens in the response. Default: 1024
        system:      Optional system instruction (sets the model's persona/behaviour).

    Returns:
        The model's response as a plain string.

    Example:
        answer = generate("Explain RAG in one paragraph.")
        answer = generate("Write a poem.", temperature=1.2)
        answer = generate("Summarise this.", system="You are a concise technical writer.")
    """
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


def generate_with_metadata(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str | None = None,
) -> dict:
    """
    Same as generate() but also returns token usage and finish reason.

    Returns a dict with keys:
        text, input_tokens, output_tokens, total_tokens, finish_reason
    """
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return {
        "text": response.text,
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count,
        "finish_reason": str(response.candidates[0].finish_reason),
    }


def count_tokens(prompt: str, model: str = "gemini-2.5-flash-lite") -> int:
    """
    Count how many tokens a prompt uses WITHOUT making a generation request.

    Useful for:
        - Estimating cost before sending a large document
        - Checking if a prompt fits in the context window
        - Debugging token budget in RAG / agent pipelines

    Returns:
        Integer token count.
    """
    client = get_client()
    result = client.models.count_tokens(model=model, contents=prompt)
    return result.total_tokens


# --- Self-test when run directly ---
if __name__ == "__main__":
    print("Testing Gemini client...\n")

    # Token count test
    test_prompt = "Hello, this is a test of the Gemini client."
    tokens = count_tokens(test_prompt)
    print(f"Token count for test prompt: {tokens}")

    # Generation test
    print("\nCalling generate()...")
    result = generate("Say hello in 5 different languages, one per line.")
    print(result)

    # Metadata test
    print("\nCalling generate_with_metadata()...")
    meta = generate_with_metadata("What is 2 + 2? Answer in one word.")
    print(f"Answer      : {meta['text'].strip()}")
    print(f"Total tokens: {meta['total_tokens']}")
    print(f"Finish      : {meta['finish_reason']}")

    print("\nAll tests passed. Client is ready.")
