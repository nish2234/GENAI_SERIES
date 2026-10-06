"""
Reusable Gemini Client — Episode 14 (Multimodal Agent — Vision + Text)
======================================================================
Multi-provider fallback: Gemini → Groq → Cerebras
Includes: generate(), generate_with_image(), count_tokens(), get_client()

New in Episode 14: generate_with_image() for sending images + text to Gemini.
"""

import io
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_MODEL = "groq/openai/gpt-oss-120b"
CEREBRAS_MODEL = "cerebras/gpt-oss-120b"
MAX_RETRIES = 2
INITIAL_WAIT = 2

_client: genai.Client | None = None


def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(k in error_str for k in [
        "503", "429", "overloaded", "resource exhausted",
        "deadline exceeded", "timeout", "unavailable", "internal", "500", "connection",
    ])


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def _call_gemini(prompt: str, temperature: float, max_tokens: int, system: str | None) -> str:
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_tokens, system_instruction=system,
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
            return response.text
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


def _call_litellm(model: str, prompt: str, temperature: float, max_tokens: int, system: str | None) -> str:
    import litellm
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


def _load_image(image_path: str) -> Image.Image:
    """Load an image from a local file path or URL."""
    if image_path.startswith(("http://", "https://")):
        import requests
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        return Image.open(image_path)


def generate(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str | None = None,
) -> str:
    """Send a text prompt, get a response. Falls through to Groq/Cerebras on failure."""
    try:
        return _call_gemini(prompt, temperature, max_tokens, system)
    except Exception:
        pass

    if os.environ.get("GROQ_API_KEY"):
        try:
            return _call_litellm(GROQ_MODEL, prompt, temperature, max_tokens, system)
        except Exception:
            pass

    if os.environ.get("CEREBRAS_API_KEY"):
        try:
            return _call_litellm(CEREBRAS_MODEL, prompt, temperature, max_tokens, system)
        except Exception:
            pass

    raise RuntimeError("All providers failed. Check your API keys.")


def generate_with_image(
    prompt: str,
    image_path: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system: str | None = None,
) -> str:
    """Send text + image to Gemini. Falls back to text-only via litellm if Gemini fails."""
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_tokens, system_instruction=system,
    )

    img = _load_image(image_path)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, img],
                config=config,
            )
            return response.text
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                break
            time.sleep(INITIAL_WAIT * (2 ** attempt))

    fallback_prompt = (
        f"{prompt}\n\n[Image was provided but cannot be forwarded to fallback provider]"
    )

    if os.environ.get("GROQ_API_KEY"):
        try:
            return _call_litellm(GROQ_MODEL, fallback_prompt, temperature, max_tokens, system)
        except Exception:
            pass

    if os.environ.get("CEREBRAS_API_KEY"):
        try:
            return _call_litellm(CEREBRAS_MODEL, fallback_prompt, temperature, max_tokens, system)
        except Exception:
            pass

    raise RuntimeError("All providers failed (including multimodal). Check your API keys.")


def count_tokens(prompt: str, model: str = GEMINI_MODEL) -> int:
    """Count tokens without generating a response."""
    client = get_client()
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.count_tokens(model=model, contents=prompt)
            return result.total_tokens
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


if __name__ == "__main__":
    print("Testing client (multi-provider fallback + multimodal)...\n")

    result = generate("Say hello in one sentence.")
    print(f"Text generate: {result[:80]}")

    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    result = generate_with_image("What do you see in this image? One sentence.", test_url)
    print(f"Image generate: {result[:100]}")

    print("\nAll tests passed.")
