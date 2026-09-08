"""
Episode 10 — File 2: Web Scraper Tool
=======================================
Run: python 02_scrape_tool.py

What this file does:
  - Fetches a web page using requests
  - Extracts readable text content using BeautifulSoup
  - Strips HTML tags, scripts, styles — gets clean text
  - Truncates to ~2000 chars (enough for the LLM to work with)
  - Handles errors gracefully (timeouts, bad URLs, etc.)

This is the second tool our research agent will use.
"""

import requests
from bs4 import BeautifulSoup


def scrape_url(url: str, max_chars: int = 2000) -> str:
    """
    Fetch a URL and extract readable text content.

    Returns clean text truncated to max_chars.
    Handles errors gracefully — returns error message on failure.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Truncate
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "\n\n[... content truncated ...]"

        return clean_text

    except requests.exceptions.Timeout:
        return f"[Error] Timeout while fetching {url}"
    except requests.exceptions.HTTPError as e:
        return f"[Error] HTTP {e.response.status_code} for {url}"
    except requests.exceptions.ConnectionError:
        return f"[Error] Could not connect to {url}"
    except Exception as e:
        return f"[Error] Failed to scrape {url}: {e}"


# --- Demo ---
if __name__ == "__main__":
    print("=" * 60)
    print("WEB SCRAPER TOOL — requests + BeautifulSoup")
    print("=" * 60)

    # Test with a well-known, stable page
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"\nScraping: {test_url}")
    print("-" * 60)

    content = scrape_url(test_url)
    print(content[:500])
    print(f"\n... ({len(content)} chars total)")

    print("\n" + "=" * 60)

    # Test error handling with a bad URL
    bad_url = "https://thisurldoesnotexist12345.com/page"
    print(f"\nScraping (bad URL): {bad_url}")
    print("-" * 60)
    content2 = scrape_url(bad_url)
    print(content2)

    print("\n" + "=" * 60)

    # Test with a real article
    article_url = "https://en.wikipedia.org/wiki/CRISPR"
    print(f"\nScraping: {article_url}")
    print("-" * 60)
    content3 = scrape_url(article_url, max_chars=1000)
    print(content3[:300])
    print(f"\n... ({len(content3)} chars total, truncated to 1000)")

    print("\nDone! Scraper working correctly.")
