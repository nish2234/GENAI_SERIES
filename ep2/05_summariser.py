"""Pattern 5: Summarise text to bullet points with length control.

Demonstrates how to control summary length and format by specifying
exact bullet point counts and output formatting rules. Uses low
temperature for factual, consistent summaries.
"""

from gemini_client import generate

article = """
Artificial intelligence has transformed the healthcare industry in numerous ways 
over the past decade. Machine learning algorithms can now detect certain cancers 
from medical imaging with accuracy rates exceeding those of experienced 
radiologists. Natural language processing enables automated analysis of clinical 
notes, helping identify patients at risk of adverse events. Drug discovery 
timelines have been reduced from years to months through AI-powered molecular 
simulation. Remote patient monitoring using AI-driven wearables can predict 
cardiac events hours before they occur. However, challenges remain: data privacy 
concerns, algorithmic bias in underrepresented populations, the need for 
regulatory frameworks, and the risk of over-reliance on automated systems without 
human oversight. Despite these challenges, global investment in healthcare AI 
exceeded $45 billion in 2024, with projections suggesting the market will double 
by 2028.
"""

lengths = {
    "brief": "Summarise in exactly 2 bullet points. Be extremely concise.",
    "standard": "Summarise in 4-5 bullet points. Cover the key facts.",
    "detailed": "Summarise in 6-8 bullet points. Include specific numbers and examples.",
}


if __name__ == "__main__":
    print("=== Text Summariser with Length Control ===\n")

    for style, instruction in lengths.items():
        prompt = f"""{instruction}

Use this format:
• [bullet point]

Text to summarise:
\"\"\"
{article.strip()}
\"\"\"
"""
        print(f"--- {style.upper()} ---")
        result = generate(prompt, temperature=0.3)
        print(result.strip())
        print()
