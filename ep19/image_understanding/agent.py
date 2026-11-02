"""
Episode 19 — App 1: Image Understanding
========================================
Run: cd code && adk web       (then pick "01_image_understanding" from the dropdown)

What to try in ADK Web:
  - Click the attach (paperclip) button, upload a receipt photo, and ask:
      "What store is this from and what was the total?"
  - Same session, no new upload: "Which item was the most expensive?"
  - Upload a PDF (an invoice or a 1-2 page doc): "Summarise this in 3 bullets."
  - "What exactly did I just send you?"  (forces the inspect_attachments tool)

Where to look:
  - Events tab → click YOUR message event: the upload is an inline_data part
    (mime_type + bytes) sitting right next to your text. That is all an upload is.
  - Events tab → inspect_attachments function_response: name, MIME type, size.
  - Artifacts tab → EMPTY. Uploading a file does NOT create an artifact.
    Saving files is App 2's job.

What this file does:
  - Gemini is natively multimodal: ADK Web sends the file to the model inline,
    so the agent can read images and PDFs with zero extra code.
  - One tool reads tool_context.user_content to show what the upload looks like
    from the Python side (we reuse this trick in App 2 to save uploads).
"""

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

MODEL = "gemini-3.5-flash-lite"


def inspect_attachments(tool_context: ToolContext) -> dict:
    """Lists the files attached to the user's current message.

    Use this when the user asks what they uploaded, or when you need the exact
    file name, type, or size of an attachment.

    Returns:
        A dict with a list of attachments (file name, MIME type, size in KB).
    """
    content = tool_context.user_content
    if not content or not content.parts:
        return {"status": "error", "error_message": "The current message has no content."}

    attachments = []
    for part in content.parts:
        if part.inline_data is None:
            continue
        blob = part.inline_data
        attachments.append({
            "file_name": blob.display_name or "(unnamed)",
            "mime_type": blob.mime_type,
            "size_kb": round(len(blob.data or b"") / 1024, 1),
        })

    if not attachments:
        return {
            "status": "error",
            "error_message": "No file is attached to this message. Ask the user to "
                             "use the attach button and resend.",
        }
    return {"status": "success", "count": len(attachments), "attachments": attachments}


root_agent = Agent(
    name="vision_assistant",
    model=MODEL,
    description="Reads images and PDFs the user uploads and answers questions about them.",
    instruction=(
        "You are a sharp visual assistant. Users attach images (photos, receipts, "
        "screenshots) and PDFs to their messages.\n"
        "- Describe or analyse the attached file directly; you can see it.\n"
        "- For receipts, extract: store, date, each line item with price, tax, "
        "and total. Format line items as a markdown table.\n"
        "- If text is unreadable, say [unclear] instead of guessing.\n"
        "- If the user asks what they uploaded (name, type, size), call "
        "inspect_attachments.\n"
        "- If there is no file and the question needs one, ask them to attach it."
    ),
    tools=[inspect_attachments],
)
