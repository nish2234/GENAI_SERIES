"""
Episode 19 — App 2: Save Artifacts
===================================
Run: cd code && adk web       (then pick "02_save_artifacts" from the dropdown)

What to try in ADK Web:
  - Attach a receipt photo: "Save this file for me."
  - "Write a markdown summary of that receipt and save it as receipt_summary.md"
  - "Export the line items as a CSV called items.csv"
  - "Actually, add a column for category and save items.csv again"  (→ version 1)

Where to look:
  - Artifacts tab → the uploaded image (previews inline), receipt_summary.md,
    items.csv. Pick a version of items.csv and download it.
  - Events tab → the tool's event carries actions.artifact_delta,
    e.g. {"items.csv": 1}. That is how ADK records "a file changed here".
  - State tab → only a tiny pointer: last_saved_file. The bytes live in the
    artifact store, NOT in state.
  - On disk → code/02_save_artifacts/.adk/artifacts/ (adk web's default store)

What this file does:
  - save_uploaded_files: copies attachments from the user's message into
    artifacts with tool_context.save_artifact(filename, types.Part).
  - save_text_file: turns text the model wrote (markdown, CSV, JSON) into a
    real downloadable file artifact.
"""

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = "gemini-3.5-flash-lite"

TEXT_MIME_TYPES = {
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".json": "application/json",
}


async def save_uploaded_files(tool_context: ToolContext) -> dict:
    """Saves every file attached to the user's current message as an artifact.

    Call this in the same turn the user uploads a file and asks to save it.

    Returns:
        A dict listing each saved file name and the version number it got.
    """
    content = tool_context.user_content
    parts = content.parts if content and content.parts else []

    saved = []
    for index, part in enumerate(parts):
        if part.inline_data is None:
            continue
        filename = part.inline_data.display_name or f"upload_{index}"
        version = await tool_context.save_artifact(filename=filename, artifact=part)
        saved.append({"filename": filename, "version": version})

    if not saved:
        return {
            "status": "error",
            "error_message": "No attachment found on this message. Ask the user "
                             "to attach the file and send it again.",
        }
    tool_context.state["last_saved_file"] = saved[-1]["filename"]
    return {"status": "success", "saved": saved}


async def save_text_file(filename: str, content: str, tool_context: ToolContext) -> dict:
    """Saves text you have written as a downloadable file artifact.

    Saving to an existing filename creates a new version; nothing is overwritten.

    Args:
        filename: File name ending in .md, .csv, .txt or .json (e.g. 'summary.md').
        content: The full text content of the file.

    Returns:
        A dict with the filename, MIME type, and the version number assigned.
    """
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_type = TEXT_MIME_TYPES.get(extension)
    if mime_type is None:
        return {
            "status": "error",
            "error_message": f"Unsupported extension '{extension}'. "
                             f"Use one of: {', '.join(TEXT_MIME_TYPES)}",
        }

    artifact = types.Part.from_bytes(data=content.encode("utf-8"), mime_type=mime_type)
    version = await tool_context.save_artifact(filename=filename, artifact=artifact)
    tool_context.state["last_saved_file"] = filename
    return {"status": "success", "filename": filename, "mime_type": mime_type, "version": version}


root_agent = Agent(
    name="file_saver",
    model=MODEL,
    description="Analyses uploads and saves files (uploads, markdown, CSV) as ADK artifacts.",
    instruction=(
        "You help users turn uploads and analysis into real files.\n"
        "- When the user attaches a file and wants it kept, call save_uploaded_files.\n"
        "- When asked for a summary, table, or export, write the content yourself, "
        "then call save_text_file with a sensible filename (.md for summaries, "
        ".csv for tables).\n"
        "- After saving, tell the user the filename and version number, and that "
        "they can download it from the Artifacts tab.\n"
        "- To update a file, save it again with the SAME filename so a new "
        "version is created."
    ),
    tools=[save_uploaded_files, save_text_file],
)
