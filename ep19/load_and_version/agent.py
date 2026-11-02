"""
Episode 19 — App 3: Load, List & Version Artifacts
===================================================
Run: cd code && adk web       (then pick "03_load_and_version" from the dropdown)

What to try in ADK Web:
  - "Draft a 3-line product tagline for a coffee app, save it as tagline.md"
  - "Make it punchier and save it again"  → twice more (versions 0, 1, 2)
  - "Show me the history of tagline.md"  then  "Read me version 0"
  - "Save my writing style to my library: short sentences, no emojis, British spelling"
  - Click "+ New session" in the Sessions list, then: "What files do I have?"
    → tagline.md is gone (session-scoped), user:style_guide.md is still there.
  - Attach a photo: "Keep this in my library as user:desk.jpg". New session:
    "Describe the desk photo from my library"  → load_artifacts pulls it back in.

Where to look:
  - Artifacts tab → tagline.md with versions 0/1/2; switch versions, download.
  - Sessions list → switch between sessions and watch user: files follow you.
  - Events tab → list_files / read_file / load_artifacts function_call events.

What this file does:
  - list_files / read_file / file_history: tool_context.list_artifacts(),
    load_artifact(filename, version=...), get_artifact_version(...).
  - "user:" prefix = user-scoped artifact that survives across sessions.
  - LoadArtifactsTool: ADK's built-in tool that lets the MODEL pull an
    artifact's content (even an image) back into its context on demand.
"""

from datetime import datetime

from google.adk.agents import Agent
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = "gemini-3.5-flash-lite"

TEXT_MIME_TYPES = {
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".json": "application/json",
}


def _mime_for(filename: str) -> str | None:
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return TEXT_MIME_TYPES.get(extension)


async def save_text_file(filename: str, content: str, tool_context: ToolContext) -> dict:
    """Saves text as a file artifact. Re-saving the same filename creates a new version.

    Prefix the filename with 'user:' (e.g. 'user:style_guide.md') to store it in the
    user's library so it is available in every session.

    Args:
        filename: File name ending in .md, .csv, .txt or .json, optionally with 'user:'.
        content: The full text content of the file.

    Returns:
        A dict with the filename, scope, and the version number assigned.
    """
    mime_type = _mime_for(filename)
    if mime_type is None:
        return {
            "status": "error",
            "error_message": f"Use one of these extensions: {', '.join(TEXT_MIME_TYPES)}",
        }
    artifact = types.Part.from_bytes(data=content.encode("utf-8"), mime_type=mime_type)
    version = await tool_context.save_artifact(filename=filename, artifact=artifact)
    scope = "user (all sessions)" if filename.startswith("user:") else "this session only"
    return {"status": "success", "filename": filename, "scope": scope, "version": version}


async def save_uploaded_file(filename: str, tool_context: ToolContext) -> dict:
    """Saves the file attached to the user's current message under the given name.

    Args:
        filename: Name to store it under, e.g. 'receipt.jpg' or 'user:desk.jpg'
            (the 'user:' prefix keeps it across sessions).

    Returns:
        A dict with the filename, MIME type, and version number.
    """
    content = tool_context.user_content
    parts = content.parts if content and content.parts else []
    uploads = [part for part in parts if part.inline_data is not None]
    if not uploads:
        return {
            "status": "error",
            "error_message": "No attachment on this message. Ask the user to attach it.",
        }
    version = await tool_context.save_artifact(filename=filename, artifact=uploads[0])
    return {
        "status": "success",
        "filename": filename,
        "mime_type": uploads[0].inline_data.mime_type,
        "version": version,
    }


async def list_files(tool_context: ToolContext) -> dict:
    """Lists every artifact the user can access right now.

    Returns:
        A dict with session-scoped files and user-library ('user:') files.
    """
    names = await tool_context.list_artifacts()
    return {
        "status": "success",
        "session_files": [n for n in names if not n.startswith("user:")],
        "library_files": [n for n in names if n.startswith("user:")],
    }


async def read_file(filename: str, tool_context: ToolContext, version: int | None = None) -> dict:
    """Reads a text artifact. Reads the latest version unless a version is given.

    For images or PDFs, use the load_artifacts tool instead.

    Args:
        filename: Exact artifact name, including any 'user:' prefix.
        version: Optional version number (0 is the first save). Omit for latest.

    Returns:
        A dict with the text content and which version was read.
    """
    part = await tool_context.load_artifact(filename=filename, version=version)
    if part is None:
        return {
            "status": "error",
            "error_message": f"No artifact '{filename}'"
                             + (f" at version {version}" if version is not None else "")
                             + ". Call list_files to see what exists.",
        }
    if part.inline_data is None or _mime_for(filename) is None:
        return {
            "status": "error",
            "error_message": f"'{filename}' is not a text file. Use load_artifacts to view it.",
        }
    return {
        "status": "success",
        "filename": filename,
        "version": "latest" if version is None else version,
        "content": part.inline_data.data.decode("utf-8"),
    }


async def file_history(filename: str, tool_context: ToolContext) -> dict:
    """Shows every saved version of an artifact with its timestamp.

    Args:
        filename: Exact artifact name, including any 'user:' prefix.

    Returns:
        A dict with one entry per version (version number, MIME type, saved time).
    """
    latest = await tool_context.get_artifact_version(filename=filename)
    if latest is None:
        return {"status": "error", "error_message": f"No artifact named '{filename}'."}

    history = []
    for v in range(latest.version + 1):
        info = await tool_context.get_artifact_version(filename=filename, version=v)
        if info is None:
            continue
        history.append({
            "version": info.version,
            "mime_type": info.mime_type,
            "saved_at": datetime.fromtimestamp(info.create_time).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return {"status": "success", "filename": filename, "versions": history}


root_agent = Agent(
    name="file_librarian",
    model=MODEL,
    description="Saves, lists, reads and versions files, including a cross-session user library.",
    instruction=(
        "You are a file librarian for the user.\n"
        "- Save drafts with save_text_file. To revise, save again with the SAME name.\n"
        "- If the user says 'my library', 'keep this', or 'remember this file', put "
        "'user:' in front of the filename so it survives across sessions.\n"
        "- Save attachments with save_uploaded_file.\n"
        "- Use list_files to see what exists, file_history for versions, and "
        "read_file (with a version number if asked) for text files.\n"
        "- To look at an image or PDF artifact, call load_artifacts with its name.\n"
        "- Always mention filename and version number when you save or read."
    ),
    tools=[
        save_text_file,
        save_uploaded_file,
        list_files,
        read_file,
        file_history,
        LoadArtifactsTool(),
    ],
)
