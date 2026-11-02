"""
Expense Assistant (FINAL BUILD)
====================================================
Run: cd code && adk web       (then pick "04_expense_assistant" from the dropdown)

What to try in ADK Web:
  - Attach a receipt photo (or PDF): "Log this receipt."
  - Attach a second receipt in the same session: "And this one."
  - "Generate my expense report for 2026-10"
  - "List my past reports"
  - Click "+ New session": "What have I spent on food this month?"
    then "Show me the coffee receipt again"  (load_artifacts brings the image back)
  - Upload the SAME receipt again: "Log this"  → duplicate is rejected.

Where to look:
  - Artifacts tab → user:receipts/<file> (image preview), user:expenses.csv
    (one new version per logged receipt), user:reports/expense_report_<month>.md
  - Events tab → record_expense function_call: the structured Receipt object the
    model extracted from the image (merchant, date, items[], tax, total).
  - State tab → user:receipt_count / user:last_receipt (small facts in state,
    files in artifacts).

What this file does:
  - Vision (Gemini reads the upload) + structured extraction (Pydantic tool arg)
    + artifacts: an archived receipt, a versioned CSV ledger, markdown reports.
  - Everything is "user:" scoped, so the ledger follows the user across sessions.
"""

import csv
import io
from collections import defaultdict

from google.adk.agents import Agent
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

MODEL = "gemini-3.5-flash-lite"

LEDGER = "user:expenses.csv"
LEDGER_COLUMNS = [
    "date", "merchant", "category", "item", "quantity",
    "unit_price", "line_total", "currency", "receipt_file",
]
CATEGORIES = ["food", "groceries", "travel", "office", "software", "other"]


class LineItem(BaseModel):
    name: str = Field(description="Item name as printed on the receipt.")
    quantity: float = Field(description="Quantity purchased. Use 1 if not shown.")
    unit_price: float = Field(description="Price per unit, as a plain number.")


class Receipt(BaseModel):
    merchant: str = Field(description="Store or vendor name.")
    date: str = Field(description="Purchase date as YYYY-MM-DD.")
    currency: str = Field(description="ISO currency code, e.g. USD, EUR, INR.")
    category: str = Field(description=f"One of: {', '.join(CATEGORIES)}.")
    items: list[LineItem] = Field(description="Every line item on the receipt.")
    tax: float = Field(default=0.0, description="Total tax shown, 0 if none.")
    total: float = Field(description="Grand total printed on the receipt.")


async def _read_ledger(tool_context: ToolContext) -> list[dict]:
    part = await tool_context.load_artifact(filename=LEDGER)
    if part is None or part.inline_data is None:
        return []
    return list(csv.DictReader(io.StringIO(part.inline_data.data.decode("utf-8"))))


async def _write_ledger(rows: list[dict], tool_context: ToolContext) -> int:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    artifact = types.Part.from_bytes(data=buffer.getvalue().encode("utf-8"), mime_type="text/csv")
    return await tool_context.save_artifact(filename=LEDGER, artifact=artifact)


async def archive_receipt(tool_context: ToolContext) -> dict:
    """Stores the receipt file attached to the user's current message.

    Call this FIRST whenever the user uploads a receipt.

    Returns:
        A dict with receipt_file (pass this to record_expense) and its version.
    """
    content = tool_context.user_content
    parts = content.parts if content and content.parts else []
    uploads = [part for part in parts if part.inline_data is not None]
    if not uploads:
        return {
            "status": "error",
            "error_message": "No receipt attached to this message. Ask the user to "
                             "attach a photo or PDF of the receipt.",
        }
    upload = uploads[0]
    original_name = upload.inline_data.display_name or f"receipt_{tool_context.invocation_id}"
    receipt_file = f"user:receipts/{original_name}"
    version = await tool_context.save_artifact(filename=receipt_file, artifact=upload)
    return {"status": "success", "receipt_file": receipt_file, "version": version}


async def record_expense(receipt: Receipt, receipt_file: str, tool_context: ToolContext) -> dict:
    """Adds an extracted receipt to the user's expense ledger (user:expenses.csv).

    Args:
        receipt: Structured data you extracted from the receipt image.
        receipt_file: The receipt_file value returned by archive_receipt.

    Returns:
        A dict with the new ledger version, rows added, and a total check.
    """
    if receipt.category not in CATEGORIES:
        return {
            "status": "error",
            "error_message": f"category must be one of {CATEGORIES}, got '{receipt.category}'.",
        }

    rows = await _read_ledger(tool_context)
    if any(row["receipt_file"] == receipt_file for row in rows):
        return {
            "status": "error",
            "error_message": f"{receipt_file} is already in the ledger. Not adding it twice.",
        }

    new_rows = [
        {
            "date": receipt.date,
            "merchant": receipt.merchant,
            "category": receipt.category,
            "item": item.name,
            "quantity": item.quantity,
            "unit_price": f"{item.unit_price:.2f}",
            "line_total": f"{item.quantity * item.unit_price:.2f}",
            "currency": receipt.currency,
            "receipt_file": receipt_file,
        }
        for item in receipt.items
    ]
    if receipt.tax:
        new_rows.append({
            "date": receipt.date, "merchant": receipt.merchant, "category": receipt.category,
            "item": "TAX", "quantity": 1, "unit_price": f"{receipt.tax:.2f}",
            "line_total": f"{receipt.tax:.2f}", "currency": receipt.currency,
            "receipt_file": receipt_file,
        })

    ledger_version = await _write_ledger(rows + new_rows, tool_context)

    computed = round(sum(float(r["line_total"]) for r in new_rows), 2)
    tool_context.state["user:receipt_count"] = tool_context.state.get("user:receipt_count", 0) + 1
    tool_context.state["user:last_receipt"] = f"{receipt.merchant} {receipt.date} {receipt.total}"

    return {
        "status": "success",
        "ledger": LEDGER,
        "ledger_version": ledger_version,
        "rows_added": len(new_rows),
        "total_check": {
            "printed_total": receipt.total,
            "sum_of_lines": computed,
            "matches": abs(computed - receipt.total) < 0.05,
        },
    }


async def generate_report(month: str, tool_context: ToolContext) -> dict:
    """Builds a markdown expense report for one month and saves it as an artifact.

    Args:
        month: Month to report on, formatted YYYY-MM (e.g. '2026-10').

    Returns:
        A dict with the report filename, version, and per-category totals.
    """
    rows = [row for row in await _read_ledger(tool_context) if row["date"].startswith(month)]
    if not rows:
        return {"status": "error", "error_message": f"No expenses recorded for {month}."}

    by_category: dict[str, float] = defaultdict(float)
    by_receipt: dict[str, dict] = {}
    for row in rows:
        amount = float(row["line_total"])
        by_category[row["category"]] += amount
        receipt = by_receipt.setdefault(
            row["receipt_file"],
            {"date": row["date"], "merchant": row["merchant"], "total": 0.0},
        )
        receipt["total"] += amount
    currency = rows[0]["currency"]
    grand_total = sum(by_category.values())

    lines = [
        f"# Expense Report — {month}",
        "",
        f"**Receipts:** {len(by_receipt)}  |  **Total:** {grand_total:.2f} {currency}",
        "",
        "## By category",
        "",
        "| Category | Amount |",
        "|---|---|",
        *[f"| {cat} | {amt:.2f} |" for cat, amt in sorted(by_category.items())],
        "",
        "## Receipts",
        "",
        "| Date | Merchant | Total | File |",
        "|---|---|---|---|",
        *[
            f"| {r['date']} | {r['merchant']} | {r['total']:.2f} | {name} |"
            for name, r in sorted(by_receipt.items(), key=lambda kv: kv[1]["date"])
        ],
    ]
    report_file = f"user:reports/expense_report_{month}.md"
    artifact = types.Part.from_bytes(data="\n".join(lines).encode("utf-8"), mime_type="text/markdown")
    version = await tool_context.save_artifact(filename=report_file, artifact=artifact)

    return {
        "status": "success",
        "report_file": report_file,
        "version": version,
        "grand_total": round(grand_total, 2),
        "currency": currency,
        "by_category": {cat: round(amt, 2) for cat, amt in by_category.items()},
    }


async def list_reports(tool_context: ToolContext) -> dict:
    """Lists the user's saved expense reports and archived receipts.

    Returns:
        A dict with report filenames, receipt filenames, and whether a ledger exists.
    """
    names = await tool_context.list_artifacts()
    return {
        "status": "success",
        "reports": sorted(n for n in names if n.startswith("user:reports/")),
        "receipts": sorted(n for n in names if n.startswith("user:receipts/")),
        "has_ledger": LEDGER in names,
    }


root_agent = Agent(
    name="expense_assistant",
    model=MODEL,
    description="Scans receipts, keeps a CSV expense ledger, and writes monthly reports.",
    instruction=(
        "You are a meticulous expense assistant.\n"
        "When the user uploads a receipt:\n"
        "  1. Call archive_receipt.\n"
        "  2. Read the receipt carefully and extract merchant, date (YYYY-MM-DD), "
        "currency, a category, every line item (name, quantity, unit_price), tax and "
        "the printed total. Never invent items; skip lines you cannot read.\n"
        "  3. Call record_expense with that data and the receipt_file from step 1.\n"
        "  4. Reply with a short markdown table of the items and the total. If "
        "total_check.matches is false, warn the user and show both numbers.\n"
        "For reports, call generate_report with the month as YYYY-MM, then summarise "
        "the category totals and tell the user the report is in the Artifacts tab.\n"
        "For questions about past spending, call load_artifacts on user:expenses.csv "
        "and answer from it. To show or re-check a past receipt, call load_artifacts "
        "with its user:receipts/ name. Use list_reports to see what exists."
    ),
    tools=[archive_receipt, record_expense, generate_report, list_reports, LoadArtifactsTool()],
)
