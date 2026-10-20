"""
Episode 17 — App 4: Action Confirmations (human-in-the-loop approval)
======================================================================
Run: cd code && adk web       (then pick "04_confirmation_tools" in the dropdown)

What this app shows (ADK Tool Confirmation — experimental feature):
  1. Boolean confirmation:   FunctionTool(close_account, require_confirmation=True)
     -> EVERY call pauses for a yes/no in ADK Web.
  2. Conditional confirmation: FunctionTool(transfer_money, require_confirmation=needs_approval)
     -> only transfers above $500 pause; small ones go straight through.
  3. Advanced confirmation:  tool_context.request_confirmation(hint=..., payload=...)
     -> the approver sends back structured data (here: how much overdraft to approve).

What to try in ADK Web:
  - "What's my balance?"
  - "Send $50 to Priya"                 (no prompt — under the threshold)
  - "Transfer $2000 to my landlord"     (confirmation prompt appears — approve it)
  - "Transfer $900 to Sam"              (prompt again — reject it this time)
  - "I need a $1500 overdraft"          (advanced prompt — edit approved_amount to 800)
  - "Close my account"                  (always asks)

Panel to watch:
  - Chat: the confirmation dialog ADK Web shows for the paused tool call
  - Events tab: the adk_request_confirmation function_call, then your response
  - State tab: 'balance' only changes AFTER you approve
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext

MODEL = "gemini-3.5-flash-lite"

STARTING_BALANCE = 3000.0
APPROVAL_THRESHOLD = 500.0


def check_balance(tool_context: ToolContext) -> dict:
    """Get the user's current account balance in US dollars.

    Returns:
        A dict with status and the current balance.
    """
    balance = tool_context.state.get("balance", STARTING_BALANCE)
    tool_context.state["balance"] = balance
    return {"status": "success", "balance": balance}


def transfer_money(amount: float, to_account: str, tool_context: ToolContext) -> dict:
    """Transfer money from the user's account to another person or account.

    Args:
        amount: Amount in US dollars to send, for example 250.
        to_account: Name of the recipient or account, for example 'Priya'.

    Returns:
        A dict with status, the amount sent, the recipient, and the new balance.
    """
    balance = tool_context.state.get("balance", STARTING_BALANCE)
    if amount <= 0:
        return {"status": "error", "error_message": "Amount must be greater than zero."}
    if amount > balance:
        return {
            "status": "error",
            "error_message": f"Insufficient funds: balance is ${balance:.2f}, tried to send ${amount:.2f}.",
        }

    tool_context.state["balance"] = balance - amount
    return {
        "status": "success",
        "sent": amount,
        "to": to_account,
        "new_balance": balance - amount,
    }


def needs_approval(amount: float, to_account: str, tool_context: ToolContext) -> bool:
    """Only transfers above the threshold need a human to approve them."""
    return amount > APPROVAL_THRESHOLD


def close_account(reason: str, tool_context: ToolContext) -> dict:
    """Permanently close the user's bank account.

    Args:
        reason: Why the user wants to close the account.

    Returns:
        A dict with status and a confirmation message.
    """
    tool_context.state["account_status"] = "closed"
    tool_context.state["balance"] = 0.0
    return {"status": "success", "message": f"Account closed. Reason: {reason}"}


def request_overdraft(amount: float, tool_context: ToolContext) -> dict:
    """Request an overdraft limit. A bank manager decides how much to approve.

    Args:
        amount: The overdraft limit the user is asking for, in US dollars.

    Returns:
        A dict with status and the approved overdraft amount.
    """
    confirmation = tool_context.tool_confirmation
    if not confirmation:
        tool_context.request_confirmation(
            hint=(
                f"Customer requests a ${amount:.0f} overdraft. "
                "Set approved_amount to the limit you approve (0 to reject)."
            ),
            payload={"approved_amount": 0},
        )
        return {"status": "pending", "message": "Waiting for manager approval."}

    if not confirmation.confirmed:
        return {"status": "rejected", "approved_amount": 0}

    approved = amount
  

    tool_context.state["overdraft_limit"] = approved
    return {"status": "approved", "requested": amount, "approved_amount": approved}


root_agent = Agent(
    name="bank_assistant",
    model=MODEL,
    description="A banking assistant that asks a human before risky actions.",
    instruction=(
        "You are a careful banking assistant.\n"
        "- Use check_balance for balance questions.\n"
        "- Use transfer_money for any payment or transfer. Pass only the number as amount.\n"
        "- Use request_overdraft for overdraft requests.\n"
        "- Use close_account only if the user clearly asks to close their account.\n"
        "Some actions need human approval. If a tool result says it was rejected, "
        "tell the user plainly and do not retry it."
    ),
    tools=[
        check_balance,
        FunctionTool(transfer_money, require_confirmation=needs_approval),
        FunctionTool(close_account, require_confirmation=True),
        request_overdraft,
    ],
)
