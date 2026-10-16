"""
Episode 16 — App 02: State Shopping Cart
=========================================
Run: cd code && adk web       (then pick "02_state_shopping_cart" from the dropdown)

What this app does:
  - A shopping assistant whose cart lives in session state (state["cart"])
  - Tools read and write state through ToolContext — ADK records every change
    as a state_delta on the tool's event
  - The instruction uses {cart?} templating, so the model always sees the
    current cart (the ? means "empty if the key does not exist yet")

Try in ADK Web:
  - "What do you sell?"
  - "Add 2 coffees and a croissant."
  - "Actually remove one coffee. What's my total?"
  - Edit the cart in the State tab (e.g. set "croissant" to 5), then ask
    "What's in my cart now?"
  - Click "New Session" — the cart is empty again (no-prefix state is per session).

Look at: the State tab (cart updates after every tool call), and the Events tab
— click a function_response event and find its state_delta.
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext

MODEL = "gemini-3.1-flash-lite"

CATALOG = {
    "coffee": 3.50,
    "tea": 2.75,
    "croissant": 2.25,
    "bagel": 2.00,
    "muffin": 2.50,
    "orange juice": 4.00,
}


def _cart_summary(cart: dict) -> dict:
    lines = [
        {"item": item, "quantity": qty, "line_total": round(CATALOG[item] * qty, 2)}
        for item, qty in cart.items()
    ]
    total = round(sum(line["line_total"] for line in lines), 2)
    return {"items": lines, "total": total}


def list_products() -> dict:
    """List every product in the shop with its price.

    Returns:
        A dict with status and a products mapping of item name to price in USD.
    """
    return {"status": "success", "products": CATALOG}


def add_to_cart(item: str, quantity: int, tool_context: ToolContext) -> dict:
    """Add a product to the user's shopping cart.

    Args:
        item: The product name, exactly as listed by list_products (e.g., 'coffee').
        quantity: How many to add. Must be 1 or more.

    Returns:
        A dict with status and the updated cart (items and total) — or an
        error_message if the item is unknown or the quantity is invalid.
    """
    item = item.strip().lower()
    if item not in CATALOG:
        return {"status": "error", "error_message": f"We don't sell '{item}'. Try list_products."}
    if quantity < 1:
        return {"status": "error", "error_message": "quantity must be at least 1."}

    cart = dict(tool_context.state.get("cart", {}))
    cart[item] = cart.get(item, 0) + quantity
    tool_context.state["cart"] = cart
    return {"status": "success", "cart": _cart_summary(cart)}


def remove_from_cart(item: str, quantity: int, tool_context: ToolContext) -> dict:
    """Remove some or all of a product from the user's shopping cart.

    Args:
        item: The product name to remove (e.g., 'coffee').
        quantity: How many to remove. Removing at least as many as are in the
            cart removes the item completely.

    Returns:
        A dict with status and the updated cart (items and total) — or an
        error_message if the item is not in the cart.
    """
    item = item.strip().lower()
    cart = dict(tool_context.state.get("cart", {}))
    if item not in cart:
        return {"status": "error", "error_message": f"'{item}' is not in the cart."}

    remaining = cart[item] - max(quantity, 1)
    if remaining > 0:
        cart[item] = remaining
    else:
        del cart[item]
    tool_context.state["cart"] = cart
    return {"status": "success", "cart": _cart_summary(cart)}


def view_cart(tool_context: ToolContext) -> dict:
    """Show the current contents of the user's shopping cart and the total price.

    Returns:
        A dict with status and the cart (items with quantities, line totals, and total).
    """
    cart = {
        item: qty
        for item, qty in tool_context.state.get("cart", {}).items()
        if item in CATALOG and qty > 0
    }
    return {"status": "success", "cart": _cart_summary(cart)}


root_agent = Agent(
    name="shopping_cart_agent",
    model=MODEL,
    description="A coffee-shop assistant that manages a shopping cart in session state.",
    instruction=(
        "You are a cheerful coffee-shop assistant.\n"
        "- Use list_products when the user asks what is available.\n"
        "- Use add_to_cart / remove_from_cart to change the cart.\n"
        "- Use view_cart whenever the user asks what is in the cart or for the total "
        "— always call it rather than guessing, because the cart can change outside the chat.\n"
        "Current cart (item -> quantity): {cart?}\n"
        "If a tool returns status 'error', explain it in one sentence. Prices are in USD."
    ),
    tools=[list_products, add_to_cart, remove_from_cart, view_cart],
)
