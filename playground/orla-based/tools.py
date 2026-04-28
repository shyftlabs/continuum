"""
In-process MCP tools using @function_tool + MCPServerFunction.

No external server needed — all tools run in the same Python process.
Demonstrates: @function_tool schema generation, sync + async tools,
MCPServerFunction registration.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from orchestrator.tools.mcp import MCPServerFunction, function_tool

# ---------------------------------------------------------------------------
# In-memory store (shared across the session)
# ---------------------------------------------------------------------------

PRODUCTS = [
    {"id": "p1", "name": "Dog Food (Dry) 5kg",       "price": 29.99, "category": "food",        "animal": "dog"},
    {"id": "p2", "name": "Cat Food (Wet) 12-pack",   "price": 18.99, "category": "food",        "animal": "cat"},
    {"id": "p3", "name": "Dog Leash (Nylon)",         "price": 12.99, "category": "accessories", "animal": "dog"},
    {"id": "p4", "name": "Cat Toy - Feather Wand",   "price":  8.99, "category": "toys",        "animal": "cat"},
    {"id": "p5", "name": "Dog Toy - Tennis Ball 3pk","price":  6.99, "category": "toys",        "animal": "dog"},
    {"id": "p6", "name": "Pet Shampoo (All breeds)", "price": 11.99, "category": "grooming",    "animal": "all"},
    {"id": "p7", "name": "Cat Litter (Clumping) 10L","price": 15.99, "category": "litter",      "animal": "cat"},
    {"id": "p8", "name": "Dog Collar (Adjustable)",  "price":  9.99, "category": "accessories", "animal": "dog"},
]

_cart: list[dict] = []


# ---------------------------------------------------------------------------
# Tool definitions — natural Python signatures, schema auto-generated
# ---------------------------------------------------------------------------

@function_tool
def search_products(query: str) -> list:
    """Search pet products by keyword. Returns up to 5 matches."""
    words = query.lower().split()
    results = [
        p for p in PRODUCTS
        if any(w in p["name"].lower() or w in p["category"] or w in p["animal"] for w in words)
    ]
    return results[:5]


@function_tool
def get_product(product_id: str) -> dict:
    """Get full details for a product by its ID (e.g. p1, p2)."""
    for p in PRODUCTS:
        if p["id"] == product_id:
            return p
    return {"error": f"Product '{product_id}' not found"}


@function_tool
def add_to_cart(product_id: str, quantity: int) -> dict:
    """Add a product to the cart."""
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not product:
        return {"error": f"Product '{product_id}' not found"}
    for item in _cart:
        if item["product_id"] == product_id:
            item["quantity"] += quantity
            return {"message": f"Updated {product['name']} quantity", "cart_size": len(_cart)}
    _cart.append({
        "product_id": product_id,
        "name": product["name"],
        "price": product["price"],
        "quantity": quantity,
    })
    return {"message": f"Added {product['name']} to cart", "cart_size": len(_cart)}


@function_tool
def view_cart() -> dict:
    """View current cart contents and total price."""
    if not _cart:
        return {"items": [], "total": 0.0, "message": "Cart is empty"}
    total = sum(i["price"] * i["quantity"] for i in _cart)
    return {"items": _cart, "total": round(total, 2), "item_count": len(_cart)}


@function_tool
def checkout() -> dict:
    """Complete purchase for all items in the cart."""
    if not _cart:
        return {"error": "Cart is empty — add items first"}
    total = sum(i["price"] * i["quantity"] for i in _cart)
    order_id = f"ORD-{abs(hash(str(_cart))) % 100000:05d}"
    items_bought = len(_cart)
    _cart.clear()
    return {
        "order_id": order_id,
        "total": round(total, 2),
        "items_purchased": items_bought,
        "message": f"Order {order_id} placed successfully!",
    }


# ---------------------------------------------------------------------------
# MCPServerFunction — wraps all tools as an in-process MCP server
# ---------------------------------------------------------------------------

def build_tool_server() -> MCPServerFunction:
    """Create the in-process MCP server with all shop tools."""
    return MCPServerFunction(
        "shop-tools",
        [search_products, get_product, add_to_cart, view_cart, checkout],
    )


def reset_cart() -> None:
    """Clear the cart (used between demo runs)."""
    _cart.clear()
