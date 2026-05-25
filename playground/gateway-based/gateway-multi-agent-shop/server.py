"""
Gateway Multi-Agent Shop MCP server — same products as multi-agent-shop, port 8890.

Run standalone:  python server.py
Tools: search_products, get_product, add_to_cart, view_cart, checkout
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gateway-workflow-shop")

_carts: dict[str, list[dict]] = {}

PRODUCTS = [
    {"id": "p1", "name": "Dog Food (Dry) 5kg",         "price": 29.99, "category": "food",        "animal": "dog"},
    {"id": "p2", "name": "Cat Food (Wet) 12-pack",      "price": 18.99, "category": "food",        "animal": "cat"},
    {"id": "p3", "name": "Dog Leash (Nylon)",            "price": 12.99, "category": "accessories", "animal": "dog"},
    {"id": "p4", "name": "Cat Toy - Feather Wand",       "price":  8.99, "category": "toys",        "animal": "cat"},
    {"id": "p5", "name": "Dog Toy - Tennis Ball 3-pack", "price":  6.99, "category": "toys",        "animal": "dog"},
    {"id": "p6", "name": "Pet Shampoo (All breeds)",     "price": 11.99, "category": "grooming",    "animal": "all"},
    {"id": "p7", "name": "Cat Litter (Clumping) 10L",    "price": 15.99, "category": "litter",      "animal": "cat"},
    {"id": "p8", "name": "Dog Collar (Adjustable)",      "price":  9.99, "category": "accessories", "animal": "dog"},
]


@mcp.tool()
def search_products(query: Optional[str] = None, animal: Optional[str] = None, category: Optional[str] = None) -> list[dict]:
    """Search for pet products. Filter by animal (dog/cat/all) or category."""
    results = PRODUCTS
    q = (query or "").lower()
    if q:
        results = [p for p in results if q in p["name"].lower() or q in p["category"]]
    if animal:
        results = [p for p in results if p["animal"] in ((animal or "").lower(), "all")]
    if category:
        results = [p for p in results if p["category"] == (category or "").lower()]
    return results[:5]


@mcp.tool()
def get_product(product_id: str) -> dict:
    """Get details for a specific product by ID."""
    for p in PRODUCTS:
        if p["id"] == product_id:
            return p
    return {"error": f"Product {product_id!r} not found"}


@mcp.tool()
def add_to_cart(session_id: str, product_id: str, quantity: int = 1) -> dict:
    """Add a product to the cart for a given session."""
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not product:
        return {"error": f"Product {product_id!r} not found"}
    cart = _carts.setdefault(session_id, [])
    for item in cart:
        if item["product_id"] == product_id:
            item["quantity"] += quantity
            return {"message": f"Updated quantity for {product['name']}", "cart_size": len(cart)}
    cart.append({"product_id": product_id, "name": product["name"],
                 "price": product["price"], "quantity": quantity})
    return {"message": f"Added {product['name']} to cart", "cart_size": len(cart)}


@mcp.tool()
def view_cart(session_id: str) -> dict:
    """View current cart contents and total for a session."""
    cart = _carts.get(session_id, [])
    if not cart:
        return {"items": [], "total": 0.0, "message": "Cart is empty"}
    total = sum(i["price"] * i["quantity"] for i in cart)
    return {"items": cart, "total": round(total, 2), "item_count": len(cart)}


@mcp.tool()
def checkout(session_id: str) -> dict:
    """Complete checkout for the session cart."""
    cart = _carts.get(session_id, [])
    if not cart:
        return {"error": "Cart is empty"}
    total = sum(i["price"] * i["quantity"] for i in cart)
    order_id = f"ORD-{abs(hash(session_id)) % 100000:05d}"
    _carts[session_id] = []
    return {
        "order_id": order_id,
        "total": round(total, 2),
        "items_purchased": len(cart),
        "message": f"Order {order_id} placed successfully!",
    }


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    print("Gateway Workflow Shop MCP server running at http://localhost:8890/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8890)
