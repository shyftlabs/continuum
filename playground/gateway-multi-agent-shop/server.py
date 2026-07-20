"""
Gateway Multi-Agent Shop MCP server — same products as multi-agent-shop, port 8890.

Run standalone:  python server.py
Tools: search_products, get_product, add_to_cart, view_cart, checkout
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gateway-workflow-shop")

_carts: dict[str, list[dict]] = {}

PRODUCTS = [
    {"id": "p1", "name": "Dog Food (Dry) 5kg", "price": 29.99, "category": "food", "animal": "dog"},
    {
        "id": "p2",
        "name": "Cat Food (Wet) 12-pack",
        "price": 18.99,
        "category": "food",
        "animal": "cat",
    },
    {
        "id": "p3",
        "name": "Dog Leash (Nylon)",
        "price": 12.99,
        "category": "accessories",
        "animal": "dog",
    },
    {
        "id": "p4",
        "name": "Cat Toy - Feather Wand",
        "price": 8.99,
        "category": "toys",
        "animal": "cat",
    },
    {
        "id": "p5",
        "name": "Dog Toy - Tennis Ball 3-pack",
        "price": 6.99,
        "category": "toys",
        "animal": "dog",
    },
    {
        "id": "p6",
        "name": "Pet Shampoo (All breeds)",
        "price": 11.99,
        "category": "grooming",
        "animal": "all",
    },
    {
        "id": "p7",
        "name": "Cat Litter (Clumping) 10L",
        "price": 15.99,
        "category": "litter",
        "animal": "cat",
    },
    {
        "id": "p8",
        "name": "Dog Collar (Adjustable)",
        "price": 9.99,
        "category": "accessories",
        "animal": "dog",
    },
]


@mcp.tool()
def search_products(
    query: str | None = None, animal: str | None = None, category: str | None = None
) -> list[dict]:
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
    cart.append(
        {
            "product_id": product_id,
            "name": product["name"],
            "price": product["price"],
            "quantity": quantity,
        }
    )
    return {"message": f"Added {product['name']} to cart", "cart_size": len(cart)}


@mcp.tool()
def view_cart(session_id: str) -> dict:
    """View current cart contents and total for a session."""
    cart = _carts.get(session_id, [])
    if not cart:
        return {"items": [], "total": 0.0, "message": "Cart is empty"}
    total = sum(i["price"] * i["quantity"] for i in cart)
    return {"items": cart, "total": round(total, 2), "item_count": len(cart)}


_WAREHOUSES = ["us-east-1", "us-west-2", "eu-central-1", "ap-south-1"]
_SUPPLIERS = ["Acme Pet Supply", "Global Kibble Co", "PawParts Ltd", "Whisker & Co", "FetchWorks"]


@mcp.tool()
def fetch_inventory(animal: str | None = None) -> list[dict]:
    """Return the FULL warehouse inventory: every SKU with stock levels, supplier,
    warehouse, rating and a description. Hundreds of rows — use this for a full
    catalog audit or stock report (NOT for a normal product search)."""
    rows: list[dict] = []
    for i in range(200):
        base = PRODUCTS[i % len(PRODUCTS)]
        if animal and base["animal"] not in ((animal or "").lower(), "all"):
            continue
        rows.append(
            {
                "sku": f"SKU-{10000 + i}",
                "product_id": base["id"],
                "name": base["name"],
                "price": round(base["price"] * (1 + (i % 7) * 0.01), 2),
                "category": base["category"],
                "animal": base["animal"],
                "stock_qty": 5 + (i * 13) % 480,
                "warehouse": _WAREHOUSES[i % len(_WAREHOUSES)],
                "supplier": _SUPPLIERS[i % len(_SUPPLIERS)],
                "rating": round(3.0 + (i % 20) * 0.1, 1),
                "last_restock": f"2026-{1 + i % 12:02d}-{1 + i % 28:02d}",
                "description": (
                    f"Batch {i:03d} of {base['name']} — {base['category']} line for "
                    f"{base['animal']} owners. Quality-checked, shelf-stable, "
                    f"restocked from {_SUPPLIERS[i % len(_SUPPLIERS)]}."
                ),
            }
        )
    return rows


# A large service config so a read→write cycle makes the stale read worth
# crushing to a CCR marker (the only shape that issues a retrieve ticket).
_SERVICE_CONFIG = "shop-service config v3\n" + "\n".join(
    f"{svc}:\n  replicas: {2 + i % 5}\n  pool_max_size: {10 + i % 40}\n  "
    f"timeout_ms: {500 + i * 11}\n  region: us-east-{i % 4}\n  "
    f"cache_ttl_s: {30 + i % 300}\n  max_conns: {50 + i % 450}\n  "
    f"feature_flags: [flag_{i % 9}, flag_{(i + 3) % 9}]"
    for i, svc in enumerate(f"service_{n:03d}" for n in range(25))
)
_config_store: dict[str, str] = {"service.yaml": _SERVICE_CONFIG}


@mcp.tool()
def read(path: str = "service.yaml") -> str:
    """Read a config file (large YAML with every service's pool sizes, timeouts,
    regions, and feature flags). Use path='service.yaml'."""
    return _config_store.get(path, f"# {path} not found")


@mcp.tool()
def write(path: str, content: str) -> str:
    """Overwrite a config file with new content."""
    _config_store[path] = content
    return f"Wrote {len(content)} bytes to {path}. Change applied."


_LOG_LEVELS = ["INFO", "INFO", "INFO", "INFO", "WARN", "ERROR"]
_ORDER_EVENTS = [
    "order.created",
    "order.paid",
    "order.shipped",
    "cart.updated",
    "payment.authorized",
    "inventory.reserved",
]


@mcp.tool()
def fetch_order_logs(limit: int = 300) -> str:
    """Return the raw order-service event log (hundreds of lines) for the last
    period. Use this to investigate recent order activity, anomalies, or errors.
    Returns plain log lines, not structured products."""
    n = max(1, min(limit, 1000))
    lines = []
    for i in range(n):
        lines.append(
            f"2026-07-{1 + i % 12:02d}T{i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}Z "
            f"{_LOG_LEVELS[i % len(_LOG_LEVELS)]} order-svc "
            f"event={_ORDER_EVENTS[i % len(_ORDER_EVENTS)]} order_id=ORD-{50000 + i} "
            f"user=u{1000 + i % 400} amount_usd={round(5 + (i * 3.7) % 295, 2)} "
            f"items={1 + i % 6} status={'ok' if i % 11 else 'retry'} "
            f"latency_ms={20 + (i * 13) % 600}"
        )
    return "\n".join(lines)


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
