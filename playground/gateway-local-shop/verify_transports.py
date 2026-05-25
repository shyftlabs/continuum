"""
Verify all 4 MCP transport types work with Continuum.

For each transport, this script:
  1. Connects to the shop MCP server
  2. Discovers tools via executor.initialize()
  3. Calls search_products to verify tool execution works
  4. Reports pass/fail

Prerequisites:
  - StreamableHttp: python server.py          (port 8888)
  - SSE:            python server_sse.py       (port 8889)
  - Stdio:          server_stdio.py must be on disk (spawned as subprocess)
  - Function:       no server needed (in-process)

Run: python verify_transports.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from orchestrator.llm.types import FunctionCall, ToolCall
from orchestrator.tools.executor import ToolExecutor
from orchestrator.tools.mcp import (
    MCPServerFunction,
    MCPServerSse,
    MCPServerStdio,
    MCPServerStreamableHttp,
)

EXPECTED_TOOLS = {"search_products", "get_product", "add_to_cart", "view_cart", "checkout"}
STDIO_SERVER_PATH = os.path.join(os.path.dirname(__file__), "server_stdio.py")


def _search_tool_call() -> ToolCall:
    return ToolCall(
        id="verify-1",
        type="function",
        function=FunctionCall(
            name="search_products",
            arguments=json.dumps({"query": "dog", "animal": "dog"}),
        ),
    )


async def _verify(label: str, server) -> bool:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    try:
        await server.connect()
        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        # 1. Tool discovery
        defs = executor.get_tool_definitions()
        discovered = {d.function.name for d in defs}
        missing = EXPECTED_TOOLS - discovered
        if missing:
            print(f"  ✗ Missing tools: {missing}")
            return False
        print(f"  ✓ Tools discovered: {', '.join(sorted(discovered))}")

        # 2. Tool execution
        result = await executor.execute_tool_call(_search_tool_call())
        if not result.content:
            print(f"  ✗ search_products returned empty content")
            return False
        parsed = json.loads(result.content)
        if not isinstance(parsed, list) or len(parsed) == 0:
            print(f"  ✗ search_products returned no results: {result.content}")
            return False
        print(f"  ✓ search_products returned {len(parsed)} result(s)")

        # 3. Cart tool execution
        cart_call = ToolCall(
            id="verify-2",
            type="function",
            function=FunctionCall(
                name="add_to_cart",
                arguments=json.dumps({"session_id": "verify-session", "product_id": "p5", "quantity": 1}),
            ),
        )
        cart_result = await executor.execute_tool_call(cart_call)
        cart_data = json.loads(cart_result.content)
        if "error" in cart_data:
            print(f"  ✗ add_to_cart failed: {cart_data['error']}")
            return False
        print(f"  ✓ add_to_cart succeeded: {cart_data.get('message', '')}")

        print(f"  PASS")
        return True

    except (Exception, asyncio.CancelledError) as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        print(f"  FAIL")
        return False
    finally:
        try:
            await server.cleanup()
        except (Exception, asyncio.CancelledError):
            pass


def _make_function_server() -> MCPServerFunction:
    """In-process server — same shop logic as Python functions."""
    _carts: dict[str, list] = {}
    PRODUCTS = [
        {"id": "p1", "name": "Dog Food (Dry) 5kg", "price": 29.99, "category": "food", "animal": "dog"},
        {"id": "p2", "name": "Cat Food (Wet) 12-pack", "price": 18.99, "category": "food", "animal": "cat"},
        {"id": "p3", "name": "Dog Leash (Nylon)", "price": 12.99, "category": "accessories", "animal": "dog"},
        {"id": "p4", "name": "Cat Toy - Feather Wand", "price": 8.99, "category": "toys", "animal": "cat"},
        {"id": "p5", "name": "Dog Toy - Tennis Ball 3-pack", "price": 6.99, "category": "toys", "animal": "dog"},
    ]

    def search_products(query: str, animal: str = "", category: str = "") -> list:
        """Search for pet products. Filter by animal type (dog/cat/all) or category."""
        results = PRODUCTS
        if query:
            results = [p for p in results if query.lower() in p["name"].lower()]
        if animal:
            results = [p for p in results if p["animal"] in (animal.lower(), "all")]
        return results[:5]

    def get_product(product_id: str) -> dict:
        """Get details for a specific product by ID."""
        for p in PRODUCTS:
            if p["id"] == product_id:
                return p
        return {"error": f"Product {product_id!r} not found"}

    def add_to_cart(session_id: str, product_id: str, quantity: int = 1) -> dict:
        """Add a product to the cart for a given session."""
        product = next((p for p in PRODUCTS if p["id"] == product_id), None)
        if not product:
            return {"error": f"Product {product_id!r} not found"}
        cart = _carts.setdefault(session_id, [])
        cart.append({"product_id": product_id, "name": product["name"],
                     "price": product["price"], "quantity": quantity})
        return {"message": f"Added {product['name']} to cart", "cart_size": len(cart)}

    def view_cart(session_id: str) -> dict:
        """View current cart contents and total for a session."""
        cart = _carts.get(session_id, [])
        total = sum(i["price"] * i["quantity"] for i in cart)
        return {"items": cart, "total": round(total, 2), "item_count": len(cart)}

    def checkout(session_id: str) -> dict:
        """Complete checkout for the session cart."""
        cart = _carts.get(session_id, [])
        if not cart:
            return {"error": "Cart is empty"}
        total = sum(i["price"] * i["quantity"] for i in cart)
        order_id = f"ORD-{abs(hash(session_id)) % 100000:05d}"
        _carts[session_id] = []
        return {"order_id": order_id, "total": round(total, 2), "message": "Order placed!"}

    return MCPServerFunction(
        "local-shop-function",
        [search_products, get_product, add_to_cart, view_cart, checkout],
    )


def main():
    """
    Run each transport in its own anyio.run() call so each gets a fresh event loop.
    This prevents anyio cancel scopes from one transport's cleanup leaking into
    the next transport's session.initialize().
    """
    import anyio

    print("Verifying all 4 MCP transport types for local-shop\n")

    results = {}

    # 1. MCPServerFunction (in-process — no server needed)
    results["MCPServerFunction"] = anyio.run(
        _verify,
        "MCPServerFunction (in-process)",
        _make_function_server(),
    )

    # 2. MCPServerStreamableHttp (requires: python server.py on port 8888)
    results["MCPServerStreamableHttp"] = anyio.run(
        _verify,
        "MCPServerStreamableHttp → http://localhost:8888/mcp",
        MCPServerStreamableHttp(
            params={"url": "http://localhost:8888/mcp"},
            client_session_timeout_seconds=5,
            validate_on_connect=True,
        ),
    )

    # 3. MCPServerSse (requires: python server_sse.py on port 8889)
    results["MCPServerSse"] = anyio.run(
        _verify,
        "MCPServerSse → http://localhost:8889/sse",
        MCPServerSse(
            params={"url": "http://localhost:8889/sse", "timeout": 5},
            client_session_timeout_seconds=5,
            validate_on_connect=True,
        ),
    )

    # 4. MCPServerStdio (spawns server_stdio.py as subprocess)
    results["MCPServerStdio"] = anyio.run(
        _verify,
        f"MCPServerStdio → python {os.path.basename(STDIO_SERVER_PATH)}",
        MCPServerStdio(
            params={
                "command": sys.executable,
                "args": [STDIO_SERVER_PATH],
            },
            client_session_timeout_seconds=10,
            validate_on_connect=True,
        ),
    )

    # Summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for transport, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {transport}")

    total = sum(results.values())
    print(f"\n  {total}/{len(results)} transport types verified")

    if total < len(results):
        print("\n  NOTE: HTTP/SSE/Stdio failures may mean the server isn't running.")
        print("  Start servers first:")
        print("    python server.py        # StreamableHttp on port 8888")
        print("    python server_sse.py    # SSE on port 8889")
        print("  Stdio spawns automatically — no manual start needed.")


if __name__ == "__main__":
    main()
