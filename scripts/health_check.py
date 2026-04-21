#!/usr/bin/env python3
"""
Orchestrator SDK Health Check Script

A standalone script to verify all SDK dependencies are properly configured
and accessible. Run this to diagnose connectivity issues.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --json
    python scripts/health_check.py --service redis
    python scripts/health_check.py --timeout 30

Environment:
    Requires .env file or environment variables configured.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars


def print_banner():
    """Print SDK health check banner."""
    print("\n" + "=" * 60)
    print("  🏥 Orchestrator SDK Health Check")
    print("=" * 60 + "\n")


def print_status_icon(status: str) -> str:
    """Get status icon."""
    icons = {
        "healthy": "✅",
        "unhealthy": "❌",
        "degraded": "⚠️",
        "unknown": "❓",
    }
    return icons.get(status, "❓")


def print_check_result(check: dict, verbose: bool = False):
    """Print a single health check result."""
    icon = print_status_icon(check["status"])
    name = check["name"].upper()
    status = check["status"].upper()
    message = check["message"]
    latency = check.get("latency_ms", 0)
    
    print(f"  {icon} {name:12} │ {status:10} │ {latency:>7.1f}ms │ {message}")
    
    if verbose and check.get("details"):
        for key, value in check["details"].items():
            if key not in ("enabled", "configured"):
                print(f"     └─ {key}: {value}")


async def run_health_check(
    timeout: float = 10.0,
    service: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run health checks on SDK dependencies.
    
    Args:
        timeout: Timeout for all checks in seconds
        service: Specific service to check (redis, qdrant, langfuse, llm)
        verbose: Show detailed output
        
    Returns:
        Health check results dictionary
    """
    from orchestrator.core.health import get_health_checker, HealthStatus
    
    checker = get_health_checker()
    
    if service:
        # Check specific service
        check_methods = {
            "redis": checker.check_redis,
            "vector_store": checker.check_vector_store,
            "qdrant": checker.check_qdrant,
            "milvus": checker.check_milvus,
            "langfuse": checker.check_langfuse,
            "llm": checker.check_llm,
        }
        
        if service.lower() not in check_methods:
            print(f"Unknown service: {service}")
            print(f"Available services: {', '.join(check_methods.keys())}")
            sys.exit(1)
        
        check_fn = check_methods[service.lower()]
        result = await asyncio.wait_for(check_fn(), timeout=timeout)
        
        return {
            "status": result.status.value,
            "total_latency_ms": result.latency_ms,
            "checked_at": datetime.now().isoformat(),
            "checks": {result.name: result.to_dict()},
        }
    else:
        # Check all services
        result = await checker.check_all(timeout=timeout)
        return result.to_dict()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check health of Orchestrator SDK dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/health_check.py              # Check all services
  python scripts/health_check.py --json       # Output as JSON
  python scripts/health_check.py -s redis     # Check only Redis
  python scripts/health_check.py -v           # Verbose output
  python scripts/health_check.py -t 30        # 30 second timeout
        """,
    )
    
    parser.add_argument(
        "-s", "--service",
        choices=["redis", "vector_store", "qdrant", "milvus", "langfuse", "llm"],
        help="Check specific service only (vector_store dispatches to configured provider)",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only show failures (exit code indicates status)",
    )
    
    args = parser.parse_args()
    
    try:
        result = await run_health_check(
            timeout=args.timeout,
            service=args.service,
            verbose=args.verbose,
        )
    except asyncio.TimeoutError:
        if args.json:
            print(json.dumps({"status": "unhealthy", "error": "Health check timed out"}))
        else:
            print(f"❌ Health check timed out after {args.timeout}s")
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "unhealthy", "error": str(e)}))
        else:
            print(f"❌ Health check failed: {e}")
        sys.exit(1)
    
    # Output results
    if args.json:
        print(json.dumps(result, indent=2))
    elif not args.quiet:
        print_banner()
        
        # Summary
        status = result["status"]
        icon = print_status_icon(status)
        total_latency = result.get("total_latency_ms", 0)
        
        print(f"  Status: {icon} {status.upper()}")
        print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Latency: {total_latency:.1f}ms total")
        print()
        
        # Individual checks
        print("  " + "-" * 56)
        print(f"  {'SERVICE':14} │ {'STATUS':10} │ {'LATENCY':>9} │ MESSAGE")
        print("  " + "-" * 56)
        
        for name, check in result.get("checks", {}).items():
            print_check_result(check, verbose=args.verbose)
        
        print("  " + "-" * 56)
        print()
        
        # Summary counts
        checks = result.get("checks", {})
        healthy = sum(1 for c in checks.values() if c["status"] == "healthy")
        unhealthy = sum(1 for c in checks.values() if c["status"] == "unhealthy")
        degraded = sum(1 for c in checks.values() if c["status"] == "degraded")
        
        print(f"  Summary: {healthy} healthy, {degraded} degraded, {unhealthy} unhealthy")
        print()
        
        # Recommendations
        if unhealthy > 0:
            print("  💡 Recommendations:")
            for name, check in checks.items():
                if check["status"] == "unhealthy":
                    _print_recommendation(name, check)
            print()
    
    # Exit code based on status
    if result["status"] == "healthy":
        if not args.quiet:
            print("  ✅ All checks passed!\n")
        sys.exit(0)
    elif result["status"] == "degraded":
        if not args.quiet:
            print("  ⚠️  Some services degraded (non-critical)\n")
        sys.exit(0)  # Degraded is acceptable
    else:
        if not args.quiet:
            print("  ❌ Health check failed!\n")
        sys.exit(1)


def _print_recommendation(service: str, check: dict):
    """Print recommendation for a failing service."""
    recommendations = {
        "redis": [
            "  • Check if Redis is running: docker ps | grep redis",
            "  • Verify SESSION_REDIS_HOST and SESSION_REDIS_PORT in .env",
            "  • Test connection: redis-cli -h <host> -p <port> ping",
        ],
        "qdrant": [
            "  • Check if Qdrant is running: docker ps | grep qdrant",
            "  • Verify QDRANT_HOST and QDRANT_PORT in .env",
            "  • Test connection: curl http://<host>:<port>/collections",
        ],
        "langfuse": [
            "  • Verify LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env",
            "  • Check LANGFUSE_HOST points to correct server",
            "  • Test API keys at https://cloud.langfuse.com",
        ],
        "llm": [
            "  • Verify at least one API key is set (OPENAI_API_KEY, etc.)",
            "  • Check DEFAULT_LLM_MODEL matches your provider",
            "  • Test API key with provider's dashboard",
        ],
    }
    
    if service in recommendations:
        print(f"\n  {service.upper()}:")
        for rec in recommendations[service]:
            print(rec)


if __name__ == "__main__":
    asyncio.run(main())

