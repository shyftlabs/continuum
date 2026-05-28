"""
E2E test fixtures and shared helpers.
"""

from __future__ import annotations

import functools
import os

import pytest


def skip_if_no_api_key():
    """Skip test if no LLM API key configured."""
    has_key = any(os.getenv(k) for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"])
    if not has_key:
        pytest.skip("No LLM API key configured")


def skip_on_api_error(func):
    """Decorator to skip tests when API key is expired/invalid.

    Checks the full exception chain, not just the top-level message.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check the full exception chain for API key issues
            err_chain = str(e).lower()
            cause = e
            while cause.__cause__ is not None:
                cause = cause.__cause__
                err_chain += " " + str(cause).lower()
            if cause.__context__ is not None:
                err_chain += " " + str(cause.__context__).lower()

            api_keywords = ["expired", "api_key_invalid", "invalid_argument", "api key", "renew"]
            if any(kw in err_chain for kw in api_keywords):
                pytest.skip(f"API key issue (skipped): {type(e).__name__}")
            raise

    return wrapper
