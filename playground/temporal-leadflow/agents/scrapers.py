"""
Three parallel scraper agents (LLM-simulated).

Each agent receives the same niche+location prompt and produces
leads from a different source perspective. The MCP tool_executor
slot is wired but empty — swap in a real Playwright MCP URL to
enable actual browser scraping with no code changes.
"""

from __future__ import annotations

from orchestrator import AgentConfig, AgentMemoryConfig, BaseAgent

_NO_MEMORY = AgentMemoryConfig(search_memories=False, store_memories=False)


def _scraper(name: str, instructions: str, model: str) -> BaseAgent:
    return BaseAgent(
        name=name,
        instructions=instructions,
        model=model,
        memory_config=_NO_MEMORY,
        config=AgentConfig(log_to_session=False, session_history_turns=0, max_turns=3),
    )


def make_google_maps_agent(model: str, n: int = 5) -> BaseAgent:
    return _scraper(
        "google-maps-agent",
        (
            f"You simulate a Google Maps scraper. Given a niche and location, "
            f"output exactly {n} realistic fictional businesses as if scraped from Google Maps. "
            "For each business output a line in this exact format:\n"
            "NAME: <name> | ADDRESS: <address> | PHONE: <phone> | RATING: <X.X> | CATEGORY: <category>\n"
            "Use plausible but clearly invented data. "
            "Start your output with the header: [GOOGLE MAPS]"
        ),
        model,
    )


def make_linkedin_agent(model: str, n: int = 5) -> BaseAgent:
    return _scraper(
        "linkedin-agent",
        (
            f"You simulate a LinkedIn company scraper. Given a niche and location, "
            f"output exactly {n} realistic fictional companies as if found on LinkedIn. "
            "For each company output a line in this exact format:\n"
            "NAME: <name> | SIZE: <employees> | WEBSITE: <url> | CONTACT: <First Last, Title>\n"
            "Use plausible but clearly invented data. "
            "Start your output with the header: [LINKEDIN]"
        ),
        model,
    )


def make_web_agent(model: str, n: int = 5) -> BaseAgent:
    return _scraper(
        "web-agent",
        (
            f"You simulate a general web search scraper. Given a niche and location, "
            f"output exactly {n} realistic fictional businesses found via web search. "
            "For each business output a line in this exact format:\n"
            "NAME: <name> | WEBSITE: <url> | EMAIL: <email> | DESCRIPTION: <one sentence>\n"
            "Use plausible but clearly invented data. "
            "Start your output with the header: [WEB SEARCH]"
        ),
        model,
    )
