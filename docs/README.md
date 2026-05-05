# Orchestrator SDK Documentation

Welcome to the Orchestrator SDK documentation. This SDK provides a unified interface for agentic AI orchestration with multi-LLM provider support, memory management, and observability.

## Quick Links

- [Feature Reference](features.md) - Complete list of all features
- [Installation](installation.md) - Setup and installation instructions
- [Agent](agent.md) - Agent creation, execution, and workflows
- [LLM](llm.md) - Multi-provider LLM client and context management
- [Structured JSON Mode](json_mode_support.md) - JSON mode with automatic schema validation
- [Memory](memory.md) - Long-term memory with mem0 and Qdrant
- [Session](session.md) - Short-term conversation history with Redis
- [Observability](observability.md) - Tracing, metrics, and Langfuse integration
- [Tools](tools.md) - MCP integration and tool execution
- [Core](core.md) - Container, lifecycle, health checks, and configuration

## Overview

The Orchestrator SDK enables you to:

- Build AI agents with multi-LLM provider support (100+ providers via LiteLLM)
- **Structured JSON outputs** with automatic schema validation (Pydantic models or JSON schemas)
- **Automatic compatibility handling** for models that don't support tools + JSON mode simultaneously
- Manage long-term memory with mem0 and Qdrant
- Handle conversation sessions with Redis
- Monitor and trace agent execution with Langfuse
- Execute tools via Model Context Protocol (MCP)
- Orchestrate multi-agent workflows

## Quick Start

```python
from orchestrator.agent import BaseAgent, AgentRunner

# Create an agent
agent = BaseAgent(
    name="my-agent",
    instructions="You are a helpful assistant.",
)

# Run the agent
runner = AgentRunner()
response = await runner.run(
    agent,
    "Hello!",
    user_id="user-123",
)

print(response.content)
```

For detailed information, see the [Installation Guide](installation.md) and module-specific documentation.
