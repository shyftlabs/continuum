from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from continuum.agent.services.tool_service import ToolService
from continuum.agent.utils.context_utils import create_run_context


async def test_info_logs_exclude_tool_arguments_and_results():
    secret_argument = "customer jane@example.test account 4242"
    secret_result = "Jane Example has an overdue balance of 9100"
    executor = SimpleNamespace(
        tool_registry={},
        execute_tool_calls=AsyncMock(
            return_value=[
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": secret_result,
                }
            ]
        ),
    )
    agent = SimpleNamespace(
        name="privacy-agent",
        config=None,
        mcp_servers=[],
        on_tool_call=None,
        policy_store=None,
        tool_executor=executor,
    )
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="lookup_customer", arguments=json.dumps({"query": secret_argument})),
    )

    with patch("continuum.agent.services.tool_service.logger.info") as info_logger:
        result, metadata = await ToolService().execute_tool_call(agent, tool_call, create_run_context())

    info_log = "\n".join(str(call) for call in info_logger.call_args_list)
    assert secret_argument not in info_log
    assert secret_result not in info_log
    assert "lookup_customer" in info_log
    assert result["content"] == secret_result
    assert metadata["success"] is True
