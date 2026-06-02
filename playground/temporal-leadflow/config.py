import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=True)

_file_env = dotenv_values(_ENV_PATH)
for _var in ("SMART_GATEWAY_URL", "SMART_GATEWAY_API_KEY"):
    if _var not in _file_env:
        os.environ.pop(_var, None)

from dataclasses import dataclass


@dataclass
class LeadFlowConfig:
    model: str = "gpt-4o-mini"
    max_turns: int = 8
    leads_per_source: int = 5
    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"
    task_queue: str = "leadflow"
    enable_tracing: bool = True
    approval_timeout: int = 86400  # 24h


default_config = LeadFlowConfig()
