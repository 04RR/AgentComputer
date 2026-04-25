"""Configuration loader for agent_computer."""

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("agent_computer.config")


class ModelConfig(BaseModel):
    provider: str = "lmstudio"
    model_id: str = "qwen/qwen3.5-35b-a3b"
    base_url: str = "http://100.127.32.115:1234/v1"
    max_tokens: int = 4096


class LMStudioConfig(BaseModel):
    base_url: str = "http://100.127.32.115:1234/v1"
    api_key: str = "lm-studio"


class ToolsConfig(BaseModel):
    allow: list[str] = Field(default_factory=lambda: [
        "shell", "read_file", "write_file", "list_directory", "web_fetch", "manage_tasks", "memory_search",
        "browser_navigate", "browser_snapshot", "browser_click", "browser_type",
        "browser_fill", "browser_press", "browser_scroll", "browser_text",
        "browser_screenshot", "browser_tabs",
    ])
    require_approval: list[str] = Field(default_factory=lambda: ["shell"])
    # Policy for approval-requiring tools when there is no human in the loop
    # (HTTP /api/chat, cron jobs). "deny" is the safe default.
    non_interactive_approval_policy: Literal["deny", "auto_approve"] = "deny"
    # Verification-mode tools registered in the registry but excluded from the
    # agent's allow list in Phase 1. Reserved for Phase 2 — no code reads this
    # in Phase 1; the /api/verify/raw endpoint calls these tools directly.
    verification_tools: list[str] = Field(default_factory=lambda: [
        "reverse_image_search",
        "extract_image_metadata",
        "fact_check_lookup",
    ])
    shell_blocked_commands: list[str] = Field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:", "> /dev/sda",
        "chmod -R 777 /", "curl | sh", "wget | sh",
        "curl|sh", "wget|sh",
    ])
    shell_allow_absolute_paths: bool = False


class DeepWorkConfig(BaseModel):
    max_iterations: int = 200
    token_budget: int = 500000
    warning_threshold: float = 0.8


class ReflectionConfig(BaseModel):
    enabled: bool = True
    model_id: str = ""  # empty = use agent's model
    max_tokens: int = 4096
    max_sessions_per_startup: int = 10


class MemoryConfig(BaseModel):
    enabled: bool = True
    embedding_base_url: str = "http://100.127.32.115:1234/v1"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    top_k: int = 5


class VerificationConfig(BaseModel):
    """API keys + dev knobs for Phase 1 verification-mode tools.

    tineye_stub_mode:
      "off"  — call the real TinEye API (requires tineye_api_key)
      "hit"  — return a canned misattribution-style response (no key needed)
      "miss" — return an empty result (no key needed)
    """
    tineye_api_key: str = ""
    tineye_api_url: str = "https://api.tineye.com/rest/"
    google_factcheck_api_key: str = ""
    tineye_stub_mode: Literal["off", "hit", "miss"] = "off"


class PinchTabConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:9867"
    token: str = ""
    default_profile: str = "agent"
    headless: bool = True


class AgentConfig(BaseModel):
    id: str = "main"
    name: str = "agent_computer Agent"
    workspace: str = "./workspace"
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    max_loop_iterations: int = 15
    deep_work: DeepWorkConfig = Field(default_factory=DeepWorkConfig)


class SessionsConfig(BaseModel):
    directory: str = "./sessions"
    max_history_tokens: int = 100000


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Config(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sessions: SessionsConfig = Field(default_factory=SessionsConfig)
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    pinchtab: PinchTabConfig = Field(default_factory=PinchTabConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    openrouter_api_key: str = "ADD_KEY_HERE"


def load_config(path: str = "config.json") -> Config:
    """Load config from JSON file, falling back to defaults."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        try:
            config = Config.model_validate(raw)
        except ValidationError as e:
            logger.error(f"Config validation failed:\n{e}")
            raise
    else:
        config = Config()

    config.agent.workspace = str(Path(config.agent.workspace).resolve())
    Path(config.sessions.directory).mkdir(parents=True, exist_ok=True)
    Path(config.agent.workspace).mkdir(parents=True, exist_ok=True)

    return config
