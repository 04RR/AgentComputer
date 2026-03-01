"""Configuration loader for agent_computer."""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    provider: str = "lmstudio"
    model_id: str = "qwen/qwen3.5-35b-a3b"
    base_url: str = "http://10.5.0.2:1234/v1"
    max_tokens: int = 4096


@dataclass
class LMStudioConfig:
    base_url: str = "http://10.5.0.2:1234/v1"
    api_key: str = "lm-studio"


@dataclass
class ToolsConfig:
    allow: list[str] = field(default_factory=lambda: [
        "shell", "read_file", "write_file", "list_directory", "web_fetch", "manage_tasks", "memory_search"
    ])
    require_approval: list[str] = field(default_factory=lambda: ["shell"])


@dataclass
class DeepWorkConfig:
    max_iterations: int = 200
    token_budget: int = 500000
    warning_threshold: float = 0.8


@dataclass
class ReflectionConfig:
    enabled: bool = True
    model_id: str = ""  # empty = use agent's model
    max_tokens: int = 4096
    max_sessions_per_startup: int = 10


@dataclass
class MemoryConfig:
    enabled: bool = True
    embedding_base_url: str = "http://10.5.0.2:1234/v1"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    top_k: int = 5


@dataclass
class AgentConfig:
    id: str = "main"
    name: str = "agent_computer Agent"
    workspace: str = "./workspace"
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    max_loop_iterations: int = 15
    deep_work: DeepWorkConfig = field(default_factory=DeepWorkConfig)


@dataclass
class SessionsConfig:
    directory: str = "./sessions"
    max_history_tokens: int = 100000


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config(path: str = "config.json") -> Config:
    """Load config from JSON file, falling back to defaults."""
    config = Config()

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            raw = json.load(f)

        if "gateway" in raw:
            gw = raw["gateway"]
            config.gateway.host = gw.get("host", config.gateway.host)
            config.gateway.port = gw.get("port", config.gateway.port)

        if "agent" in raw:
            ag = raw["agent"]
            config.agent.id = ag.get("id", config.agent.id)
            config.agent.name = ag.get("name", config.agent.name)
            config.agent.workspace = ag.get("workspace", config.agent.workspace)
            config.agent.max_loop_iterations = ag.get("max_loop_iterations", config.agent.max_loop_iterations)

            if "model" in ag:
                m = ag["model"]
                config.agent.model.provider = m.get("provider", config.agent.model.provider)
                config.agent.model.model_id = m.get("model_id", config.agent.model.model_id)
                config.agent.model.base_url = m.get("base_url", config.agent.model.base_url)
                config.agent.model.max_tokens = m.get("max_tokens", config.agent.model.max_tokens)

            if "tools" in ag:
                t = ag["tools"]
                config.agent.tools.allow = t.get("allow", config.agent.tools.allow)
                config.agent.tools.require_approval = t.get("require_approval", config.agent.tools.require_approval)

            if "deep_work" in ag:
                dw = ag["deep_work"]
                config.agent.deep_work.max_iterations = dw.get("max_iterations", config.agent.deep_work.max_iterations)
                config.agent.deep_work.token_budget = dw.get("token_budget", config.agent.deep_work.token_budget)
                config.agent.deep_work.warning_threshold = dw.get("warning_threshold", config.agent.deep_work.warning_threshold)

        if "sessions" in raw:
            s = raw["sessions"]
            config.sessions.directory = s.get("directory", config.sessions.directory)
            config.sessions.max_history_tokens = s.get("max_history_tokens", config.sessions.max_history_tokens)

        if "lmstudio" in raw:
            lm = raw["lmstudio"]
            config.lmstudio.base_url = lm.get("base_url", config.lmstudio.base_url)
            config.lmstudio.api_key = lm.get("api_key", config.lmstudio.api_key)

        if "reflection" in raw:
            r = raw["reflection"]
            config.reflection.enabled = r.get("enabled", config.reflection.enabled)
            config.reflection.model_id = r.get("model_id", config.reflection.model_id)
            config.reflection.max_tokens = r.get("max_tokens", config.reflection.max_tokens)
            config.reflection.max_sessions_per_startup = r.get("max_sessions_per_startup", config.reflection.max_sessions_per_startup)

        if "memory" in raw:
            mem = raw["memory"]
            config.memory.enabled = mem.get("enabled", config.memory.enabled)
            config.memory.embedding_base_url = mem.get("embedding_base_url", config.memory.embedding_base_url)
            config.memory.embedding_model = mem.get("embedding_model", config.memory.embedding_model)
            config.memory.top_k = mem.get("top_k", config.memory.top_k)

    config.agent.workspace = str(Path(config.agent.workspace).resolve())
    Path(config.sessions.directory).mkdir(parents=True, exist_ok=True)
    Path(config.agent.workspace).mkdir(parents=True, exist_ok=True)

    return config
