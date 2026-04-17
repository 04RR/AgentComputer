# agent_computer

A Python-based autonomous agent framework with tool use, long-term memory,
scheduled tasks, and auto-reflection. Supports local models via **LM Studio**
and cloud models via **OpenRouter**.

## Core Concepts

- **Gateway** — FastAPI process handling WebSocket + HTTP connections
- **Agent Runtime** — Agentic loop: context assembly → LLM → tool execution → reply
- **Sessions** — Stateful conversations persisted as JSONL, with serial execution
- **Tools** — Modular, registerable capabilities (shell, files, web scraping, web search, browser automation, tasks, memory search)
- **Workspace** — Identity files (SOUL.md, USER.md) and agent memory
- **Deep Work Mode** — Extended autonomous execution with task decomposition, auto-compaction, and token budgets
- **Reflection Engine** — Extracts knowledge, learnings, and reusable skills from completed sessions
- **Memory Search** — Hybrid vector + keyword search (SQLite FTS5 BM25 + vectorized cosine similarity via Reciprocal Rank Fusion)
- **Cron Scheduler** — Scheduled agent tasks that run on a timer or at startup
- **Skill Loader** — Auto-discovered Python skills registered as tools at runtime
- **Context Compactor** — Lightweight per-iteration tool_result truncation to keep context lean

## Quick Start

```bash
pip install -r requirements.txt

# Option A: Local models via LM Studio (default)
# Make sure LM Studio is running with an API server
python gateway.py

# Option B: Cloud models via OpenRouter
export OPENROUTER_API_KEY=sk-or-...
# Then change provider to "openrouter" in config.json
python gateway.py
```

Open `http://localhost:8000` for the web UI, or connect via WebSocket at
`ws://localhost:8000/ws/{session_id}`.

## Configuration

Edit `config.json` to configure the agent. All fields have sensible defaults.

```json
{
  "gateway": { "host": "0.0.0.0", "port": 8000 },
  "agent": {
    "name": "agent_computer Agent",
    "workspace": "./workspace",
    "model": {
      "provider": "lmstudio",
      "model_id": "your-model-id",
      "base_url": "http://your-lmstudio-host:1234/v1",
      "max_tokens": 8192
    },
    "tools": {
      "allow": ["shell", "read_file", "write_file", "list_directory",
                 "web_fetch", "web_fetch_js", "web_fetch_stealth",
                 "web_search", "manage_tasks", "memory_search"],
      "require_approval": ["shell"]
    },
    "max_loop_iterations": 20,
    "deep_work": {
      "max_iterations": 200,
      "token_budget": 1500000,
      "warning_threshold": 0.8
    }
  },
  "sessions": { "directory": "./sessions", "max_history_tokens": 100000 },
  "reflection": {
    "enabled": true,
    "model_id": "",
    "max_tokens": 4096,
    "max_sessions_per_startup": 10
  },
  "memory": {
    "enabled": true,
    "embedding_base_url": "http://your-lmstudio-host:1234/v1",
    "embedding_model": "text-embedding-nomic-embed-text-v1.5",
    "top_k": 5
  }
}
```

### Switching Models

Models can be changed in `config.json` or at runtime via the API:

```bash
# Runtime model switch
curl -X POST http://localhost:8000/api/models/select \
  -H "Content-Type: application/json" \
  -d '{"provider": "openrouter", "model_id": "anthropic/claude-sonnet-4-6"}'
```

**Providers:**
- `lmstudio` — Local models via LM Studio (default)
- `openrouter` — Cloud models (Claude, GPT, Gemini, Llama, DeepSeek, etc.)

## Project Structure

```
agent_computer/
├── gateway.py            # FastAPI gateway (entry point)
├── agent.py              # Agent runtime + agentic loop
├── session.py            # Session management, JSONL persistence, compaction
├── tool_registry.py      # Tool registration and execution (with per-call context passing)
├── context.py            # PromptContext dataclass + system prompt assembly
├── context_compactor.py  # Lightweight tool_result truncation for context management
├── config.py             # Configuration loader (Pydantic models) with defaults
├── config.json           # Runtime configuration
├── task_store.py         # Hierarchical task management for deep work mode
├── cron.py               # Cron scheduler (scheduled agent tasks)
├── reflection.py         # Auto-reflection engine (knowledge extraction)
├── memory_search.py      # Hybrid memory search (FTS5 + vectorized NumPy, SQLite)
├── skill_loader.py       # Dynamic skill loader from Python files
├── tools/
│   ├── __init__.py       # Built-in tools (shell, files, tasks, memory)
│   ├── web_scrapling.py  # Web fetching tools (3 tiers via Scrapling)
│   ├── web_search.py     # DuckDuckGo web search tool
│   └── pinchtab.py       # PinchTab browser automation tools (CDP proxy)
├── workspace/            # Agent identity + memory (auto-created)
│   ├── SOUL.md           # Agent personality, grounding rules, and constraints
│   ├── USER.md           # User context + preferences
│   ├── cron.json         # Scheduled job definitions
│   └── memory/           # Long-term memory
│       ├── knowledge.md  # Extracted knowledge from sessions
│       ├── learnings.md  # Extracted learnings (mistake → correction)
│       ├── index.json    # Reflection processing index
│       ├── memory.db     # SQLite DB (memories table + FTS5 index + embeddings)
│       └── skills/       # Auto-extracted reusable Python skills
├── sessions/             # JSONL session transcripts (auto-created)
├── web/
│   └── index.html        # Chat web UI (activity panels, tool events, task tracking)
├── start_gateway.sh      # Startup script
└── requirements.txt
```

## Operating Modes

### Bounded Mode (default)

Standard chat mode. The agent runs up to `max_loop_iterations` (default 20)
per message, executing tools as needed and returning a response.

### Deep Work Mode

Extended autonomous execution for complex multi-step tasks. Activated per-session
via WebSocket or per-request via the HTTP API.

Deep work has two phases:

1. **Planning** — The agent researches the request, explores files/APIs, and creates a task breakdown using `manage_tasks`. It presents a plan for user review.
2. **Execution** — After the user approves (or revises) the plan, the agent autonomously works through tasks one by one.

Features:
- **Task decomposition** — Agent creates a hierarchical task tree with parent tasks and subtasks
- **Token budget** — Configurable limit (default 1.5M tokens) with warning threshold
- **Auto-compaction** — When approaching the budget, conversation is archived to a markdown file and the context is trimmed, allowing up to 5 compaction cycles
- **Nudge system** — If the agent produces text without using tools while tasks remain, the system injects a nudge to keep it on track
- **Plan approval** — User can approve, revise, or provide feedback on the plan before execution begins

```bash
# HTTP API with deep work mode
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "Research and compile a report on...", "mode": "deep_work"}'
```

## Architecture Notes

### Prompt Context

System prompt assembly uses a `PromptContext` dataclass that holds all inputs
(workspace, agent name, mode, tool names, memories, etc.). The prompt is built
in two parts:

- **Static prefix** — Identity, SOUL.md, USER.md, tool list, memories, deep work instructions. Cached across iterations in deep work mode and only rebuilt when `session_summary` changes (every 10 iterations).
- **Dynamic suffix** — Task state, budget warnings, archived context references. Rebuilt every iteration.

This separation avoids redundant string construction during long deep work sessions.

### Context Compactor

Before each LLM call, `truncate_tool_results()` deep-copies the message history
and truncates large tool_result contents (>2000 chars) in older messages, leaving
the most recent 3 messages untouched. This is a lightweight per-iteration optimization
that complements the heavier `Session.compact()` archival path.

### Session State Machine

Session mode (`bounded` / `deep_work`) and deep work phase (`None` / `planning` / `executing`)
are managed through methods on the `Session` class:

- `set_mode(mode)` — Validates and switches mode, resets phase
- `begin_deep_work_if_needed()` — Transitions `None → planning` on first deep work run
- `approve_plan()` — Transitions `planning → executing` with precondition checks

### Tool Context Passing

Tool execution uses explicit context passing instead of global state. When the
agent loop runs, it builds a `tool_context` dict containing the session's task
store, mode, and session ID. The `ToolRegistry.execute()` method accepts an
optional `context` parameter and forwards it to any tool handler that declares a
`_context` keyword argument (detected once at registration via `inspect.signature`
and cached). This eliminates shared mutable globals and makes concurrent sessions
safe.

### Memory Search Internals

`MemorySearch` uses two complementary search strategies merged via Reciprocal
Rank Fusion (RRF):

- **Keyword search** — SQLite FTS5 virtual table (`memories_fts`) kept in sync
  with the `memories` table via INSERT/DELETE/UPDATE triggers. Queries use the
  built-in BM25 ranking (`rank` column). A one-time backfill populates FTS from
  pre-existing rows on first init.

- **Vector search** — All embeddings are loaded into a single pre-normalized
  `(n, dim)` NumPy matrix. Cosine similarities are computed in one matrix-vector
  multiply, with top-k selection via `np.argpartition` (O(n) instead of
  O(n log n) full sort). The matrix is rebuilt lazily when `_matrix_dirty` is set
  (after `index_text()`).

The RRF merge step fetches only the final top-k rows from SQLite by ID, avoiding
loading the entire table into Python.

## Tools

| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands (runs in workspace directory, sandboxed) |
| `read_file` | Read file contents (relative to workspace, with path traversal protection) |
| `write_file` | Write/append to files (auto-creates parent directories) |
| `list_directory` | List directory tree with depth control |
| `web_fetch` | Fast HTTP fetch with browser TLS impersonation (static HTML only, no JS) |
| `web_fetch_js` | Full browser JS rendering for dynamic sites (GitHub repos, Reddit, etc.) |
| `web_fetch_stealth` | Stealth browser with Cloudflare challenge bypass (last resort) |
| `web_search` | DuckDuckGo search — returns titles, URLs, and snippets (no API key needed) |
| `manage_tasks` | Create/update/complete/delete tasks (deep work mode only) |
| `memory_search` | Search long-term memory via hybrid vector + keyword |

**Web fetching tiers:**
- `web_fetch` — Fastest. Use for APIs, static pages, documentation. Does NOT execute JavaScript.
- `web_fetch_js` — Slower (5-10s). Use for JS-rendered sites (GitHub star counts, Reddit, pricing pages).
- `web_fetch_stealth` — Slowest (30-120s). Use only when `web_fetch_js` hits a Cloudflare challenge.
- `web_search` — Use instead of `google.com/search` (which is blocked). Returns structured search results via DuckDuckGo.

**Browser automation (PinchTab):** When enabled, provides `browser_navigate`, `browser_snapshot`,
`browser_click`, `browser_type`, `browser_fill`, `browser_press`, `browser_scroll`,
`browser_text`, `browser_screenshot`, and `browser_tabs` tools via a local CDP proxy.

Auto-extracted skills from `workspace/memory/skills/` are also registered as tools at startup.

## Cron Scheduler

Define scheduled jobs in `workspace/cron.json`:

```json
[
  {
    "id": "daily-check",
    "name": "Daily status check",
    "schedule": "daily 09:00",
    "prompt": "Check system status and summarize any issues.",
    "session_id": "cron-daily-check",
    "enabled": true
  }
]
```

**Schedule formats:**
- `every 30m` / `every 2h` / `every 1d` — interval-based
- `daily 09:00` — daily at a specific UTC time
- `hourly :15` — every hour at minute 15
- `startup` — run once when the gateway starts

## Reflection Engine

On startup, the reflection engine scans completed sessions and uses the LLM to extract:

- **Knowledge** — Facts, API endpoints, configurations that were confirmed to work
- **Learnings** — Mistake → correction patterns to avoid repeating errors
- **Skills** — Reusable async Python functions, saved as `.py` files and auto-loaded as tools

Extracted items are stored in `workspace/memory/` and indexed for semantic search.

## Web UI

The web UI (`web/index.html`) is a single-file application featuring:

- **Chat interface** with markdown rendering
- **Activity panels** — Collapsible per-iteration logs showing sub-phase events (memory search, LLM calls, tool execution) with elapsed timers
- **Tool event nesting** — Tool calls and results render inside activity panels with expandable detail views
- **Deep work controls** — Mode toggle, plan approval/revision cards, task panel with progress tracking
- **Session management** — Sidebar with session list, create/switch/delete
- **Model switching** — Search and select models from LM Studio or OpenRouter

## HTTP API

```bash
# Chat (blocking)
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "List the files in my workspace"}'

# Chat with deep work mode
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "...", "mode": "deep_work"}'

# Status
curl http://localhost:8000/api/status

# List sessions
curl http://localhost:8000/api/sessions

# Session history
curl http://localhost:8000/api/sessions/my-session/history

# Session tasks (deep work)
curl http://localhost:8000/api/sessions/my-session/tasks

# Delete session
curl -X DELETE http://localhost:8000/api/sessions/my-session

# Token usage (range: today, 7d, 30d, all)
curl http://localhost:8000/api/usage?range=7d

# Recent activity
curl http://localhost:8000/api/activity?limit=50

# List available models
curl http://localhost:8000/api/models

# Search OpenRouter models
curl http://localhost:8000/api/models/search?q=claude

# Switch model at runtime
curl -X POST http://localhost:8000/api/models/select \
  -H "Content-Type: application/json" \
  -d '{"provider": "lmstudio", "model_id": "your-model-id"}'

# Cron job status
curl http://localhost:8000/api/cron

# Trigger a cron job manually
curl -X POST http://localhost:8000/api/cron/daily-check/run

# Toggle a cron job on/off
curl -X POST http://localhost:8000/api/cron/daily-check/toggle

# Reload cron.json
curl -X POST http://localhost:8000/api/cron/reload
```

## WebSocket Protocol

Connect to `ws://localhost:8000/ws/{session_id}` and send/receive JSON.

**Client → Server:**
```json
{"type": "message", "content": "your message"}
{"type": "set_mode", "mode": "deep_work"}
{"type": "approve_plan", "feedback": "optional notes"}
{"type": "revise_plan", "feedback": "change X to Y"}
{"type": "clear"}
{"type": "ping"}
```

**Server → Client:**
```json
{"type": "thinking", "iteration": 0, "phase": "memory_search"}
{"type": "thinking", "iteration": 1}
{"type": "thinking", "iteration": 1, "phase": "llm_call", "model": "...", "prompt_tokens_estimate": 5000}
{"type": "thinking", "iteration": 1, "phase": "llm_response", "response_tokens": 500}
{"type": "thinking", "iteration": 2, "max_iterations": 200, "tokens_used": 5000, "token_budget": 1500000, "task_summary": "..."}
{"type": "tool_call", "tool": "shell", "input": {"command": "ls"}, "tool_call_id": "..."}
{"type": "tool_result", "tool": "shell", "result_preview": "...", "duration_ms": 150, "success": true}
{"type": "text", "text": "Here are your files..."}
{"type": "task_update", "tasks": [...], "summary": "..."}
{"type": "plan_ready", "text": "...", "tasks": [...], "summary": "...", "iterations": 5, "usage": {...}}
{"type": "mode_changed", "mode": "deep_work"}
{"type": "done", "text": "...", "iterations": 2, "model": "...", "usage": {...}, "mode": "bounded"}
{"type": "error", "message": "..."}
```

**Activity WebSocket** at `ws://localhost:8000/ws/activity` streams tool calls and results across all sessions in real time.

## Memory Search CLI

Test memory search independently:

```bash
# Default query
python memory_search.py

# Custom query
python memory_search.py "What API patterns have been discovered?"
```
