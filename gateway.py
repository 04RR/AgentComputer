"""agent_computer Gateway — single-process entry point.

Handles:
- WebSocket connections for real-time streaming
- HTTP REST API for simple request/response
- Session management
- Static file serving for the web UI

Uses OpenRouter as the model provider, giving access to hundreds of
models (Claude, GPT, Gemini, Llama, DeepSeek, etc.) through one API.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import uvicorn

from config import load_config
from agent import AgentRuntime, subscribe_activity, unsubscribe_activity, get_recent_activity
from session import SessionManager
from tool_registry import ToolRegistry
from tools import register_builtin_tools, register_task_tool, register_memory_search_tool
from cron import CronScheduler
from reflection import ReflectionEngine
from skill_loader import load_skills
from memory_search import MemorySearch

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_computer.gateway")

# ─── Bootstrap ───
config = load_config("config.json")
logger.info(f"Loaded config: agent={config.agent.name}, model={config.agent.model.model_id}")

# Tool registry
tool_registry = ToolRegistry()
register_builtin_tools(tool_registry, config.agent.workspace)
register_task_tool(tool_registry)
from tools.web_scrapling import register_scrapling_tools
register_scrapling_tools(tool_registry, config.agent.workspace)
tool_registry.set_approval_requirements(config.agent.tools.require_approval)

# Load existing skills into registry
skills_dir = Path(config.agent.workspace) / "memory" / "skills"
skill_names = load_skills(skills_dir, tool_registry)
if skill_names:
    config.agent.tools.allow.extend(skill_names)
    logger.info(f"Loaded {len(skill_names)} skills: {', '.join(skill_names)}")

logger.info(f"Registered {len(tool_registry.list_tools())} tools")

# Memory search
memory_search = None
if config.memory.enabled:
    memory_search = MemorySearch(
        workspace=config.agent.workspace,
        embedding_base_url=config.memory.embedding_base_url,
        embedding_model=config.memory.embedding_model,
        top_k=config.memory.top_k,
    )
    logger.info(f"Memory search initialized (model: {config.memory.embedding_model})")

# Register memory_search tool (works even if memory_search is None — returns helpful message)
register_memory_search_tool(tool_registry, memory_search, config.agent.workspace)

# Session manager
session_mgr = SessionManager(config.sessions.directory)

# Agent runtime
agent = AgentRuntime(config, tool_registry, memory_search=memory_search)

# Cron scheduler
scheduler = CronScheduler(config.agent.workspace, agent, session_mgr)
scheduler.load_jobs()

# ─── FastAPI App ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    scheduler.start()
    if memory_search:
        asyncio.create_task(memory_search.async_index_all())
    if config.reflection.enabled:
        asyncio.create_task(_run_reflection())
    yield
    await scheduler.stop()

app = FastAPI(title="agent_computer Gateway", version="0.1.0", lifespan=lifespan)


async def _run_reflection():
    """Background task: process unprocessed sessions to extract memory."""
    await asyncio.sleep(2)  # Let server finish starting

    try:
        model_id = config.reflection.model_id or config.agent.model.model_id
        engine = ReflectionEngine(
            workspace=config.agent.workspace,
            client=agent.client,
            model_id=model_id,
            max_tokens=config.reflection.max_tokens,
            provider=config.agent.model.provider,
            memory_search=memory_search,
        )

        # Scan for all session JSONL files
        sessions_dir = Path(config.sessions.directory)
        all_ids = [f.stem for f in sessions_dir.glob("*.jsonl") if f.stat().st_size > 0]
        unprocessed = engine.get_unprocessed(all_ids)

        if not unprocessed:
            logger.info("Auto-reflection: no unprocessed sessions")
            return

        batch = unprocessed[:config.reflection.max_sessions_per_startup]
        logger.info(f"Reflecting on {len(batch)} unprocessed sessions")

        for session_id in batch:
            try:
                messages = _load_session_messages(sessions_dir / f"{session_id}.jsonl")
                await engine.process_session(session_id, messages)
            except Exception as e:
                logger.error(f"Reflection error for session {session_id}: {e}")
                engine._update_index(session_id, {
                    "session_id": session_id,
                    "processed_at": time.time(),
                    "status": "error",
                    "error": str(e),
                })

        # Reload skills in case new ones were extracted
        new_skills = load_skills(skills_dir, tool_registry)
        if new_skills:
            config.agent.tools.allow.extend(new_skills)
            logger.info(f"New skills loaded after reflection: {', '.join(new_skills)}")

        logger.info("Auto-reflection complete")

    except Exception as e:
        logger.error(f"Auto-reflection failed: {e}")


def _load_session_messages(jsonl_path: Path) -> list[dict]:
    """Load raw message dicts from a JSONL session file."""
    messages = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return messages


# ─── WebSocket endpoint (primary interface) ───

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket connection for real-time agent interaction.

    Client sends: {"type": "message", "content": "..."}
    Server sends: AgentEvent objects as JSON
    """
    await websocket.accept()
    session = session_mgr.get_or_create(session_id)
    logger.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "message":
                content = data.get("content", "").strip()
                if not content:
                    continue

                # Always use session.mode as source of truth (set via set_mode message).
                # Per-message mode field is ignored for WebSocket — prevents stale
                # JS state from overriding the session mode.
                effective_mode = session.mode
                logger.info(f"Processing message in session={session_id} mode={effective_mode}")

                # Serial execution — acquire session lock
                async with session.lock:
                    async for event in agent.run(session, content, mode=effective_mode):
                        await websocket.send_json({
                            "type": event.type,
                            **event.data,
                        })

            elif data.get("type") == "set_mode":
                new_mode = data.get("mode", "bounded")
                if new_mode in ("bounded", "deep_work"):
                    session.mode = new_mode
                    await websocket.send_json({
                        "type": "mode_changed",
                        "mode": new_mode,
                    })
                    logger.info(f"Session {session_id} mode set to {new_mode}")
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Invalid mode: {new_mode}",
                    })

            elif data.get("type") == "clear":
                session.clear()
                await websocket.send_json({"type": "cleared"})

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── HTTP REST API ───

@app.post("/api/chat/{session_id}")
async def chat(session_id: str, body: dict):
    """Simple HTTP chat endpoint. Blocks until the agent loop completes.

    POST body: {"message": "your message here", "mode": "bounded"|"deep_work"}
    Response: {"response": "...", "session_id": "...", "mode": "..."}
    """
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(400, "Empty message")

    chat_mode = body.get("mode")
    session = session_mgr.get_or_create(session_id)

    async with session.lock:
        response_text = await agent.run_simple(session, message, mode=chat_mode)

    return JSONResponse({
        "response": response_text,
        "session_id": session_id,
        "mode": chat_mode or session.mode,
    })


@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse({"sessions": session_mgr.list_sessions()})


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_mgr.delete_session(session_id):
        return JSONResponse({"status": "deleted"})
    raise HTTPException(404, "Session not found")


@app.get("/api/sessions/{session_id}/history")
async def session_history(session_id: str):
    """Get full message history for a session (for loading into UI)."""
    session = session_mgr.get_or_create(session_id)
    return JSONResponse({
        "session_id": session_id,
        "messages": session.get_history(),
        "token_usage": session.get_token_usage(),
    })


@app.get("/api/sessions/{session_id}/tasks")
async def session_tasks(session_id: str):
    """Get task list for a session (deep work mode)."""
    session = session_mgr.get_or_create(session_id)
    return JSONResponse({
        "session_id": session_id,
        "mode": session.mode,
        "tasks": session.get_tasks(),
        "summary": session.task_store.summary(),
    })


@app.get("/api/usage")
async def usage(range: str = Query("all")):
    """Aggregated token usage across sessions.

    Query params:
      range: "today", "7d", "30d", "all" (default: "all")
    """
    since = None
    now = time.time()
    if range == "today":
        since = now - 86400
    elif range == "7d":
        since = now - 7 * 86400
    elif range == "30d":
        since = now - 30 * 86400

    data = session_mgr.get_aggregate_usage(since=since)
    return JSONResponse(data)


@app.get("/api/activity")
async def recent_activity(limit: int = Query(50)):
    """Recent tool call activity across all sessions."""
    return JSONResponse({"events": get_recent_activity(limit)})


@app.websocket("/ws/activity")
async def activity_ws(websocket: WebSocket):
    """WebSocket for live agent activity across all sessions."""
    await websocket.accept()
    queue = subscribe_activity()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe_activity(queue)


@app.get("/api/status")
async def status():
    return JSONResponse({
        "status": "ok",
        "agent": config.agent.name,
        "model": config.agent.model.model_id,
        "provider": config.agent.model.provider,
        "base_url": config.agent.model.base_url,
        "tools": [t.name for t in tool_registry.list_tools()],
        "sessions": len(session_mgr.list_sessions()),
        "cron_jobs": len(scheduler.jobs),
        "deep_work": {
            "max_iterations": config.agent.deep_work.max_iterations,
            "token_budget": config.agent.deep_work.token_budget,
            "warning_threshold": config.agent.deep_work.warning_threshold,
        },
    })


# ─── Models API ───

# Curated list of popular OpenRouter models
OPENROUTER_MODELS = [
    {"id": "anthropic/claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
    {"id": "anthropic/claude-opus-4-6", "name": "Claude Opus 4.6"},
    {"id": "anthropic/claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
    {"id": "openai/gpt-4o", "name": "GPT-4o"},
    {"id": "openai/gpt-4.1", "name": "GPT-4.1"},
    {"id": "openai/o3-mini", "name": "o3-mini"},
    {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro"},
    {"id": "google/gemini-2.5-flash-preview", "name": "Gemini 2.5 Flash"},
    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1"},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3"},
    {"id": "meta-llama/llama-4-maverick", "name": "Llama 4 Maverick"},
    {"id": "meta-llama/llama-4-scout", "name": "Llama 4 Scout"},
    {"id": "qwen/qwen3-235b-a22b", "name": "Qwen3 235B"},
]


@app.get("/api/models")
async def list_models():
    """List available models from all providers."""
    result = {
        "current": {
            "provider": config.agent.model.provider,
            "model_id": config.agent.model.model_id,
        },
        "providers": {
            "openrouter": {
                "name": "OpenRouter",
                "models": OPENROUTER_MODELS,
            },
            "lmstudio": {
                "name": "LM Studio",
                "models": [],
                "reachable": False,
            },
        },
    }

    # Fetch LM Studio models
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{config.lmstudio.base_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                result["providers"]["lmstudio"]["models"] = [
                    {"id": m["id"], "name": m.get("id", "")}
                    for m in models
                ]
                result["providers"]["lmstudio"]["reachable"] = True
    except Exception:
        pass  # LM Studio not reachable, return empty list

    return JSONResponse(result)


@app.post("/api/models/select")
async def select_model(body: dict):
    """Switch the active model.

    POST body: {"provider": "openrouter"|"lmstudio", "model_id": "..."}
    """
    provider = body.get("provider", "").strip()
    model_id = body.get("model_id", "").strip()
    if not provider or not model_id:
        raise HTTPException(400, "provider and model_id required")

    if provider == "lmstudio":
        base_url = config.lmstudio.base_url
        api_key = config.lmstudio.api_key
    elif provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = agent._openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    else:
        raise HTTPException(400, f"Unknown provider: {provider}")

    agent.set_model(provider, model_id, base_url, api_key)

    logger.info(f"Model switched to {model_id} ({provider})")
    return JSONResponse({
        "status": "ok",
        "provider": provider,
        "model_id": model_id,
        "base_url": base_url,
    })


# ─── Cron API ───

@app.get("/api/cron")
async def cron_status():
    """List all cron jobs and their status."""
    return JSONResponse({"jobs": scheduler.get_status()})


@app.post("/api/cron/{job_id}/run")
async def cron_run_now(job_id: str):
    """Manually trigger a cron job immediately."""
    job = scheduler.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Cron job '{job_id}' not found")
    await scheduler._run_job(job)
    return JSONResponse({"status": "ok", "job": job_id, "run_count": job.run_count})


@app.post("/api/cron/{job_id}/toggle")
async def cron_toggle(job_id: str):
    """Enable or disable a cron job."""
    job = scheduler.jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Cron job '{job_id}' not found")
    job.enabled = not job.enabled
    state = "enabled" if job.enabled else "disabled"
    logger.info(f"Cron job {job_id} {state}")
    return JSONResponse({"status": state, "job": job_id})


@app.post("/api/cron/reload")
async def cron_reload():
    """Reload cron.json from the workspace."""
    count = scheduler.load_jobs()
    return JSONResponse({"status": "reloaded", "jobs_loaded": count})


# ─── Web UI ───

@app.get("/")
async def root():
    ui_path = Path(__file__).parent / "web" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse(
        "<h1>agent_computer Gateway</h1>"
        "<p>Web UI not found. Use WebSocket at /ws/{session_id}</p>"
    )


# ─── Entry point ───

def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set!")
        logger.error("Get your key at https://openrouter.ai/keys")
        logger.error("Then: export OPENROUTER_API_KEY=sk-or-...")
        return

    # Patch the agent's client with the real key
    agent.client.api_key = api_key

    logger.info(f"Starting agent_computer Gateway on {config.gateway.host}:{config.gateway.port}")
    logger.info(f"Agent: {config.agent.name}")
    logger.info(f"Model: {config.agent.model.model_id} via {config.agent.model.base_url}")
    logger.info(f"Workspace: {config.agent.workspace}")
    logger.info(f"Web UI: http://localhost:{config.gateway.port}")
    logger.info(f"WebSocket: ws://localhost:{config.gateway.port}/ws/{{session_id}}")
    logger.info(f"Cron jobs: {len(scheduler.jobs)} loaded ({sum(1 for j in scheduler.jobs.values() if j.enabled)} enabled)")

    uvicorn.run(app, host=config.gateway.host, port=config.gateway.port, log_level="info")


if __name__ == "__main__":
    main()
