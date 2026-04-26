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

import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import uvicorn

from config import load_config
from agent import AgentRuntime, build_policy_callback, subscribe_activity, unsubscribe_activity, get_recent_activity
from session import Session, SessionManager
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


def _get_openrouter_key() -> str:
    """Resolve OpenRouter API key: agent cache → env var → config file."""
    return (
        agent._openrouter_api_key
        or os.environ.get("OPENROUTER_API_KEY", "")
        or config.openrouter_api_key
    )

# Tool registry
tool_registry = ToolRegistry()
allowed_tools = config.agent.tools.allow
register_builtin_tools(tool_registry, config.agent.workspace, allowed=allowed_tools, tools_config=config.agent.tools)
register_task_tool(tool_registry, allowed=allowed_tools)
from tools.web_scrapling import register_scrapling_tools
register_scrapling_tools(tool_registry, config.agent.workspace, allowed=allowed_tools)
from tools.web_search import register_web_search_tool
register_web_search_tool(tool_registry, allowed=allowed_tools)
if config.pinchtab.enabled:
    from tools.pinchtab import register_pinchtab_tools
    register_pinchtab_tools(tool_registry, config.agent.workspace, allowed=allowed_tools, pinchtab_config=config.pinchtab)
# Verification-mode tools (Phase 1): registered unconditionally so the
# /api/verify/raw endpoint can call them, but absent from agent.tools.allow
# so the agent loop can't see them yet. Phase 2 will flip them on for agent use.
from tools.verification import register_verification_tools
register_verification_tools(tool_registry, config.agent.workspace, config.verification)
if config.verification.tineye_stub_mode != "off":
    logger.warning(
        f"tineye_stub_mode={config.verification.tineye_stub_mode!r} "
        f"— reverse_image_search returns canned fixture data, NOT real "
        f"TinEye results. Set to 'off' for production use."
    )
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
register_memory_search_tool(tool_registry, memory_search, config.agent.workspace, allowed=allowed_tools)

# Session manager
session_mgr = SessionManager(config.sessions.directory)

# Agent runtime
agent = AgentRuntime(config, tool_registry, memory_search=memory_search)

# Cron scheduler — cron jobs are headless, so they inherit the non-interactive
# approval policy. Individual jobs can still override via cron.json.
scheduler = CronScheduler(
    config.agent.workspace,
    agent,
    session_mgr,
    default_approval_policy=config.agent.tools.non_interactive_approval_policy,
)
scheduler.load_jobs()

# Signals that the server has finished initializing
server_ready = asyncio.Event()

# ─── FastAPI App ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    scheduler.start()
    if memory_search:
        asyncio.create_task(memory_search.async_index_all())
    if config.reflection.enabled:
        asyncio.create_task(_run_reflection())
    if config.pinchtab.enabled:
        asyncio.create_task(_check_pinchtab())
    server_ready.set()
    # Pre-warm the OpenRouter model cache if an API key is available
    or_api_key = _get_openrouter_key()
    if config.agent.model.provider == "openrouter" and or_api_key:
        asyncio.create_task(get_openrouter_models(or_api_key))
    yield
    await scheduler.stop()

app = FastAPI(title="agent_computer Gateway", version="0.1.0", lifespan=lifespan)


async def _run_reflection():
    """Background task: process unprocessed sessions to extract memory."""
    await server_ready.wait()  # Wait for server initialization to complete

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


async def _check_pinchtab():
    """Check PinchTab connectivity at startup."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            headers = {}
            if config.pinchtab.token:
                headers["Authorization"] = f"Bearer {config.pinchtab.token}"
            resp = await client.get(f"{config.pinchtab.base_url}/health", headers=headers)
            if resp.status_code == 200:
                logger.info(f"PinchTab connected: {config.pinchtab.base_url}")
            else:
                logger.warning(f"PinchTab returned {resp.status_code} — browser tools may not work")
    except Exception as e:
        logger.warning(f"PinchTab not reachable at {config.pinchtab.base_url}: {e}")


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

    # Approval plumbing: map tool_call_id → pending Future. The agent loop is
    # awaiting inside asyncio.gather; we park each request here and resolve when
    # the client sends an approval_response message.
    pending_approvals: dict[str, asyncio.Future[bool]] = {}

    async def _safe_send(payload: dict) -> bool:
        """Send a JSON payload, swallowing disconnects. Returns True on success."""
        try:
            await websocket.send_json(payload)
            return True
        except Exception as e:
            logger.warning(f"WS send failed for session={session_id}: {e}")
            return False

    async def ws_approval_callback(tool_call_id: str, tool_name: str, tool_args: dict) -> bool:
        """Emit an approval_request, await the client's response, emit resolution.

        Returns True only on explicit user approval. Timeouts, disconnects, and
        denies all return False so the agent treats them uniformly as refusals.
        """
        sent = await _safe_send({
            "type": "approval_request",
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "input": tool_args,
        })
        if not sent:
            # Connection is gone; no point waiting for a response that can't arrive.
            return False

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        pending_approvals[tool_call_id] = fut
        reason = "deny"
        try:
            approved = await asyncio.wait_for(fut, timeout=300.0)
            reason = "allow" if approved else "deny"
            return approved
        except asyncio.TimeoutError:
            logger.warning(
                f"Approval timeout for {tool_name} (call {tool_call_id}, session={session_id})"
            )
            reason = "timeout"
            return False
        finally:
            pending_approvals.pop(tool_call_id, None)
            await _safe_send({
                "type": "approval_resolved",
                "tool_call_id": tool_call_id,
                "tool": tool_name,
                "approved": reason == "allow",
                "reason": reason,
            })

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
                    async for event in agent.run(
                        session, content,
                        mode=effective_mode,
                        approval_callback=ws_approval_callback,
                    ):
                        await websocket.send_json({
                            "type": event.type,
                            **event.data,
                        })

            elif data.get("type") == "approval_response":
                tc_id = data.get("tool_call_id")
                approved = bool(data.get("approved", False))
                fut = pending_approvals.get(tc_id)
                if fut and not fut.done():
                    fut.set_result(approved)
                else:
                    logger.warning(
                        f"approval_response for unknown or resolved tool_call_id: {tc_id}"
                    )

            elif data.get("type") == "approve_plan":
                # User approved the plan — transition to execution phase
                try:
                    session.approve_plan()
                except ValueError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue
                feedback = data.get("feedback", "")
                approval_msg = "[PLAN APPROVED] Execute the plan now."
                if feedback:
                    approval_msg += f"\n\nAdditional notes from user: {feedback}"
                logger.info(f"Plan approved for session={session_id}, transitioning to execution phase")
                async with session.lock:
                    async for event in agent.run(
                        session, approval_msg,
                        mode="deep_work",
                        approval_callback=ws_approval_callback,
                    ):
                        await websocket.send_json({
                            "type": event.type,
                            **event.data,
                        })

            elif data.get("type") == "revise_plan":
                # User wants to refine the plan — stay in planning phase
                feedback = data.get("feedback", "").strip()
                if not feedback:
                    continue
                logger.info(f"Plan revision requested for session={session_id}")
                async with session.lock:
                    async for event in agent.run(
                        session, feedback,
                        mode="deep_work",
                        approval_callback=ws_approval_callback,
                    ):
                        await websocket.send_json({
                            "type": event.type,
                            **event.data,
                        })

            elif data.get("type") == "set_mode":
                new_mode = data.get("mode", "bounded")
                if new_mode in Session.VALID_MODES:
                    session.set_mode(new_mode)
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
    finally:
        # Resolve any outstanding approval Futures so awaiting tasks unblock
        # immediately instead of waiting for the 5-minute timeout.
        for fut in pending_approvals.values():
            if not fut.done():
                fut.set_result(False)
        pending_approvals.clear()


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

    http_approval = build_policy_callback(
        config.agent.tools.non_interactive_approval_policy,
        f"http:{session_id}",
    )

    async with session.lock:
        response_text = await agent.run_simple(
            session, message,
            mode=chat_mode,
            approval_callback=http_approval,
        )

    return JSONResponse({
        "response": response_text,
        "session_id": session_id,
        "mode": chat_mode or session.mode,
    })


# ─── Verification mode (Phase 1) ───

@app.post("/api/verify/raw")
async def verify_raw(image: UploadFile = File(...), caption: str = Form(...)):
    """Run all three verification tools on an uploaded image + caption.

    Phase 1 endpoint: bypasses the agent loop entirely and returns raw tool
    outputs in parallel. Phase 2 will add a separate /api/verify endpoint
    that runs the agent.

    No file size cap, no content-type validation in Phase 1.
    TODO Phase 4: cap upload size, validate that the file is actually an image.
    """
    t0 = time.monotonic()

    upload_dir = Path(config.agent.workspace) / ".verify_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename or "").suffix or ".jpg"
    image_id = f"{uuid.uuid4().hex}{ext}"
    image_path = upload_dir / image_id

    contents = await image.read()
    image_path.write_bytes(contents)

    async def _run(name: str, params: dict):
        result_str = await tool_registry.execute(name, params)
        return name, result_str

    pairs = await asyncio.gather(
        _run("reverse_image_search", {
            "image_path": str(image_path),
            "caption": caption,
        }),
        _run("extract_image_metadata", {"image_path": str(image_path)}),
        _run("fact_check_lookup", {"query": caption}),
    )

    elapsed_ms = round((time.monotonic() - t0) * 1000)

    parsed: dict[str, dict] = {}
    errors: list[dict] = []
    for name, result_str in pairs:
        try:
            result_dict = json.loads(result_str)
        except json.JSONDecodeError:
            result_dict = {"error": "tool returned non-JSON", "raw": result_str[:500]}
        parsed[name] = result_dict
        if isinstance(result_dict, dict) and "error" in result_dict:
            errors.append({"tool": name, "error": result_dict["error"]})

    # Spec uses the key "image_metadata" for the metadata tool (not the
    # registered name "extract_image_metadata"). Map at the response boundary.
    return JSONResponse({
        "image_path": str(image_path),
        "caption": caption,
        "results": {
            "reverse_image_search": parsed.get("reverse_image_search"),
            "image_metadata": parsed.get("extract_image_metadata"),
            "fact_check_lookup": parsed.get("fact_check_lookup"),
        },
        "errors": errors,
        "elapsed_ms": elapsed_ms,
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
        "pinchtab": {
            "enabled": config.pinchtab.enabled,
            "base_url": config.pinchtab.base_url,
        },
    })


# ─── Models API ───

# Fallback used only when the OpenRouter API fetch fails and the cache is empty
_OPENROUTER_FALLBACK_MODELS = [
    {"id": "anthropic/claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
    {"id": "anthropic/claude-opus-4-6", "name": "Claude Opus 4.6"},
    {"id": "openai/gpt-4o", "name": "GPT-4o"},
    {"id": "google/gemini-2.5-pro-preview", "name": "Gemini 2.5 Pro"},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3"},
]

# Dynamic cache for OpenRouter models
_openrouter_models_cache: list[dict] = []
_openrouter_models_fetched_at: float = 0.0


async def fetch_openrouter_models(api_key: str) -> list[dict]:
    """Fetch the full model list from the OpenRouter API. Returns [] on failure."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m["id"], "name": m.get("name") or m["id"]}
                for m in data.get("data", [])
            ]
            models.sort(key=lambda m: m["name"])
            logger.info(f"Fetched {len(models)} models from OpenRouter API")
            return models
    except Exception as e:
        logger.warning(f"Failed to fetch OpenRouter models: {e}")
        return []


async def get_openrouter_models(api_key: str, max_age_seconds: int = 3600) -> list[dict]:
    """Return cached OpenRouter models, refreshing if stale or empty."""
    global _openrouter_models_cache, _openrouter_models_fetched_at
    if _openrouter_models_cache and time.time() - _openrouter_models_fetched_at < max_age_seconds:
        return _openrouter_models_cache

    models = await fetch_openrouter_models(api_key)
    if models:
        _openrouter_models_cache = models
        _openrouter_models_fetched_at = time.time()
        return models

    # Fetch failed — return existing cache if available, otherwise fallback
    if _openrouter_models_cache:
        return _openrouter_models_cache
    return list(_OPENROUTER_FALLBACK_MODELS)


_PINNED_MODELS = [
    {"id": "z-ai/glm-5", "name": "GLM-5", "provider": "openrouter"},
    {"id": "qwen/qwen3.6-35b-a3b", "name": "Qwen 3.6 35B A3B", "provider": "lmstudio"},
]


@app.get("/api/models")
async def list_models():
    """List available models from all providers."""
    api_key = _get_openrouter_key()

    # Pre-warm OpenRouter cache in the background (don't block the response)
    if api_key and not _openrouter_models_cache:
        asyncio.create_task(get_openrouter_models(api_key))

    result = {
        "current": {
            "provider": config.agent.model.provider,
            "model_id": config.agent.model.model_id,
        },
        "pinned": _PINNED_MODELS,
        "providers": {
            "lmstudio": {
                "name": "LM Studio",
                "models": [],
                "reachable": False,
            },
            "openrouter": {
                "name": "OpenRouter",
                "searchable": bool(api_key),
                "note": "" if api_key else "OPENROUTER_API_KEY not set",
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


@app.get("/api/models/search")
async def search_models(q: str = Query("", min_length=1)):
    """Search OpenRouter models by name/id. Returns top 20 matches."""
    api_key = _get_openrouter_key()
    if not api_key:
        return JSONResponse({"models": [], "note": "OPENROUTER_API_KEY not set"})

    all_models = await get_openrouter_models(api_key)
    query = q.lower()
    matches = [m for m in all_models if query in m["id"].lower() or query in m["name"].lower()]
    return JSONResponse({"models": matches[:20], "query": q})


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
        api_key = _get_openrouter_key()
        if not api_key:
            logger.warning("OpenRouter selected but no API key found (agent._openrouter_api_key and OPENROUTER_API_KEY env var are both empty)")
    else:
        raise HTTPException(400, f"Unknown provider: {provider}")

    agent.set_model(provider, model_id, base_url, api_key)

    # Log if the selected model is not in the cached list
    cached_ids = {m["id"] for m in _openrouter_models_cache}
    if provider == "openrouter" and model_id not in cached_ids:
        logger.info(f"Model {model_id} not in cached model list (custom or stale cache)")

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
    provider = config.agent.model.provider
    api_key = os.environ.get("OPENROUTER_API_KEY") or config.openrouter_api_key

    # Always store the OpenRouter key on the agent so model switching works
    if api_key:
        agent._openrouter_api_key = api_key

    if provider == "lmstudio":
        # LM Studio uses its own base_url and api_key from config
        agent.client.api_key = config.lmstudio.api_key
        logger.info("Using LM Studio as provider (OPENROUTER_API_KEY is optional)")
    elif api_key:
        agent.client.api_key = api_key
    else:
        logger.error("No OpenRouter API key found (checked OPENROUTER_API_KEY env var and config.json openrouter_api_key)")
        logger.error("Get your key at https://openrouter.ai/keys")
        logger.error("Then set it in config.json: \"openrouter_api_key\": \"sk-or-...\"")
        logger.error("Or via env var: set OPENROUTER_API_KEY=sk-or-...")
        logger.error("Or switch to LM Studio by setting provider to 'lmstudio' in config.json")
        return

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
