"""Agent runtime for agent_computer (OpenRouter via OpenAI SDK).

Implements the agentic loop:
  intake → context assembly → model inference → tool execution → reply

Uses the OpenAI Python SDK pointed at OpenRouter's base URL, which gives us
access to hundreds of models through a single API. Tool calling follows the
OpenAI function-calling format, which OpenRouter supports natively.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from config import Config
from context import build_system_prompt, load_static_context
from session import Session
from tool_registry import ToolRegistry
from tools import set_current_task_store

logger = logging.getLogger("agent_computer.agent")

# ─── Activity broadcasting ───
_activity_log: deque[dict] = deque(maxlen=200)
_activity_listeners: list[asyncio.Queue] = []


def subscribe_activity() -> asyncio.Queue:
    """Subscribe to live activity events. Returns a queue to read from."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _activity_listeners.append(q)
    return q


def unsubscribe_activity(q: asyncio.Queue) -> None:
    """Unsubscribe from activity events."""
    if q in _activity_listeners:
        _activity_listeners.remove(q)


def get_recent_activity(limit: int = 50) -> list[dict]:
    """Get recent activity events from the ring buffer."""
    return list(_activity_log)[-limit:]


def _broadcast_activity(event: dict) -> None:
    """Store event in ring buffer and push to all listeners."""
    _activity_log.append(event)
    for q in _activity_listeners:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


@dataclass
class AgentEvent:
    """Events emitted during the agent loop for streaming to clients."""
    type: str  # "thinking", "text", "tool_call", "tool_result", "error", "done", "task_update"
    data: dict[str, Any]


class AgentRuntime:
    """The agent runtime — runs the agentic loop for a given session.

    Uses OpenRouter as the model gateway via the OpenAI-compatible API.
    This means you can use any model on OpenRouter (Claude, GPT, Gemini,
    Llama, DeepSeek, etc.) just by changing the model_id in config.
    """

    def __init__(self, config: Config, tool_registry: ToolRegistry, memory_search=None):
        self.config = config
        self.agent_config = config.agent
        self.tools = tool_registry
        self.memory_search = memory_search
        self._openrouter_api_key: str | None = None

        # OpenAI SDK pointed at OpenRouter
        self.client = AsyncOpenAI(
            base_url=config.agent.model.base_url,
            api_key="placeholder",  # Will be overridden by env var
            default_headers={
                "X-OpenRouter-Title": config.agent.name,
            },
        )

    def set_model(self, provider: str, model_id: str, base_url: str, api_key: str | None = None) -> None:
        """Switch the active model at runtime."""
        # Save OpenRouter key the first time we see it
        if self._openrouter_api_key is None and self.client.api_key and self.client.api_key != "placeholder":
            self._openrouter_api_key = self.client.api_key

        self.agent_config.model.provider = provider
        self.agent_config.model.model_id = model_id
        self.agent_config.model.base_url = base_url

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or self._openrouter_api_key or "placeholder",
            default_headers={
                "X-OpenRouter-Title": self.agent_config.name,
            },
        )
        logger.info(f"Model switched: {model_id} via {provider} ({base_url})")

    async def run(self, session: Session, user_message: str, mode: str | None = None) -> AsyncIterator[AgentEvent]:
        """Run the full agentic loop for a user message.

        Yields AgentEvent objects for real-time streaming to the client.
        """
        # Resolve mode: explicit param > session mode > default
        effective_mode = mode or session.mode
        is_deep_work = effective_mode == "deep_work"
        logger.info(f"Agent.run() session={session.session_id} mode_param={mode} session_mode={session.mode} effective={effective_mode}")

        # 1. Add user message to session
        session.add_message("user", user_message)

        # 2. Determine limits based on mode
        if is_deep_work:
            max_iterations = self.config.agent.deep_work.max_iterations
            token_budget = self.config.agent.deep_work.token_budget
            warning_threshold = self.config.agent.deep_work.warning_threshold
        else:
            max_iterations = self.agent_config.max_loop_iterations
            token_budget = 0  # No budget enforcement in bounded mode
            warning_threshold = 0

        # 3. Bind task store for deep work mode
        if is_deep_work:
            set_current_task_store(session.task_store, "deep_work")
        else:
            set_current_task_store(None, "bounded")

        try:
            # 4. Build tool schemas — filter manage_tasks out in bounded mode
            allowed_tools = list(self.agent_config.tools.allow)
            if not is_deep_work and "manage_tasks" in allowed_tools:
                allowed_tools.remove("manage_tasks")
            tool_schemas = self.tools.get_openai_tools(allowed=allowed_tools)

            # 5. Cache static context (read files once per run, not every iteration)
            static_ctx = load_static_context(self.agent_config.workspace)
            tool_name_list = [t.name for t in self.tools.list_tools() if t.name in allowed_tools]

            # 6. Search relevant memories for this user message (once per run)
            relevant_memories = None
            if self.memory_search:
                try:
                    results = await self.memory_search.async_search(user_message)
                    if results:
                        relevant_memories = [
                            {"source_type": r.source_type, "source_id": r.source_id,
                             "title": r.title, "content": r.content, "score": r.score}
                            for r in results
                        ]
                except Exception as e:
                    logger.warning(f"Memory search failed: {e}")

            session_summary = ""

            # 7. Build initial system prompt
            task_summary = session.task_store.summary() if is_deep_work else ""
            pending_task_count = sum(
                1 for t in session.task_store.list_all()
                if t.status in ("pending", "in_progress")
            ) if is_deep_work else 0
            system_prompt = build_system_prompt(
                self.agent_config.workspace,
                self.agent_config.name,
                mode=effective_mode,
                task_summary=task_summary,
                pending_task_count=pending_task_count,
                context_file="",
                relevant_memories=relevant_memories,
                user_message=user_message,
                tool_names=tool_name_list,
                session_summary=session_summary,
                **static_ctx,
            )

            # 6. Run the agentic loop
            iteration = 0
            consecutive_text_only = 0  # Safety valve: exit after 2 consecutive text-only responses
            run_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            context_file = ""  # Path to compacted context MD file (set after compaction)
            compaction_count = 0
            max_compactions = 5  # Safety cap — effectively 6x budget total
            compaction_threshold = 0.75

            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Agent loop iteration {iteration}/{max_iterations} (mode={effective_mode})")

                # Deep work: rebuild system prompt each iteration with fresh task summary + budget warning
                if is_deep_work and iteration > 1:
                    task_summary = session.task_store.summary()
                    pending_task_count = sum(
                        1 for t in session.task_store.list_all()
                        if t.status in ("pending", "in_progress")
                    )
                    budget_warning = ""
                    if token_budget > 0:
                        usage_ratio = run_usage["total_tokens"] / token_budget
                        if usage_ratio >= warning_threshold:
                            pct = round(usage_ratio * 100)
                            remaining = token_budget - run_usage["total_tokens"]
                            budget_warning = (
                                f"WARNING: You have used {pct}% of your token budget "
                                f"({run_usage['total_tokens']:,}/{token_budget:,} tokens). "
                                f"{remaining:,} tokens remaining. "
                                f"Wrap up your work soon — prioritize completing critical tasks."
                            )
                    # Update session summary every 10 iterations from completed tasks
                    if iteration % 10 == 0:
                        completed = [t for t in session.task_store.list_all() if t.status == "completed"]
                        if completed:
                            session_summary = "Completed: " + "; ".join(t.title for t in completed[:20])
                    system_prompt = build_system_prompt(
                        self.agent_config.workspace,
                        self.agent_config.name,
                        mode=effective_mode,
                        task_summary=task_summary,
                        budget_warning=budget_warning,
                        pending_task_count=pending_task_count,
                        context_file=context_file,
                        relevant_memories=relevant_memories,
                        user_message=user_message,
                        tool_names=tool_name_list,
                        session_summary=session_summary,
                        **static_ctx,
                    )

                # Emit thinking event (enhanced in deep-work mode)
                thinking_data: dict[str, Any] = {"iteration": iteration}
                if is_deep_work:
                    thinking_data.update({
                        "max_iterations": max_iterations,
                        "tokens_used": run_usage["total_tokens"],
                        "token_budget": token_budget,
                        "task_summary": task_summary,
                    })
                yield AgentEvent("thinking", thinking_data)

                # Auto-compaction: when approaching budget limit, compact and reset
                if (is_deep_work and token_budget > 0
                        and compaction_count < max_compactions
                        and run_usage["total_tokens"] / token_budget >= compaction_threshold):
                    compaction_count += 1
                    task_summary = session.task_store.summary()
                    context_file = session.compact(
                        self.agent_config.workspace, task_summary
                    )
                    logger.info(
                        f"Auto-compaction #{compaction_count}: "
                        f"{run_usage['total_tokens']:,} tokens used, "
                        f"context saved to {context_file}"
                    )
                    # Reset token budget counter
                    run_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    # Notify the user/client
                    yield AgentEvent("text", {
                        "text": (
                            f"[Auto-compacted conversation (#{compaction_count}/{max_compactions}). "
                            f"Context saved to {context_file}. Continuing work...]"
                        ),
                    })
                    # Rebuild system prompt with context_file reference
                    pending_task_count = sum(
                        1 for t in session.task_store.list_all()
                        if t.status in ("pending", "in_progress")
                    )
                    system_prompt = build_system_prompt(
                        self.agent_config.workspace,
                        self.agent_config.name,
                        mode=effective_mode,
                        task_summary=task_summary,
                        pending_task_count=pending_task_count,
                        context_file=context_file,
                        relevant_memories=relevant_memories,
                        user_message=user_message,
                        tool_names=tool_name_list,
                        session_summary=session_summary,
                        **static_ctx,
                    )

                # Token budget enforcement (hard stop — safety net after compaction cap)
                if is_deep_work and token_budget > 0 and run_usage["total_tokens"] >= token_budget:
                    yield AgentEvent("error", {
                        "message": f"Token budget exhausted ({run_usage['total_tokens']:,}/{token_budget:,} tokens). Stopping.",
                    })
                    return

                # Build messages for the API
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(session.get_openai_messages())

                # Call the model via OpenRouter
                try:
                    kwargs: dict[str, Any] = {
                        "model": self.agent_config.model.model_id,
                        "max_tokens": self.agent_config.model.max_tokens,
                        "messages": messages,
                    }
                    if tool_schemas:
                        kwargs["tools"] = tool_schemas

                    response = await self.client.chat.completions.create(**kwargs)

                except Exception as e:
                    logger.error(f"OpenRouter API error: {e}")
                    yield AgentEvent("error", {"message": f"API error: {e}"})
                    return

                choice = response.choices[0]
                message = choice.message

                # Record token usage
                if response.usage:
                    run_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                    run_usage["completion_tokens"] += response.usage.completion_tokens or 0
                    run_usage["total_tokens"] += response.usage.total_tokens or 0
                    session.add_message("meta", {
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        },
                        "model": response.model,
                        "iteration": iteration,
                    })

                # Check for tool calls
                if message.tool_calls:
                    # Save the assistant message with tool calls
                    session.add_message("assistant", _serialize_assistant_message(message))

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        fn = tool_call.function
                        tool_name = fn.name
                        try:
                            tool_args = json.loads(fn.arguments) if fn.arguments else {}
                        except json.JSONDecodeError:
                            tool_args = {}

                        yield AgentEvent("tool_call", {
                            "tool": tool_name,
                            "input": tool_args,
                            "tool_call_id": tool_call.id,
                        })

                        _broadcast_activity({
                            "type": "tool_call",
                            "session_id": session.session_id,
                            "tool": tool_name,
                            "input": tool_args,
                            "timestamp": time.time(),
                        })

                        # Execute the tool with timing
                        logger.info(f"Executing tool: {tool_name}({tool_args})")
                        t0 = time.monotonic()
                        result = await self.tools.execute(tool_name, tool_args)
                        duration_ms = round((time.monotonic() - t0) * 1000)

                        success = not (result.startswith('{"error"') if isinstance(result, str) else False)

                        yield AgentEvent("tool_result", {
                            "tool": tool_name,
                            "tool_call_id": tool_call.id,
                            "result_preview": result[:500] if len(result) > 500 else result,
                            "duration_ms": duration_ms,
                            "success": success,
                            "result_length": len(result),
                        })

                        _broadcast_activity({
                            "type": "tool_result",
                            "session_id": session.session_id,
                            "tool": tool_name,
                            "duration_ms": duration_ms,
                            "success": success,
                            "result_preview": result[:200] if len(result) > 200 else result,
                            "timestamp": time.time(),
                        })

                        # Add tool result to session
                        session.add_message("tool", result, tool_call_id=tool_call.id, tool_name=tool_name)

                        # Emit task_update event after manage_tasks execution
                        if tool_name == "manage_tasks" and is_deep_work:
                            yield AgentEvent("task_update", {
                                "tasks": session.task_store.to_dict(),
                                "summary": session.task_store.summary(),
                            })

                    # Loop continues — model will see tool results and decide next step
                    consecutive_text_only = 0  # Reset: model is actively using tools
                    continue

                # No tool calls — check if we should continue or exit
                final_text = message.content or ""

                if final_text:
                    session.add_message("assistant", final_text)
                    yield AgentEvent("text", {"text": final_text})

                # Deep work: don't exit if there are still pending tasks
                if is_deep_work:
                    pending = [t for t in session.task_store.list_all()
                               if t.status in ("pending", "in_progress")]
                    if pending and consecutive_text_only < 4:
                        consecutive_text_only += 1
                        logger.info(
                            f"Deep work: text-only response but {len(pending)} tasks remain "
                            f"(consecutive_text_only={consecutive_text_only}). Injecting nudge."
                        )
                        # Inject a nudge to get the model back on track
                        pending_titles = ", ".join(
                            f"[{t.id}] {t.title}" for t in pending[:5]
                        )
                        nudge = (
                            f"[SYSTEM: You have {len(pending)} pending/in-progress task(s): "
                            f"{pending_titles}. Do NOT ask the user — pick up the next task "
                            f"and continue working. Use tools to make progress.]"
                        )
                        session.add_message("user", nudge)
                        continue  # Keep looping — model will see nudge and act

                # Exit normally (bounded mode, or no pending tasks, or safety valve hit)
                yield AgentEvent("done", {
                    "text": final_text,
                    "iterations": iteration,
                    "finish_reason": choice.finish_reason,
                    "model": response.model,
                    "usage": run_usage,
                    "mode": effective_mode,
                })
                return

            # Hit max iterations
            yield AgentEvent("error", {
                "message": f"Agent loop hit max iterations ({max_iterations}). Stopping.",
            })

        finally:
            # Always unbind the task store
            set_current_task_store(None, "bounded")

    async def run_simple(self, session: Session, user_message: str, mode: str | None = None) -> str:
        """Run the agent loop and return the final text response (non-streaming)."""
        final_text = ""
        async for event in self.run(session, user_message, mode=mode):
            if event.type == "done":
                final_text = event.data.get("text", "")
            elif event.type == "error":
                final_text = f"Error: {event.data.get('message', 'Unknown error')}"
        return final_text


def _serialize_assistant_message(message) -> dict:
    """Serialize an OpenAI assistant message (with tool calls) for session storage."""
    data: dict[str, Any] = {"role": "assistant"}

    if message.content:
        data["content"] = message.content

    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]

    return data
