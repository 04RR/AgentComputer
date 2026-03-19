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
from context import build_system_prompt, build_static_prompt_prefix, build_dynamic_suffix, load_static_context
from persona import Persona, PersonaStore
from session import Session
from tool_registry import ToolRegistry

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
        self.persona_store: PersonaStore | None = None
        self.cron_scheduler = None
        self._openrouter_api_key: str | None = None

        # OpenAI-compatible client — key depends on provider
        if config.agent.model.provider == "lmstudio":
            api_key = config.lmstudio.api_key
        else:
            api_key = "placeholder"  # Overridden by env var in gateway.main()

        self.client = AsyncOpenAI(
            base_url=config.agent.model.base_url,
            api_key=api_key,
            default_headers={
                "X-OpenRouter-Title": config.agent.name,
            },
        )

    def set_model(self, provider: str, model_id: str, base_url: str, api_key: str | None = None) -> None:
        """Switch the active model at runtime."""
        # Save any real API key so we can restore it when switching back to OpenRouter
        if api_key and api_key not in ("placeholder", "lm-studio"):
            self._openrouter_api_key = api_key

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

    async def run(self, session: Session, user_message: str, mode: str | None = None,
                   persona: Persona | None = None, memory_search_override=None) -> AsyncIterator[AgentEvent]:
        """Run the full agentic loop for a user message.

        Yields AgentEvent objects for real-time streaming to the client.
        When persona is provided, uses the persona's workspace and settings.
        """
        # Derive effective workspace
        effective_workspace = persona.workspace_path if persona else self.agent_config.workspace

        # Model override for persona
        _model_overridden = False
        _saved_client = None
        _saved_model_id = None
        _saved_provider = None
        _saved_base_url = None

        if persona and persona.model_override:
            override = persona.model_override
            if override.startswith("openrouter/"):
                ov_provider = "openrouter"
                ov_model_id = override[len("openrouter/"):]
                ov_base_url = "https://openrouter.ai/api/v1"
                ov_api_key = self._openrouter_api_key
            elif override.startswith("lmstudio/"):
                ov_provider = "lmstudio"
                ov_model_id = override[len("lmstudio/"):]
                ov_base_url = self.config.lmstudio.base_url
                ov_api_key = self.config.lmstudio.api_key
            else:
                ov_provider = self.agent_config.model.provider
                ov_model_id = override
                ov_base_url = self.agent_config.model.base_url
                ov_api_key = self.client.api_key

            _model_overridden = True
            _saved_client = self.client
            _saved_model_id = self.agent_config.model.model_id
            _saved_provider = self.agent_config.model.provider
            _saved_base_url = self.agent_config.model.base_url

            self.agent_config.model.provider = ov_provider
            self.agent_config.model.model_id = ov_model_id
            self.agent_config.model.base_url = ov_base_url
            self.client = AsyncOpenAI(
                base_url=ov_base_url,
                api_key=ov_api_key or "placeholder",
                default_headers={"X-OpenRouter-Title": self.agent_config.name},
            )
            logger.info(f"Model override active: {ov_model_id} via {ov_provider}")

        try:
            async for event in self._run_inner(session, user_message, mode, persona, memory_search_override, effective_workspace):
                yield event
        finally:
            if _model_overridden:
                self.client = _saved_client
                self.agent_config.model.model_id = _saved_model_id
                self.agent_config.model.provider = _saved_provider
                self.agent_config.model.base_url = _saved_base_url
                logger.info(f"Model override restored: {_saved_model_id}")

    async def _run_inner(self, session: Session, user_message: str, mode: str | None = None,
                          persona: Persona | None = None, memory_search_override=None,
                          effective_workspace: str = "") -> AsyncIterator[AgentEvent]:
        """Inner run logic, separated to allow model override try/finally wrapper."""

        # Resolve mode: explicit param > session mode > default
        effective_mode = mode or session.mode
        is_deep_work = effective_mode == "deep_work"
        logger.info(f"Agent.run() session={session.session_id} mode_param={mode} session_mode={session.mode} effective={effective_mode}")

        # 1. Add user message to session
        session.add_message("user", user_message)

        # 2. Determine limits based on mode and phase
        is_planning = False
        if is_deep_work:
            # Auto-transition: first message in deep work → planning phase
            if session.deep_work_phase is None:
                session.deep_work_phase = "planning"

            is_planning = session.deep_work_phase == "planning"

            if is_planning:
                max_iterations = min(30, self.config.agent.deep_work.max_iterations)
                token_budget = 0  # No budget enforcement during planning
                warning_threshold = 0
            else:
                max_iterations = self.config.agent.deep_work.max_iterations
                token_budget = self.config.agent.deep_work.token_budget
                warning_threshold = self.config.agent.deep_work.warning_threshold
        else:
            max_iterations = self.agent_config.max_loop_iterations
            token_budget = 0
            warning_threshold = 0

        # 3. Build tool context for this run (replaces global task store binding)
        tool_context = {
            "task_store": session.task_store if is_deep_work else None,
            "mode": effective_mode,
            "session_id": session.session_id,
            "workspace": effective_workspace,
            "memory_search": memory_search_override,
            "persona_store": self.persona_store,
            "cron_scheduler": self.cron_scheduler,
        }

        # 4. Build tool schemas — filter manage_tasks out in bounded mode
        if persona and self.persona_store:
            allowed_tools = self.persona_store.get_effective_tools(persona, list(self.agent_config.tools.allow))
        else:
            allowed_tools = list(self.agent_config.tools.allow)
        if not is_deep_work and "manage_tasks" in allowed_tools:
            allowed_tools.remove("manage_tasks")
        tool_schemas = self.tools.get_openai_tools(allowed=allowed_tools)

        # 5. Cache static context (read files once per run, not every iteration)
        static_ctx = load_static_context(
            effective_workspace,
            user_md_path=persona.user_md_path if persona else None,
        )
        tool_name_list = [t.name for t in self.tools.list_tools() if t.name in allowed_tools]

        # 6. Search relevant memories for this user message (once per run)
        effective_memory_search = memory_search_override or self.memory_search
        relevant_memories = None
        if effective_memory_search:
            try:
                results = await effective_memory_search.async_search(user_message)
                if results:
                    relevant_memories = [
                        {"source_type": r.source_type, "source_id": r.source_id,
                         "title": r.title, "content": r.content, "score": r.score}
                        for r in results
                    ]
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")

        session_summary = ""

        # 7. Build system prompt — cache static prefix for deep work reuse
        if is_deep_work:
            static_prefix = build_static_prompt_prefix(
                workspace=effective_workspace,
                agent_name=self.agent_config.name,
                mode=effective_mode,
                deep_work_phase=session.deep_work_phase,
                relevant_memories=relevant_memories,
                tool_names=tool_name_list,
                session_summary=session_summary,
                **static_ctx,
            )
            task_summary = session.task_store.summary()
            pending_task_count = session.task_store.pending_count()
            suffix = build_dynamic_suffix(
                task_summary=task_summary,
                pending_task_count=pending_task_count,
            )
            system_prompt = static_prefix + ("\n\n" + suffix if suffix else "")
        else:
            static_prefix = ""
            task_summary = ""
            pending_task_count = 0
            system_prompt = build_system_prompt(
                effective_workspace,
                self.agent_config.name,
                mode=effective_mode,
                relevant_memories=relevant_memories,
                user_message=user_message,
                tool_names=tool_name_list,
                session_summary=session_summary,
                max_iterations=max_iterations,
                provider=self.agent_config.model.provider,
                **static_ctx,
            )

        # 8. Run the agentic loop
        iteration = 0
        consecutive_text_only = 0  # Safety valve: exit after 2 consecutive text-only responses
        # Circuit breaker for repetitive tool calls (lmstudio only)
        consecutive_same_tool = 0
        last_tool_name: str | None = None
        run_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        context_file = ""  # Path to compacted context MD file (set after compaction)
        compaction_count = 0
        max_compactions = 5  # Safety cap — effectively 6x budget total
        compaction_threshold = 0.75

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}/{max_iterations} (mode={effective_mode})")

            # Deep work: rebuild dynamic suffix each iteration (static prefix is cached)
            if is_deep_work and iteration > 1:
                task_summary = session.task_store.summary()
                pending_task_count = session.task_store.pending_count()
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
                    completed = session.task_store.completed_list()
                    if completed:
                        session_summary = "Completed: " + "; ".join(t.title for t in completed[:20])
                        # Rebuild static prefix with updated session summary
                        static_prefix = build_static_prompt_prefix(
                            workspace=effective_workspace,
                            agent_name=self.agent_config.name,
                            mode=effective_mode,
                            deep_work_phase=session.deep_work_phase,
                            relevant_memories=relevant_memories,
                            tool_names=tool_name_list,
                            session_summary=session_summary,
                            **static_ctx,
                        )
                suffix = build_dynamic_suffix(
                    task_summary=task_summary,
                    budget_warning=budget_warning,
                    pending_task_count=pending_task_count,
                    context_file=context_file,
                )
                system_prompt = static_prefix + ("\n\n" + suffix if suffix else "")

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
                    effective_workspace, task_summary
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
                pending_task_count = session.task_store.pending_count()
                # Reset openai message cache since messages were compacted
                suffix = build_dynamic_suffix(
                    task_summary=task_summary,
                    pending_task_count=pending_task_count,
                    context_file=context_file,
                )
                system_prompt = static_prefix + ("\n\n" + suffix if suffix else "")

            # Token budget enforcement (hard stop — safety net after compaction cap)
            if is_deep_work and token_budget > 0 and run_usage["total_tokens"] >= token_budget:
                session.flush()
                session.task_store.flush()
                yield AgentEvent("error", {
                    "message": f"Token budget exhausted ({run_usage['total_tokens']:,}/{token_budget:,} tokens). Stopping.",
                })
                return

            # Iteration warning for bounded mode on local models
            if not is_deep_work and iteration >= max_iterations - 3 and self.agent_config.model.provider == "lmstudio":
                nudge = (
                    f"[SYSTEM: You have {max_iterations - iteration} iteration(s) remaining. "
                    f"Stop using tools and provide your final answer NOW with the information you already have.]"
                )
                session.add_message("user", nudge)

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
                session.flush()
                session.task_store.flush()
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

                # Parse all tool calls upfront
                parsed_calls = []
                for tool_call in message.tool_calls:
                    fn = tool_call.function
                    tool_name = fn.name
                    try:
                        tool_args = json.loads(fn.arguments) if fn.arguments else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                    parsed_calls.append((tool_call, tool_name, tool_args))

                # Emit all tool_call events and broadcast activity
                for tool_call, tool_name, tool_args in parsed_calls:
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

                # Execute tool calls in parallel (defer task store saves during batch)
                if is_deep_work:
                    session.task_store._auto_save = False

                async def _exec_tool(tc_name, tc_args):
                    t0 = time.monotonic()
                    res = await self.tools.execute(tc_name, tc_args, context=tool_context)
                    dur = round((time.monotonic() - t0) * 1000)
                    return res, dur

                logger.info(f"Executing {len(parsed_calls)} tool(s) in parallel: {[n for _, n, _ in parsed_calls]}")
                exec_results = await asyncio.gather(
                    *(_exec_tool(name, args) for _, name, args in parsed_calls)
                )

                if is_deep_work:
                    session.task_store._auto_save = True
                    session.task_store.flush()

                # Emit results and add to session in order
                for (tool_call, tool_name, tool_args), (result, duration_ms) in zip(parsed_calls, exec_results):
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

                # Flush buffered writes before next iteration
                session.flush()

                # Circuit breaker: detect repetitive same-tool calls (lmstudio only)
                if self.agent_config.model.provider == "lmstudio":
                    tool_names_this_iter = set(name for _, name, _ in parsed_calls)
                    if len(tool_names_this_iter) == 1 and tool_names_this_iter.pop() == last_tool_name:
                        consecutive_same_tool += 1
                    else:
                        consecutive_same_tool = 1
                    last_tool_name = next(iter(set(name for _, name, _ in parsed_calls)))
                    if consecutive_same_tool >= 3:
                        session.add_message("user",
                            "[SYSTEM: You have called the same tool multiple times consecutively. "
                            "Synthesize the information you already have and respond to the user.]"
                        )
                        consecutive_same_tool = 0

                # Loop continues — model will see tool results and decide next step
                consecutive_text_only = 0  # Reset: model is actively using tools
                continue

            # No tool calls — check if we should continue or exit
            final_text = message.content or ""

            if final_text:
                session.add_message("assistant", final_text)
                yield AgentEvent("text", {"text": final_text})
                # Reset circuit breaker on text response
                consecutive_same_tool = 0
                last_tool_name = None

            # Deep work execution phase: don't exit if there are still pending tasks
            if is_deep_work and not is_planning:
                pending_count_now = session.task_store.pending_count()
                if pending_count_now > 0 and consecutive_text_only < 2:
                    consecutive_text_only += 1
                    logger.info(
                        f"Deep work: text-only response but {pending_count_now} tasks remain "
                        f"(consecutive_text_only={consecutive_text_only}). Injecting nudge."
                    )
                    pending_tasks = [t for t in session.task_store.list_all()
                                     if t.status in ("pending", "in_progress")]
                    pending_titles = ", ".join(
                        f"[{t.id}] {t.title}" for t in pending_tasks[:5]
                    )
                    nudge = (
                        f"[SYSTEM: You have {pending_count_now} pending/in-progress task(s): "
                        f"{pending_titles}. Do NOT ask the user — pick up the next task "
                        f"and continue working. Use tools to make progress.]"
                    )
                    session.add_message("user", nudge)
                    continue

            # Planning phase complete — emit plan_ready instead of done
            if is_planning:
                session.flush()
                session.task_store.flush()
                yield AgentEvent("plan_ready", {
                    "text": final_text,
                    "tasks": session.task_store.to_dict(),
                    "summary": session.task_store.summary(),
                    "iterations": iteration,
                    "model": response.model,
                    "usage": run_usage,
                })
                return

            # Exit normally (bounded mode, or no pending tasks, or safety valve hit)
            session.flush()
            session.task_store.flush()
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
        session.flush()
        session.task_store.flush()
        if is_planning:
            yield AgentEvent("plan_ready", {
                "text": "Planning reached iteration limit. Here's what I have so far.",
                "tasks": session.task_store.to_dict(),
                "summary": session.task_store.summary(),
                "iterations": iteration,
                "usage": run_usage,
            })
        else:
            yield AgentEvent("error", {
                "message": f"Agent loop hit max iterations ({max_iterations}). Stopping.",
            })

    async def run_simple(self, session: Session, user_message: str, mode: str | None = None,
                          persona: Persona | None = None, memory_search_override=None) -> str:
        """Run the agent loop and return the final text response (non-streaming)."""
        final_text = ""
        async for event in self.run(session, user_message, mode=mode, persona=persona, memory_search_override=memory_search_override):
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
