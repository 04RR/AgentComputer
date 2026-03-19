"""Session management for agent_computer.

Each session is a stateful conversation persisted as a JSONL file.
Sessions enforce serial execution (one message processed at a time)
to prevent state corruption — a key OpenClaw design principle.

Message format follows the OpenAI chat completions API (used by OpenRouter):
  - user: {"role": "user", "content": "..."}
  - assistant: {"role": "assistant", "content": "...", "tool_calls": [...]}
  - tool: {"role": "tool", "content": "...", "tool_call_id": "..."}
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from task_store import TaskStore

logger = logging.getLogger("agent_computer.session")


@dataclass
class Message:
    role: str  # "user", "assistant", "tool", "system", "meta"
    content: Any  # str, dict (assistant with tool_calls), or list
    timestamp: float = field(default_factory=time.time)
    tool_call_id: str | None = None
    tool_name: str | None = None

    def to_openai(self) -> dict:
        """Convert to OpenAI messages API format for the next API call."""
        if self.role == "tool":
            return {
                "role": "tool",
                "content": self.content if isinstance(self.content, str) else json.dumps(self.content),
                "tool_call_id": self.tool_call_id,
            }

        if self.role == "assistant" and isinstance(self.content, dict):
            # Assistant message with tool_calls — pass through as-is
            msg = dict(self.content)
            return msg

        return {"role": self.role, "content": self.content}

    def to_jsonl(self) -> str:
        """Serialize for JSONL storage."""
        return json.dumps({
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
        })


class Session:
    """A single conversation session with JSONL persistence and serial execution."""

    def __init__(self, session_id: str, storage_dir: str):
        self.session_id = session_id
        self.messages: list[Message] = []
        self._storage_path = Path(storage_dir) / f"{session_id}.jsonl"
        self._lock = asyncio.Lock()
        self.task_store = TaskStore(Path(storage_dir) / f"{session_id}.tasks.json")
        self.mode: str = "bounded"  # "bounded" or "deep_work"
        self.deep_work_phase: str | None = None  # None, "planning", "executing"
        self.persona_id: str = "default"
        # Incremental OpenAI message conversion cache
        self._openai_cache: list[dict] = []
        self._openai_cache_len: int = 0
        # Buffered JSONL writes
        self._write_buffer: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self.messages.append(Message(
                        role=data["role"],
                        content=data["content"],
                        timestamp=data.get("timestamp", 0),
                        tool_call_id=data.get("tool_call_id"),
                        tool_name=data.get("tool_name"),
                    ))
            # Scan for persona_id in meta messages
            for msg in self.messages:
                if msg.role == "meta" and isinstance(msg.content, dict) and "persona_id" in msg.content:
                    self.persona_id = msg.content["persona_id"]
                    break
            logger.info(f"Loaded session {self.session_id}: {len(self.messages)} messages (persona={self.persona_id})")
        except Exception as e:
            logger.error(f"Failed to load session {self.session_id}: {e}")

    def set_persona(self, persona_id: str) -> None:
        """Set the persona for this session and persist it as a meta message."""
        self.persona_id = persona_id
        self.add_message("meta", {"persona_id": persona_id})

    def _persist(self, message: Message) -> None:
        self._write_buffer.append(message.to_jsonl() + "\n")

    def flush(self) -> None:
        """Write all buffered JSONL lines to disk at once."""
        if not self._write_buffer:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "a", encoding="utf-8") as f:
            f.writelines(self._write_buffer)
        self._write_buffer.clear()

    def add_message(self, role: str, content: Any, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self._persist(msg)
        return msg

    def get_openai_messages(self, max_messages: int | None = None) -> list[dict]:
        """Get messages in OpenAI API format (excludes system — that's added separately).

        Uses incremental conversion: only converts new messages since last call.
        """
        # Convert only new messages
        for msg in self.messages[self._openai_cache_len:]:
            if msg.role in ("user", "assistant", "tool"):
                self._openai_cache.append(msg.to_openai())
        self._openai_cache_len = len(self.messages)

        if max_messages and len(self._openai_cache) > max_messages:
            return list(self._openai_cache[-max_messages:])

        return list(self._openai_cache)

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def get_history(self) -> list[dict]:
        """Get all displayable messages (excludes meta) for REST history endpoint."""
        result = []
        for msg in self.messages:
            if msg.role == "meta":
                continue
            entry: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.tool_name:
                entry["tool_name"] = msg.tool_name
            result.append(entry)
        return result

    def get_preview(self) -> str:
        """First user message text (truncated), for sidebar display."""
        for msg in self.messages:
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content[:80].replace("\n", " ")
        return self.session_id

    def get_last_activity(self) -> float | None:
        """Timestamp of the most recent message."""
        if self.messages:
            return self.messages[-1].timestamp
        return None

    def get_created_at(self) -> float | None:
        """Timestamp of the first message."""
        if self.messages:
            return self.messages[0].timestamp
        return None

    def get_token_usage(self, since: float | None = None) -> dict:
        """Sum token usage from meta messages, optionally filtered by time."""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        api_calls = 0
        for msg in self.messages:
            if msg.role != "meta" or not isinstance(msg.content, dict):
                continue
            if since and msg.timestamp < since:
                continue
            usage = msg.content.get("usage", {})
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            api_calls += 1
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "api_calls": api_calls,
        }

    def get_tasks(self) -> list[dict]:
        return self.task_store.to_dict()

    def compact(self, workspace: str, task_summary: str = "") -> str:
        """Compact conversation history: save full context to MD file, trim messages.

        Returns the path to the saved context file.
        """
        context_dir = Path(workspace) / ".agent_context"
        context_dir.mkdir(parents=True, exist_ok=True)
        context_file = context_dir / f"{self.session_id}_context.md"

        # Build the context markdown
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Conversation Context — Session {self.session_id}",
            f"Compacted at: {now}",
            f"Total messages before compaction: {len(self.messages)}",
            "",
        ]

        if task_summary:
            lines.extend(["## Task Progress", task_summary, ""])

        lines.append("## Conversation Log")
        lines.append("")

        for msg in self.messages:
            if msg.role == "meta":
                continue
            if msg.role == "user":
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                lines.append(f"### User\n{content}\n")
            elif msg.role == "assistant":
                if isinstance(msg.content, dict):
                    # Assistant message with tool_calls
                    text = msg.content.get("content", "") or ""
                    if text:
                        lines.append(f"### Assistant\n{text}\n")
                    tool_calls = msg.content.get("tool_calls", [])
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")
                        if len(args) > 500:
                            args = args[:500] + "...[truncated]"
                        lines.append(f"**Tool call**: `{name}`\n```\n{args}\n```\n")
                else:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    lines.append(f"### Assistant\n{content}\n")
            elif msg.role == "tool":
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                tool_label = msg.tool_name or "tool"
                if len(content) > 1000:
                    content = content[:1000] + "...[truncated]"
                lines.append(f"**Tool result** (`{tool_label}`):\n```\n{content}\n```\n")

        context_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved compaction context to {context_file} ({len(lines)} lines)")

        # Trim messages: keep first user message + compaction notice + last ~20 messages
        # Walk backwards to find safe turn boundaries (don't orphan tool results)
        first_user_msg = None
        for msg in self.messages:
            if msg.role == "user":
                first_user_msg = msg
                break

        # Find the last N messages, respecting turn boundaries
        keep_tail = self._find_safe_tail(count=20)

        # Build trimmed message list
        trimmed: list[Message] = []
        if first_user_msg:
            trimmed.append(first_user_msg)

        # Add compaction notice as assistant message
        trimmed.append(Message(
            role="assistant",
            content=(
                f"[Conversation compacted — {len(self.messages)} messages archived to "
                f"{context_file}. Use read_file to review earlier work. "
                f"Task list is the source of truth for progress.]"
            ),
        ))

        trimmed.extend(keep_tail)
        self.messages = trimmed
        # Reset caches after compaction
        self._openai_cache.clear()
        self._openai_cache_len = 0
        self._write_buffer.clear()  # Discard buffer — rewrite replaces the file
        self._rewrite_storage()

        logger.info(f"Compacted session {self.session_id}: kept {len(trimmed)} messages")
        return str(context_file)

    def _find_safe_tail(self, count: int = 20) -> list[Message]:
        """Get the last ~count messages without breaking tool_call/tool_result pairs.

        Groups messages into turns (assistant+tool_results or standalone user/assistant),
        then takes the last N turns that fit within the count.
        """
        api_messages = [m for m in self.messages if m.role in ("user", "assistant", "tool")]
        if len(api_messages) <= count:
            return list(api_messages)

        # Group into turns: each turn is a list of messages
        turns: list[list[Message]] = []
        i = 0
        while i < len(api_messages):
            msg = api_messages[i]
            if msg.role == "assistant" and isinstance(msg.content, dict) and msg.content.get("tool_calls"):
                # Assistant with tool_calls — collect it + subsequent tool results
                turn = [msg]
                i += 1
                while i < len(api_messages) and api_messages[i].role == "tool":
                    turn.append(api_messages[i])
                    i += 1
                turns.append(turn)
            else:
                turns.append([msg])
                i += 1

        # Take turns from the end until we have ~count messages
        result: list[Message] = []
        for turn in reversed(turns):
            if len(result) + len(turn) > count and result:
                break
            result = turn + result

        return result

    def _rewrite_storage(self) -> None:
        """Rewrite the JSONL file from the current (trimmed) messages."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as f:
            for msg in self.messages:
                f.write(msg.to_jsonl() + "\n")
        logger.info(f"Rewrote storage for session {self.session_id}: {len(self.messages)} messages")

    def clear(self) -> None:
        self.messages.clear()
        self._openai_cache.clear()
        self._openai_cache_len = 0
        self._write_buffer.clear()
        if self._storage_path.exists():
            self._storage_path.unlink()
        self.task_store.clear()
        logger.info(f"Cleared session {self.session_id}")


class SessionManager:
    """Manages all active sessions."""

    def __init__(self, storage_dir: str):
        self._storage_dir = storage_dir
        self._sessions: dict[str, Session] = {}
        Path(storage_dir).mkdir(parents=True, exist_ok=True)

    def get_or_create(self, session_id: str | None = None, persona_id: str = "default") -> Session:
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        if session_id not in self._sessions:
            session = Session(session_id, self._storage_dir)
            if persona_id != "default" and session.persona_id == "default":
                session.set_persona(persona_id)
            self._sessions[session_id] = session
        return self._sessions[session_id]

    def list_sessions(self) -> list[dict]:
        results = []
        storage = Path(self._storage_dir)
        if storage.exists():
            for f in sorted(storage.glob("*.jsonl")):
                sid = f.stem
                session = self.get_or_create(sid)
                results.append({
                    "session_id": sid,
                    "messages": session.message_count,
                    "preview": session.get_preview(),
                    "last_activity": session.get_last_activity(),
                    "created_at": session.get_created_at(),
                    "token_usage": session.get_token_usage(),
                    "persona_id": session.persona_id,
                })
        results.sort(key=lambda x: x.get("last_activity") or 0, reverse=True)
        return results

    def list_sessions_for_persona(self, persona_id: str) -> list[dict]:
        """List sessions filtered by persona_id."""
        return [s for s in self.list_sessions() if s.get("persona_id") == persona_id]

    def get_aggregate_usage(self, since: float | None = None) -> dict:
        """Aggregate token usage across all sessions, optionally filtered by time."""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
        per_session = []
        storage = Path(self._storage_dir)
        if storage.exists():
            for f in sorted(storage.glob("*.jsonl")):
                sid = f.stem
                session = self.get_or_create(sid)
                usage = session.get_token_usage(since=since)
                if usage["api_calls"] > 0:
                    for key in total:
                        total[key] += usage[key]
                    per_session.append({
                        "session_id": sid,
                        "preview": session.get_preview(),
                        **usage,
                    })
        return {"total": total, "sessions": per_session}

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            return True
        path = Path(self._storage_dir) / f"{session_id}.jsonl"
        tasks_path = Path(self._storage_dir) / f"{session_id}.tasks.json"
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if tasks_path.exists():
            tasks_path.unlink()
            deleted = True
        return deleted
