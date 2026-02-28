"""Auto-reflection engine for agent_computer.

Processes completed sessions to extract knowledge, learnings, and reusable
skills. Runs at startup to build cross-session memory.
"""

from __future__ import annotations
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("agent_computer.reflection")

REFLECTION_SYSTEM_PROMPT = """\
You are a reflection engine that analyzes completed AI agent sessions.
Your job is to extract useful information from the session transcript.

Extract ONLY what is genuinely useful. Most sessions produce 0-3 items total.

Rules:
- **Knowledge**: Facts, API endpoints, configurations, working URLs, credentials patterns discovered.
  Only include if the information was confirmed to work (not speculative).
- **Learnings**: Mistakes the agent made and then corrected. Only include if there was a clear
  error → correction pattern. Include the mistake AND the fix so future sessions avoid the error.
- **Skills**: Reusable code patterns. Only extract if code was refined through trial and error
  and reached a working state. The code must be a standalone async function that could be called
  as a tool. Do NOT extract trivial code or one-liners.

Be highly selective. Quality over quantity.
"""

REFLECTION_USER_PROMPT = """\
Analyze this session transcript and extract useful items.

Respond with valid JSON matching this schema:
{
  "session_summary": "one sentence describing what happened",
  "knowledge": [
    {"topic": "short topic name", "content": "the fact/config/endpoint discovered", "confidence": "high|medium|low"}
  ],
  "learnings": [
    {"title": "short title", "mistake": "what went wrong", "correction": "the fix that worked", "category": "api|auth|parsing|config|other"}
  ],
  "skills": [
    {"name": "snake_case_name", "description": "what it does", "code": "async def skill_name(param: str) -> str:\\n    ...", "dependencies": ["httpx"]}
  ]
}

If nothing useful was found, return empty arrays.

--- SESSION TRANSCRIPT ---
%s
"""


class ReflectionEngine:
    """Processes sessions to extract cross-session memory."""

    def __init__(self, workspace: str, client: AsyncOpenAI, model_id: str, max_tokens: int = 4096, provider: str = "", memory_search=None):
        self.workspace = Path(workspace)
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.provider = provider
        self.memory_search = memory_search
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / "index.json"

    async def process_session(self, session_id: str, messages: list[dict]) -> dict:
        """Process a single session and extract knowledge/learnings/skills.

        Returns the index entry for this session.
        """
        condensed = self._condense_session(session_id, messages)

        # Skip trivially short sessions
        real_count = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
        if real_count < 4:
            entry = {
                "session_id": session_id,
                "processed_at": time.time(),
                "status": "too_short",
                "message_count": real_count,
            }
            self._update_index(session_id, entry)
            logger.info(f"Skipped session {session_id}: too short ({real_count} messages)")
            return entry

        try:
            result, tokens_used = await self._call_reflection_llm(condensed)
        except Exception as e:
            entry = {
                "session_id": session_id,
                "processed_at": time.time(),
                "status": "error",
                "error": str(e),
            }
            self._update_index(session_id, entry)
            logger.error(f"Reflection failed for session {session_id}: {e}")
            return entry

        # Persist extracted items
        knowledge_count = 0
        learnings_count = 0
        skills_count = 0

        if result.get("knowledge"):
            self._append_knowledge(result["knowledge"], session_id)
            knowledge_count = len(result["knowledge"])

        if result.get("learnings"):
            self._append_learnings(result["learnings"], session_id)
            learnings_count = len(result["learnings"])

        if result.get("skills"):
            self._save_skills(result["skills"], session_id)
            skills_count = len(result["skills"])

        # Index new entries in memory search for immediate availability
        if self.memory_search:
            await self._index_extracted(result, session_id)

        entry = {
            "session_id": session_id,
            "processed_at": time.time(),
            "status": "processed",
            "summary": result.get("session_summary", ""),
            "knowledge_count": knowledge_count,
            "learnings_count": learnings_count,
            "skills_count": skills_count,
            "tokens_used": tokens_used,
        }
        self._update_index(session_id, entry)
        logger.info(
            f"Reflected on session {session_id}: "
            f"{knowledge_count} knowledge, {learnings_count} learnings, {skills_count} skills"
        )
        return entry

    def _condense_session(self, session_id: str, messages: list[dict]) -> str:
        """Build a compact markdown representation of the session for reflection."""
        lines = [f"# Session: {session_id}", ""]
        total_chars = 0
        max_total = 30_000

        for msg in messages:
            role = msg.get("role", "")

            # Skip meta messages entirely
            if role == "meta":
                continue

            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = content[:2000]
                else:
                    content = json.dumps(content)[:2000]
                chunk = f"**User**: {content}\n"

            elif role == "assistant":
                content = msg.get("content", "")
                if isinstance(content, dict):
                    text = content.get("content", "") or ""
                    text = text[:1000]
                    tool_calls = content.get("tool_calls", [])
                    tc_parts = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = fn.get("arguments", "")[:200]
                        tc_parts.append(f"  → {name}({args})")
                    tc_str = "\n".join(tc_parts)
                    chunk = f"**Assistant**: {text}\n{tc_str}\n" if text else f"**Assistant** [tool calls]:\n{tc_str}\n"
                else:
                    content = str(content)[:1000] if content else ""
                    chunk = f"**Assistant**: {content}\n"

            elif role == "tool":
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content)
                content = content[:300]
                tool_name = msg.get("tool_name", "tool")
                chunk = f"**Tool ({tool_name})**: {content}\n"

            else:
                continue

            total_chars += len(chunk)
            if total_chars > max_total:
                lines.append("...[truncated — session too long]...")
                break
            lines.append(chunk)

        return "\n".join(lines)

    async def _call_reflection_llm(self, condensed: str) -> tuple[dict, int]:
        """Call the LLM with the reflection prompt and parse the JSON response."""
        user_prompt = REFLECTION_USER_PROMPT % condensed

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.3,
        }

        # Only use JSON mode for providers that support it (not LM Studio)
        if self.provider not in ("lmstudio", ""):
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or "{}"
        tokens_used = response.usage.total_tokens if response.usage else 0

        # Parse JSON — handle markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            # Strip ```json ... ```
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reflection JSON: {e}")
            result = {"session_summary": "Parse error", "knowledge": [], "learnings": [], "skills": []}

        return result, tokens_used

    # ── Memory search indexing ──

    async def _index_extracted(self, result: dict, session_id: str) -> None:
        """Index newly extracted knowledge/learnings/summary in MemorySearch."""
        for item in result.get("knowledge", []):
            topic = item.get("topic", "Unknown")
            content = item.get("content", "")
            slug = re.sub(r"[\s-]+", "_", re.sub(r"[^a-z0-9\s-]", "", topic.lower().strip()))[:80]
            sid = slug or f"knowledge_{session_id}"
            try:
                await self.memory_search.async_index_text("knowledge", sid, topic, content)
            except Exception as e:
                logger.warning(f"Failed to index knowledge '{topic}': {e}")

        for item in result.get("learnings", []):
            title = item.get("title", "Untitled")
            content = f"Mistake: {item.get('mistake', '')} Correction: {item.get('correction', '')}"
            slug = re.sub(r"[\s-]+", "_", re.sub(r"[^a-z0-9\s-]", "", title.lower().strip()))[:80]
            sid = slug or f"learning_{session_id}"
            try:
                await self.memory_search.async_index_text("learning", sid, title, content)
            except Exception as e:
                logger.warning(f"Failed to index learning '{title}': {e}")

        summary = result.get("session_summary", "")
        if summary:
            try:
                await self.memory_search.async_index_text(
                    "session_summary", session_id, f"Session: {session_id}", summary
                )
            except Exception as e:
                logger.warning(f"Failed to index session summary: {e}")

    # ── Storage methods ──

    def _append_knowledge(self, items: list[dict], session_id: str) -> None:
        """Append knowledge items to knowledge.md."""
        path = self.memory_dir / "knowledge.md"
        lines = []
        if not path.exists():
            lines.append("# Agent Knowledge\n")
        for item in items:
            topic = item.get("topic", "Unknown")
            content = item.get("content", "")
            confidence = item.get("confidence", "medium")
            lines.append(f"\n## {topic}")
            lines.append(f"*Source: session {session_id} | Confidence: {confidence}*\n")
            lines.append(content)
            lines.append("")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _append_learnings(self, items: list[dict], session_id: str) -> None:
        """Append learning items to learnings.md."""
        path = self.memory_dir / "learnings.md"
        lines = []
        if not path.exists():
            lines.append("# Agent Learnings\n")
        for item in items:
            title = item.get("title", "Untitled")
            mistake = item.get("mistake", "")
            correction = item.get("correction", "")
            category = item.get("category", "other")
            lines.append(f"\n## {title}")
            lines.append(f"*Source: session {session_id} | Category: {category}*\n")
            lines.append(f"**Mistake**: {mistake}\n")
            lines.append(f"**Correction**: {correction}")
            lines.append("")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_skills(self, items: list[dict], session_id: str) -> None:
        """Save skill code as individual .py files in memory/skills/."""
        skills_dir = self.memory_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            name = item.get("name", "").strip()
            if not name:
                continue
            # Sanitize filename
            name = re.sub(r"[^a-z0-9_]", "_", name.lower())
            skill_path = skills_dir / f"{name}.py"
            if skill_path.exists():
                logger.info(f"Skill {name} already exists, skipping")
                continue
            description = item.get("description", "")
            code = item.get("code", "")
            deps = item.get("dependencies", [])
            content = f'"""{description}\n\nAuto-extracted from session {session_id}.\nDependencies: {", ".join(deps) if deps else "none"}\n"""\n\n{code}\n'
            skill_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved skill: {skill_path}")

    # ── Index management ──

    def _load_index(self) -> dict:
        """Load the session processing index."""
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {"sessions": {}}
        return {"sessions": {}}

    def _save_index(self, index: dict) -> None:
        """Save the session processing index."""
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _update_index(self, session_id: str, entry: dict) -> None:
        """Add or update a session entry in the index."""
        index = self._load_index()
        index["sessions"][session_id] = entry
        self._save_index(index)

    def is_processed(self, session_id: str) -> bool:
        """Check if a session has already been processed."""
        index = self._load_index()
        return session_id in index.get("sessions", {})

    def get_unprocessed(self, all_ids: list[str]) -> list[str]:
        """Filter session IDs to only unprocessed ones, excluding cron/reflection sessions."""
        index = self._load_index()
        processed = set(index.get("sessions", {}).keys())
        result = []
        for sid in all_ids:
            if sid in processed:
                continue
            if sid.startswith("cron-") or sid.startswith("reflection-"):
                continue
            result.append(sid)
        return result

    def load_memory_for_prompt(self) -> str:
        """Load knowledge + learnings for system prompt injection."""
        parts = []
        for filename, label in [("knowledge.md", "Agent Knowledge"), ("learnings.md", "Agent Learnings")]:
            path = self.memory_dir / filename
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    if len(content) > 2000:
                        content = content[:2000] + "\n...[use read_file for full content]"
                    tag = label.lower().replace(" ", "_")
                    parts.append(f"<{tag}>\n{content}\n</{tag}>")
        return "\n\n".join(parts)
