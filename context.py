"""Context assembly for agent_computer.

build_system_prompt is a pure function — all file I/O happens in the caller.
load_static_context reads workspace files once per run for caching.
"""

from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("agent_computer.context")

DEEP_WORK_INSTRUCTIONS = """
## Deep Work Mode — MANDATORY WORKFLOW

You are in **deep work mode**. You MUST follow this exact workflow. Do NOT skip steps.

### PHASE 1: PLAN (mandatory — do this BEFORE any real work)

Your FIRST tool calls MUST be `manage_tasks` calls to decompose the request. Do NOT call any
other tool (shell, read_file, web_fetch, etc.) until you have created a full task breakdown.

**Decomposition rules:**
- Create ONE parent task for the overall goal
- Create SEPARATE subtasks for each distinct unit of work (one per item, one per file, one per entity)
- Each subtask should be completable in 1-5 tool calls — if it needs more, split it further
- Subtasks must be specific and actionable, not vague summaries of the whole request

**Example — if asked "Research missing annual reports for the first 3 tickers and save to files":**

```
manage_tasks(action="create", title="Research missing annual reports for first 3 tickers")  → parent id=1
manage_tasks(action="create", title="Identify missing data years for ABB", parent_id=1)     → id=2
manage_tasks(action="create", title="Identify missing data years for ADANIENT", parent_id=1) → id=3
manage_tasks(action="create", title="Identify missing data years for ADANIGREEN", parent_id=1) → id=4
manage_tasks(action="create", title="Research ABB missing data & save files", parent_id=1)   → id=5
manage_tasks(action="create", title="Research ADANIENT missing data & save files", parent_id=1) → id=6
manage_tasks(action="create", title="Research ADANIGREEN missing data & save files", parent_id=1) → id=7
```

BAD decomposition (too coarse — do NOT do this):
```
manage_tasks(action="create", title="Research missing data for first 3 tickers")  → only 1 task!
```

### PHASE 2: EXECUTE (work through subtasks one by one)

- Before starting a subtask: `manage_tasks(action="update", task_id=X, status="in_progress")`
- Do the actual work (shell, read_file, web_fetch, write_file, etc.)
- After finishing: `manage_tasks(action="complete", task_id=X, result="brief summary of what was done")`
- Move to the next pending subtask

### PHASE 3: REVIEW (after all subtasks done)

- `manage_tasks(action="list")` — verify everything is completed
- Self-check your work: re-read files you created, verify outputs
- Mark the parent task complete with a summary
- Then give your final response to the user

### Rules:
- NEVER do real work before creating your task breakdown
- NEVER lump multiple items/entities into a single task — one task per item
- Review the task list every 5-10 iterations to reorient
- If blocked on a task, mark it "blocked" and move to the next one

### CRITICAL RULES — NEVER VIOLATE THESE:
- **NEVER ask "Would you like me to proceed?" or "Shall I continue?" or "What would you like me to do next?"** — you are FULLY autonomous. Keep working until ALL tasks are done.
- **NEVER stop to summarize progress mid-way** — just keep executing tasks. Only give a final summary when ALL tasks are done.
- **NEVER re-analyze data you already processed** — check your task list. If a task is "completed", its work is done. Move to the next "pending" task.
- **NEVER complete a parent task before ALL its subtasks are completed** — the system will block this. Complete subtasks first, then the parent.
- **NEVER produce a text-only response while tasks remain** — if you have pending tasks, your next action MUST be a tool call. Text-only responses waste iterations.
- If you return text without calling any tool, the loop will continue automatically. Use tools to make progress, not words.
- Do NOT confirm before destructive operations — in deep work mode you have full autonomy to act.
- After completing one subtask, IMMEDIATELY start the next pending subtask. Do not pause, summarize, or ask.
"""


def load_static_context(workspace: str) -> dict:
    """Read static workspace files once per run. Returns dict to unpack into build_system_prompt.

    Keys: soul_content, user_content, static_memory_fallback
    """
    workspace_path = Path(workspace)
    result = {"soul_content": "", "user_content": "", "static_memory_fallback": ""}

    for fname, key in [("SOUL.md", "soul_content"), ("USER.md", "user_content")]:
        path = workspace_path / fname
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                result[key] = content
                logger.debug(f"Loaded {fname} ({len(content)} chars)")

    # Static memory fallback (used when MemorySearch is unavailable)
    memory_dir = workspace_path / "memory"
    parts = []
    for filename, label in [("knowledge.md", "Agent Knowledge"), ("learnings.md", "Agent Learnings")]:
        file_path = memory_dir / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                if len(content) > 2000:
                    content = content[:2000] + "\n...[use read_file for full content]"
                tag = label.lower().replace(" ", "_")
                parts.append(f"<{tag}>\n{content}\n</{tag}>")
    result["static_memory_fallback"] = "\n\n".join(parts)

    return result


def build_system_prompt(
    workspace: str,
    agent_name: str,
    mode: str = "bounded",
    task_summary: str = "",
    budget_warning: str = "",
    pending_task_count: int = 0,
    context_file: str = "",
    relevant_memories: list[dict] | None = None,
    user_message: str = "",
    tool_names: list[str] | None = None,
    session_summary: str = "",
    soul_content: str = "",
    user_content: str = "",
    static_memory_fallback: str = "",
) -> str:
    """Assemble the system prompt. Pure function — no file I/O.

    Assembly order:
    1. Base identity
    2. SOUL.md content
    3. USER.md content
    4. Tool inventory
    5. Relevant memories (or static fallback)
    6. Session context
    7. Deep work instructions (if deep_work mode)
    8. Task state
    9. Budget warning
    10. Archived context
    """
    parts: list[str] = []

    # 1. Base identity (lean — detailed rules belong in SOUL.md)
    now = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")
    parts.append(f"""You are {agent_name}, a personal AI assistant powered by agent_computer.
Current date/time: {now}
Workspace directory: {workspace}

You are an autonomous agent running in an agentic loop with tool access.
Think step-by-step. Be concise. If a tool call fails, try an alternative approach.""")

    # 2. SOUL.md — personality, rules, boundaries
    if soul_content:
        parts.append(f"<agent_identity_&_rules>\n{soul_content}\n</agent_identity_&_rules>")

    # 3. USER.md — user context, preferences, project info
    if user_content:
        parts.append(f"<user_context>\n{user_content}\n</user_context>")

    # 4. Tool inventory — natural language summary of capabilities
    if tool_names:
        tool_list = ", ".join(tool_names)
        parts.append(
            f"<available_tools>\nYou have these tools: {tool_list}\n"
            "Use the appropriate tool for each task. If a tool call fails, try an alternative.\n"
            "</available_tools>"
        )

    # 5. Relevant memories (query-aware) or static fallback
    if relevant_memories:
        mem_lines = []
        for m in relevant_memories:
            content = m.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            mem_lines.append(
                f"- [{m.get('source_type', '?')}] **{m.get('title', 'Untitled')}**: {content}"
            )
        parts.append(
            "<relevant_memories>\n"
            "The following memories may be relevant:\n"
            + "\n".join(mem_lines)
            + "\n</relevant_memories>"
        )
    elif static_memory_fallback:
        parts.append(static_memory_fallback)

    # 6. Session context — helps maintain coherence in long conversations
    if session_summary:
        parts.append(f"<session_context>\nThis session so far: {session_summary}\n</session_context>")

    # 7. Deep work instructions (unchanged)
    if mode == "deep_work":
        parts.append(DEEP_WORK_INSTRUCTIONS)

        # 8. Task state
        if task_summary:
            resume_hint = ""
            if pending_task_count > 0:
                resume_hint = (
                    f"\n\nYou have {pending_task_count} task(s) still to do. "
                    "Pick up the next pending task immediately — call a tool, do NOT produce a text-only response."
                )
            parts.append(f"<current_tasks>\n{task_summary}{resume_hint}\n</current_tasks>")

        # 9. Budget warning
        if budget_warning:
            parts.append(f"<budget_warning>\n{budget_warning}\n</budget_warning>")

        # 10. Archived context
        if context_file:
            parts.append(
                f"<archived_context>\n"
                f"Your conversation was auto-compacted to save tokens. "
                f"Full prior context is saved at:\n{context_file}\n"
                f"Use `read_file` to review earlier work if needed. "
                f"Your task list is the source of truth for progress.\n"
                f"</archived_context>"
            )

    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4
