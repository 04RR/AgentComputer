"""Built-in tools for agent_computer."""

from __future__ import annotations
import asyncio
import json
import os
import re
from pathlib import Path

from tool_registry import Tool, ToolParam, ToolRegistry
from memory_search import _parse_markdown_sections


def _is_allowed(name: str, allowed: list[str] | None) -> bool:
    """Check if a tool name is in the allow list. None means allow all."""
    return allowed is None or name in allowed


def register_builtin_tools(registry: ToolRegistry, workspace: str, allowed: list[str] | None = None,
                           tools_config=None) -> None:
    """Register all built-in tools. If allowed is given, only register tools in the list."""

    # Shell security settings from config
    blocked_commands: list[str] = []
    shell_allow_abs_paths: bool = False
    if tools_config is not None:
        blocked_commands = tools_config.shell_blocked_commands
        shell_allow_abs_paths = tools_config.shell_allow_absolute_paths

    workspace_resolved = str(Path(workspace).resolve())

    # ─── Shell Exec ───

    async def shell_exec(command: str, timeout: int = 30) -> str:
        """Execute a shell command and return stdout/stderr."""
        # Check against blocked command patterns
        cmd_lower = command.lower()
        for pattern in blocked_commands:
            if pattern.lower() in cmd_lower:
                return json.dumps({"error": f"Command blocked: matches restricted pattern '{pattern}'"})

        # Check for absolute paths outside workspace
        if not shell_allow_abs_paths:
            # Find path-like strings starting with / (Unix) that aren't the workspace
            abs_paths = re.findall(r'(?:^|\s)(/[a-zA-Z0-9_./-]+)', command)
            for abs_path in abs_paths:
                resolved = str(Path(abs_path).resolve())
                if not resolved.startswith(workspace_resolved):
                    return json.dumps({
                        "error": f"Command references absolute path '{abs_path}' outside the workspace. "
                                 f"Set shell_allow_absolute_paths=true in config to override."
                    })

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            result = {
                "exit_code": proc.returncode,
                "stdout": stdout.decode(errors="replace")[:10000],
                "stderr": stderr.decode(errors="replace")[:5000],
            }
            return json.dumps(result)
        except asyncio.TimeoutError:
            return json.dumps({"error": f"Command timed out after {timeout}s"})

    if _is_allowed("shell", allowed):
        registry.register(Tool(
            name="shell",
            description="Execute a shell command. Use for running scripts, installing packages, system operations. Commands run in the agent workspace directory.",
            params=[
                ToolParam("command", "string", "The shell command to execute"),
                ToolParam("timeout", "integer", "Timeout in seconds (default 30)", required=False),
            ],
            handler=shell_exec,
        ))

    # ─── Read File ───

    async def read_file(path: str, max_lines: int = 500) -> str:
        """Read a file and return its contents."""
        try:
            file_path = _resolve_path(path, workspace)
        except PathTraversalError as e:
            return json.dumps({"error": str(e)})
        try:
            content = file_path.read_text(errors="replace")
            lines = content.splitlines()
            if len(lines) > max_lines:
                return f"[Showing first {max_lines} of {len(lines)} lines]\n" + "\n".join(lines[:max_lines])
            return content
        except Exception as e:
            return json.dumps({"error": str(e)})

    if _is_allowed("read_file", allowed):
        registry.register(Tool(
            name="read_file",
            description="Read the contents of a file. Paths are relative to the workspace.",
            params=[
                ToolParam("path", "string", "File path (relative to workspace or absolute)"),
                ToolParam("max_lines", "integer", "Maximum lines to return (default 500)", required=False),
            ],
            handler=read_file,
        ))

    # ─── Write File ───

    async def write_file(path: str, content: str, mode: str = "overwrite") -> str:
        """Write content to a file."""
        try:
            file_path = _resolve_path(path, workspace)
        except PathTraversalError as e:
            return json.dumps({"error": str(e)})
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                file_path.write_text(content, encoding="utf-8")
            return json.dumps({"status": "ok", "path": str(file_path), "bytes": len(content)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    if _is_allowed("write_file", allowed):
        registry.register(Tool(
            name="write_file",
            description="Write content to a file. Creates parent directories if needed. Paths relative to workspace.",
            params=[
                ToolParam("path", "string", "File path to write to"),
                ToolParam("content", "string", "Content to write"),
                ToolParam("mode", "string", "Write mode: 'overwrite' or 'append'", required=False),
            ],
            handler=write_file,
        ))

    # ─── List Directory ───

    async def list_directory(path: str = ".", max_depth: int = 2) -> str:
        """List directory contents."""
        try:
            dir_path = _resolve_path(path, workspace)
        except PathTraversalError as e:
            return json.dumps({"error": str(e)})
        try:
            if not dir_path.is_dir():
                return json.dumps({"error": f"Not a directory: {path}"})

            entries = []
            _walk(dir_path, entries, dir_path, max_depth, 0)
            return "\n".join(entries) if entries else "(empty directory)"
        except Exception as e:
            return json.dumps({"error": str(e)})

    if _is_allowed("list_directory", allowed):
        registry.register(Tool(
            name="list_directory",
            description="List files and directories. Paths relative to workspace.",
            params=[
                ToolParam("path", "string", "Directory path (default: workspace root)", required=False),
                ToolParam("max_depth", "integer", "Max depth to recurse (default 2)", required=False),
            ],
            handler=list_directory,
        ))


def register_memory_search_tool(registry: ToolRegistry, memory_search_instance, workspace: str,
                                allowed: list[str] | None = None) -> None:
    """Register the memory_search tool. Captures MemorySearch instance via closure."""
    if not _is_allowed("memory_search", allowed):
        return

    async def search_memory(query: str, top_k: int = 5) -> str:
        """Search long-term memory for relevant past knowledge and learnings."""
        if memory_search_instance is None:
            return "Memory search is not enabled. Enable it in config.json under the memory section."

        top_k = max(1, min(top_k, 20))

        # search_failed means the search itself errored and we should fall back.
        # An empty list is a valid result meaning no memories matched.
        search_failed = False
        try:
            results = memory_search_instance.search(query, top_k=top_k)
        except Exception:
            search_failed = True

        if search_failed:
            return _keyword_fallback(query, workspace)

        if not results:
            return "No relevant memories found."

        lines = [f"Found {len(results)} relevant memories:\n"]
        for i, r in enumerate(results, 1):
            content = r.content
            if len(content) > 600:
                content = content[:600] + "..."
            lines.append(f"--- Result {i} (score: {r.score}) ---")
            lines.append(f"Type: {r.source_type} | Title: {r.title}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    registry.register(Tool(
        name="memory_search",
        description=(
            "Search your long-term memory (past session knowledge, learnings, and summaries) "
            "for relevant information. Use this when you need to recall past work, look up "
            "previously discovered facts, API patterns, mistakes learned from, or context "
            "from earlier sessions. Returns the most relevant memory entries ranked by relevance."
        ),
        params=[
            ToolParam("query", "string", "Natural language search query describing what you want to find"),
            ToolParam("top_k", "integer", "Number of results to return (default 5, max 20)", required=False),
        ],
        handler=search_memory,
    ))


def register_task_tool(registry: ToolRegistry, allowed: list[str] | None = None) -> None:
    """Register the manage_tasks tool for deep work mode."""
    if not _is_allowed("manage_tasks", allowed):
        return

    async def manage_tasks(action: str, task_id: int | None = None, title: str | None = None,
                           description: str | None = None, status: str | None = None,
                           parent_id: int | None = None, result: str | None = None,
                           _context: dict | None = None) -> str:
        ctx = _context or {}
        mode = ctx.get("mode", "bounded")
        store = ctx.get("task_store")

        if mode != "deep_work":
            return json.dumps({"error": "Task management not available in bounded mode. Switch to deep work mode to use tasks."})

        if store is None:
            return json.dumps({"error": "No task store available"})

        if action == "create":
            if not title:
                return json.dumps({"error": "title is required for create"})
            task = store.create(title=title, description=description or "", parent_id=parent_id)
            return json.dumps({"status": "created", "task": task.to_dict()})

        elif action == "list":
            tasks = store.to_dict()
            return json.dumps({"tasks": tasks, "summary": store.summary()})

        elif action == "update":
            if task_id is None:
                return json.dumps({"error": "task_id is required for update"})
            kwargs = {}
            if title is not None:
                kwargs["title"] = title
            if description is not None:
                kwargs["description"] = description
            if status is not None:
                kwargs["status"] = status
            task = store.update(task_id, **kwargs)
            if not task:
                return json.dumps({"error": f"Task {task_id} not found"})
            return json.dumps({"status": "updated", "task": task.to_dict()})

        elif action == "complete":
            if task_id is None:
                return json.dumps({"error": "task_id is required for complete"})
            complete_result = store.complete(task_id, result=result or "")
            if complete_result is None:
                return json.dumps({"error": f"Task {task_id} not found"})
            if not complete_result.success:
                return json.dumps({
                    "error": f"Cannot complete task {task_id} — it has {len(complete_result.blocked_by)} "
                             f"incomplete subtask(s) (IDs: {complete_result.blocked_by}). "
                             f"Complete all subtasks first, then complete the parent.",
                    "task": complete_result.task.to_dict(),
                    "incomplete_subtasks": complete_result.blocked_by,
                })
            return json.dumps({"status": "completed", "task": complete_result.task.to_dict()})

        elif action == "delete":
            if task_id is None:
                return json.dumps({"error": "task_id is required for delete"})
            if store.delete(task_id):
                return json.dumps({"status": "deleted", "task_id": task_id})
            return json.dumps({"error": f"Task {task_id} not found"})

        else:
            return json.dumps({"error": f"Unknown action: {action}. Use: create, list, update, complete, delete"})

    registry.register(Tool(
        name="manage_tasks",
        description="Manage tasks for tracking progress in deep work mode. Actions: create, list, update, complete, delete.",
        params=[
            ToolParam("action", "string", "Action to perform: create, list, update, complete, delete"),
            ToolParam("task_id", "integer", "Task ID (required for update/complete/delete)", required=False),
            ToolParam("title", "string", "Task title (required for create)", required=False),
            ToolParam("description", "string", "Task description", required=False),
            ToolParam("status", "string", "Task status: pending, in_progress, completed, blocked", required=False),
            ToolParam("parent_id", "integer", "Parent task ID for subtasks", required=False),
            ToolParam("result", "string", "Result summary (for complete action)", required=False),
        ],
        handler=manage_tasks,
    ))


# ─── Helpers ───

class PathTraversalError(Exception):
    """Raised when a path resolves outside the workspace."""


def _resolve_path(path: str, workspace: str) -> Path:
    """Resolve a path and verify it is inside the workspace.

    Both relative and absolute paths are accepted, but the resolved result
    must be within the workspace directory. Raises PathTraversalError otherwise.
    """
    workspace_root = Path(workspace).resolve()
    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (workspace_root / p).resolve()
    if not resolved.is_relative_to(workspace_root):
        raise PathTraversalError(
            f"Path '{path}' resolves to '{resolved}' which is outside the workspace '{workspace_root}'"
        )
    return resolved


def _walk(dir_path: Path, entries: list, root: Path, max_depth: int, depth: int) -> None:
    """Recursively walk directory up to max_depth."""
    if depth > max_depth:
        return
    try:
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith(".") or item.name == "node_modules" or item.name == "__pycache__":
                continue
            rel = item.relative_to(root)
            prefix = "  " * depth
            if item.is_dir():
                entries.append(f"{prefix}{rel}/")
                _walk(item, entries, root, max_depth, depth + 1)
            else:
                size = item.stat().st_size
                entries.append(f"{prefix}{rel}  ({_human_size(size)})")
    except PermissionError:
        pass


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024  # type: ignore
    return f"{size:.1f}TB"


def _keyword_fallback(query: str, workspace: str) -> str:
    """Simple keyword scan of knowledge.md and learnings.md when MemorySearch fails."""
    memory_dir = Path(workspace) / "memory"
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return "No relevant memories found."

    matches: list[tuple[int, str, str, str]] = []
    for fname in ("knowledge.md", "learnings.md"):
        path = memory_dir / fname
        if not path.exists():
            continue
        for title, body in _parse_markdown_sections(path.read_text(encoding="utf-8")):
            full = (title + " " + body).lower()
            score = sum(1 for t in terms if t in full)
            if score > 0:
                matches.append((score, title, body[:400], fname))

    if not matches:
        return "No relevant memories found."

    matches.sort(reverse=True)
    lines = [f"Found {len(matches)} matching memories (keyword scan):\n"]
    for score, title, body, source in matches[:10]:
        lines.append(f"--- [{source}] {title} ---")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
