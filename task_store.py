"""Task management for deep work mode.

Provides a Task dataclass and TaskStore for creating, tracking,
and persisting tasks across agent iterations. Each session gets
its own TaskStore backed by a JSON file.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_computer.task_store")


@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: str = "pending"  # pending, in_progress, completed, blocked
    parent_id: int | None = None
    result: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


@dataclass
class CompleteResult:
    """Outcome of a TaskStore.complete() call."""
    task: Task
    success: bool
    blocked_by: list[int] = field(default_factory=list)


class TaskStore:
    """In-memory task store with JSON file persistence.

    Each mutation auto-saves to disk so state survives restarts.
    Caches sorted list, status counts, and summary text; invalidated on mutation.
    Supports deferred saves via _auto_save flag for batch operations.
    """

    VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}

    def __init__(self, persistence_path: str | Path):
        self._path = Path(persistence_path)
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        # Caches
        self._sorted_cache: list[Task] | None = None
        self._summary_cache: str | None = None
        self._status_counts: dict[str, int] = {s: 0 for s in self.VALID_STATUSES}
        # Deferred save support
        self._dirty: bool = False
        self._auto_save: bool = True
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for td in data.get("tasks", []):
                task = Task(**td)
                self._tasks[task.id] = task
            self._next_id = data.get("next_id", 1)
            # Rebuild status counts from loaded tasks
            self._status_counts = {s: 0 for s in self.VALID_STATUSES}
            for t in self._tasks.values():
                self._status_counts[t.status] = self._status_counts.get(t.status, 0) + 1
            logger.debug(f"Loaded {len(self._tasks)} tasks from {self._path}")
        except Exception as e:
            logger.error(f"Failed to load task store: {e}")

    def _invalidate_caches(self):
        self._sorted_cache = None
        self._summary_cache = None

    def _maybe_save(self):
        if self._auto_save:
            self._save()
        else:
            self._dirty = True

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._dirty = False

    def flush(self):
        """Write pending changes to disk if dirty."""
        if self._dirty:
            self._save()

    def create(self, title: str, description: str = "", parent_id: int | None = None):
        task = Task(
            id=self._next_id,
            title=title,
            description=description,
            parent_id=parent_id,
        )
        self._tasks[task.id] = task
        self._next_id += 1
        self._status_counts[task.status] = self._status_counts.get(task.status, 0) + 1
        self._invalidate_caches()
        self._maybe_save()
        logger.info(f"Task created: [{task.id}] {task.title}")
        return task

    def update(self, task_id: int, **kwargs) -> Task | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        old_status = task.status
        for key, value in kwargs.items():
            if key == "status" and value not in self.VALID_STATUSES:
                continue
            if hasattr(task, key) and key not in ("id", "created_at"):
                setattr(task, key, value)
        # Update status counts if status changed
        if task.status != old_status:
            self._status_counts[old_status] = max(0, self._status_counts.get(old_status, 0) - 1)
            self._status_counts[task.status] = self._status_counts.get(task.status, 0) + 1
        self._invalidate_caches()
        self._maybe_save()
        return task

    def complete(self, task_id: int, result: str = "") -> CompleteResult | None:
        task = self._tasks.get(task_id)
        if not task:
            return None

        # Guard: don't complete a parent task if it has incomplete children
        children = [t for t in self._tasks.values() if t.parent_id == task_id]
        incomplete = [c for c in children if c.status not in ("completed",)]
        if incomplete:
            titles = ", ".join(f"[{c.id}] {c.title}" for c in incomplete[:5])
            logger.warning(
                f"Cannot complete task {task_id}: {len(incomplete)} incomplete "
                f"subtask(s): {titles}"
            )
            return CompleteResult(task=task, success=False, blocked_by=[c.id for c in incomplete])

        self.update(task_id, status="completed", result=result)
        return CompleteResult(task=task, success=True)

    def delete(self, task_id: int) -> bool:
        task = self._tasks.get(task_id)
        if task:
            self._status_counts[task.status] = max(0, self._status_counts.get(task.status, 0) - 1)
            del self._tasks[task_id]
            self._invalidate_caches()
            self._maybe_save()
            return True
        return False

    def get(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[Task]:
        if self._sorted_cache is None:
            self._sorted_cache = sorted(self._tasks.values(), key=lambda t: t.id)
        return self._sorted_cache

    def pending_count(self) -> int:
        return self._status_counts.get("pending", 0) + self._status_counts.get("in_progress", 0)

    def completed_list(self) -> list[Task]:
        return [t for t in self.list_all() if t.status == "completed"]

    def summary(self) -> str:
        """Compact text summary for system prompt injection. Cached until mutation."""
        if self._summary_cache is not None:
            return self._summary_cache

        tasks = self.list_all()
        if not tasks:
            self._summary_cache = ""
            return ""

        parts = []
        for status in ("pending", "in_progress", "completed", "blocked"):
            c = self._status_counts.get(status, 0)
            if c > 0:
                parts.append(f"{c} {status}")

        lines = [f"Tasks: {', '.join(parts)} ({len(tasks)} total)"]

        # Build tree: top-level first, then children
        top_level = [t for t in tasks if t.parent_id is None]
        children_map: dict[int, list[Task]] = {}
        for t in tasks:
            if t.parent_id is not None:
                children_map.setdefault(t.parent_id, []).append(t)

        for t in top_level:
            lines.append(f"  [{t.id}] [{t.status}] {t.title}")
            for child in children_map.get(t.id, []):
                lines.append(f"    [{t.id}.{child.id}] [{child.status}] {child.title}")

        self._summary_cache = "\n".join(lines)
        return self._summary_cache

    def to_dict(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.list_all()]

    def clear(self) -> None:
        self._tasks.clear()
        self._next_id = 1
        self._status_counts = {s: 0 for s in self.VALID_STATUSES}
        self._invalidate_caches()
        if self._path.exists():
            self._path.unlink()

    def delete_file(self) -> None:
        if self._path.exists():
            self._path.unlink()
