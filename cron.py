"""Cron scheduler for agent_computer.

Runs inside the Gateway process as a background asyncio task.
Jobs are defined in workspace/cron.json and each fires a prompt
into a dedicated session on a schedule.

This mirrors OpenClaw's heartbeat system — the agent wakes up
on a schedule, performs a task, and goes back to sleep.

Schedule format uses simplified cron expressions:
  "every 30m"       → every 30 minutes
  "every 2h"        → every 2 hours
  "every 1d"        → every day (24h)
  "daily 09:00"     → every day at 09:00 UTC
  "daily 17:30"     → every day at 17:30 UTC
  "hourly :15"      → every hour at minute 15
  "startup"         → run once when the gateway starts
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent import AgentRuntime
    from session import SessionManager
    from persona import PersonaStore

logger = logging.getLogger("agent_computer.cron")

CRON_FILE = "cron.json"


@dataclass
class CronJob:
    """A single scheduled job."""
    id: str
    name: str
    schedule: str
    prompt: str
    session_id: str | None = None  # defaults to "cron-{id}"
    enabled: bool = True
    last_run: float | None = None
    next_run: float | None = None
    run_count: int = 0
    persona_id: str = "default"


class CronScheduler:
    """Lightweight in-process scheduler that triggers agent runs on a schedule.

    Jobs are loaded from workspace/cron.json. Each job fires its prompt
    into a dedicated cron session, so cron output doesn't pollute
    interactive conversations.
    """

    def __init__(
        self,
        workspace: str,
        agent: AgentRuntime,
        session_mgr: SessionManager,
    ):
        self.workspace = workspace
        self.agent = agent
        self.session_mgr = session_mgr
        self.jobs: dict[str, CronJob] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.persona_store: PersonaStore | None = None
        self._get_memory_search = None  # callback: (persona_id) -> MemorySearch | None

    def load_jobs(self) -> int:
        """Load jobs from workspace/cron.json and persona cron files. Returns count of jobs loaded."""
        self.jobs.clear()

        # Load root cron.json
        cron_path = Path(self.workspace) / CRON_FILE
        if cron_path.exists():
            self._load_cron_file(cron_path, persona_id="default")
        else:
            logger.info(f"No {CRON_FILE} found in workspace — root cron disabled")

        # Load persona cron jobs
        if self.persona_store:
            for persona in self.persona_store.list_all():
                if persona.id == "default" or not persona.enabled or not persona.cron_enabled:
                    continue
                p_cron_path = Path(persona.workspace_path) / CRON_FILE
                if p_cron_path.exists():
                    self._load_cron_file(p_cron_path, persona_id=persona.id)

        return len(self.jobs)

    def _load_cron_file(self, cron_path: Path, persona_id: str = "default") -> None:
        """Load jobs from a single cron.json file."""
        try:
            with open(cron_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {cron_path}: {e}")
            return

        jobs_data = raw if isinstance(raw, list) else raw.get("jobs", [])

        for entry in jobs_data:
            raw_id = entry.get("id", f"job-{len(self.jobs)}")
            # Namespace persona job IDs to prevent collisions
            job_id = f"{persona_id}:{raw_id}" if persona_id != "default" else raw_id
            session_id = entry.get("session_id", f"cron-{job_id}")
            job = CronJob(
                id=job_id,
                name=entry.get("name", raw_id),
                schedule=entry.get("schedule", "every 1h"),
                prompt=entry.get("prompt", ""),
                session_id=session_id,
                enabled=entry.get("enabled", True),
                persona_id=persona_id,
            )
            if job.prompt and job.enabled:
                job.next_run = _compute_next_run(job.schedule)
                self.jobs[job_id] = job
                logger.info(
                    f"Loaded cron job: {job.name} [{job.schedule}] "
                    f"(persona={persona_id}) → next run at {_fmt_ts(job.next_run)}"
                )

    def start(self) -> None:
        """Start the scheduler background task."""
        if not self.jobs:
            logger.info("No cron jobs to schedule")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Cron scheduler started with {len(self.jobs)} job(s)")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Cron scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop — checks every 15 seconds for due jobs."""
        # Handle "startup" jobs immediately
        for job in self.jobs.values():
            if job.schedule.strip().lower() == "startup":
                await self._run_job(job)

        while not self._stop.is_set():
            try:
                now = time.time()
                for job in list(self.jobs.values()):
                    if not job.enabled or job.schedule.strip().lower() == "startup":
                        continue
                    if job.next_run and now >= job.next_run:
                        await self._run_job(job)
                        job.next_run = _compute_next_run(job.schedule)
                        logger.debug(
                            f"Job {job.name}: next run at {_fmt_ts(job.next_run)}"
                        )

                # Sleep 15s between checks (responsive but not wasteful)
                await asyncio.sleep(15)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _run_job(self, job: CronJob) -> None:
        """Execute a single cron job by sending its prompt to the agent."""
        session_id = job.session_id or f"cron-{job.id}"
        logger.info(f"⏰ Running cron job: {job.name} → session={session_id} (persona={job.persona_id})")

        session = self.session_mgr.get_or_create(session_id, persona_id=job.persona_id)

        # Resolve persona and memory search for non-default personas
        persona = None
        ms_override = None
        if self.persona_store and job.persona_id != "default":
            persona = self.persona_store.get(job.persona_id)
        if self._get_memory_search:
            ms_override = self._get_memory_search(job.persona_id)

        try:
            async with session.lock:
                response = await self.agent.run_simple(
                    session, job.prompt, persona=persona, memory_search_override=ms_override
                )

            job.last_run = time.time()
            job.run_count += 1

            # Log a preview of the response
            preview = response[:200].replace("\n", " ")
            logger.info(
                f"✅ Cron job done: {job.name} (run #{job.run_count}) → {preview}..."
                if len(response) > 200 else
                f"✅ Cron job done: {job.name} (run #{job.run_count}) → {preview}"
            )

        except Exception as e:
            logger.error(f"❌ Cron job failed: {job.name} — {e}")

    def get_status(self) -> list[dict]:
        """Get status of all jobs for the API."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "schedule": job.schedule,
                "enabled": job.enabled,
                "session_id": job.session_id,
                "last_run": _fmt_ts(job.last_run) if job.last_run else None,
                "next_run": _fmt_ts(job.next_run) if job.next_run else None,
                "run_count": job.run_count,
                "prompt_preview": job.prompt[:100],
                "persona_id": job.persona_id,
            }
            for job in self.jobs.values()
        ]


# ─── Schedule Parsing ───

def _compute_next_run(schedule: str) -> float:
    """Parse a schedule string and return the next run timestamp."""
    s = schedule.strip().lower()
    now = datetime.now(timezone.utc)

    # "startup" — already handled, return far future
    if s == "startup":
        return float("inf")

    # "every Xm" / "every Xh" / "every Xd"
    m = re.match(r"every\s+(\d+)\s*(m|min|minutes?|h|hours?|d|days?|s|seconds?)", s)
    if m:
        val = int(m.group(1))
        unit = m.group(2)[0]
        seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return time.time() + val * seconds

    # "daily HH:MM"
    m = re.match(r"daily\s+(\d{1,2}):(\d{2})", s)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    # "hourly :MM"
    m = re.match(r"hourly\s+:(\d{2})", s)
    if m:
        minute = int(m.group(1))
        target = now.replace(minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(hours=1)
        return target.timestamp()

    # Fallback: treat as "every 1h"
    logger.warning(f"Unrecognized schedule '{schedule}', defaulting to every 1h")
    return time.time() + 3600


def _fmt_ts(ts: float | None) -> str | None:
    """Format a timestamp for display."""
    if ts is None or ts == float("inf"):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
