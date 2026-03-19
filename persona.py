"""Persona system for agent_computer.

Named agents that each get their own workspace, personality, cron jobs,
and memory, while sharing the user's USER.md. The "default" persona
maps to the current system with zero migration.
"""

from __future__ import annotations
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("agent_computer.persona")


@dataclass
class Persona:
    """A named agent persona with its own workspace and configuration."""
    id: str
    name: str
    description: str
    enabled: bool = True
    tools_allow: list[str] | None = None
    tools_deny: list[str] | None = None
    model_override: str | None = None
    cron_enabled: bool = True
    created_at: float = field(default_factory=time.time)
    workspace_path: str = ""
    user_md_path: str = ""


class PersonaStore:
    """Manages persona creation, loading, and lifecycle."""

    def __init__(self, root_workspace: str):
        self.root_workspace = root_workspace
        self.personas_dir = Path(root_workspace) / "personas"
        self.personas_dir.mkdir(parents=True, exist_ok=True)

    def get_default(self) -> Persona:
        """Return the default persona mapped to the root workspace."""
        soul_path = Path(self.root_workspace) / "SOUL.md"
        name = "Default"
        description = "Default agent persona"
        if soul_path.exists():
            content = soul_path.read_text(encoding="utf-8").strip()
            # Try to extract name from first line
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    name = line.lstrip("#").strip() or name
                    break
        return Persona(
            id="default",
            name=name,
            description=description,
            workspace_path=self.root_workspace,
            user_md_path=str(Path(self.root_workspace) / "USER.md"),
        )

    def list_all(self) -> list[Persona]:
        """List all personas, default first, then sorted by name."""
        personas = [self.get_default()]
        if self.personas_dir.exists():
            for d in sorted(self.personas_dir.iterdir()):
                if d.is_dir():
                    p = self._load_persona(d)
                    if p:
                        personas.append(p)
        return personas

    def get(self, persona_id: str) -> Persona | None:
        """Get a persona by ID. Returns None if not found."""
        if persona_id == "default":
            return self.get_default()
        persona_dir = self.personas_dir / persona_id
        if persona_dir.is_dir():
            return self._load_persona(persona_dir)
        return None

    def create(
        self,
        id: str,
        name: str,
        description: str,
        soul_content: str = "",
        tools_allow: list[str] | None = None,
        tools_deny: list[str] | None = None,
        cron_jobs: list[dict] | None = None,
        model_override: str | None = None,
    ) -> Persona:
        """Create a new persona with its directory tree."""
        if id == "default":
            raise ValueError("Cannot create a persona with id 'default'")
        persona_dir = self.personas_dir / id
        if persona_dir.exists():
            raise ValueError(f"Persona '{id}' already exists")

        # Create directory tree
        persona_dir.mkdir(parents=True)
        (persona_dir / "memory").mkdir()
        (persona_dir / "memory" / "skills").mkdir()

        # Write SOUL.md
        if soul_content:
            (persona_dir / "SOUL.md").write_text(soul_content, encoding="utf-8")

        # Write cron.json
        cron_data = cron_jobs or []
        (persona_dir / "cron.json").write_text(
            json.dumps({"jobs": cron_data}, indent=2), encoding="utf-8"
        )

        persona = Persona(
            id=id,
            name=name,
            description=description,
            tools_allow=tools_allow,
            tools_deny=tools_deny,
            model_override=model_override,
            created_at=time.time(),
            workspace_path=str(persona_dir),
            user_md_path=str(Path(self.root_workspace) / "USER.md"),
        )
        self._save_persona(persona)
        logger.info(f"Created persona: {id} ({name})")
        return persona

    def update(self, persona_id: str, **kwargs) -> Persona:
        """Update a persona's configuration."""
        if persona_id == "default":
            raise ValueError("Cannot update the default persona")
        persona = self.get(persona_id)
        if not persona:
            raise ValueError(f"Persona '{persona_id}' not found")

        # Handle soul_content separately
        soul_content = kwargs.pop("soul_content", None)
        if soul_content is not None:
            soul_path = Path(persona.workspace_path) / "SOUL.md"
            soul_path.write_text(soul_content, encoding="utf-8")

        # Update dataclass fields
        for key, value in kwargs.items():
            if hasattr(persona, key) and key not in ("id", "workspace_path", "user_md_path", "created_at"):
                setattr(persona, key, value)

        self._save_persona(persona)
        logger.info(f"Updated persona: {persona_id}")
        return persona

    def delete(self, persona_id: str) -> bool:
        """Delete a persona and its directory. Cannot delete default."""
        if persona_id == "default":
            raise ValueError("Cannot delete the default persona")
        persona_dir = self.personas_dir / persona_id
        if not persona_dir.exists():
            return False
        shutil.rmtree(persona_dir)
        logger.info(f"Deleted persona: {persona_id}")
        return True

    def get_effective_tools(self, persona: Persona, global_allow: list[str]) -> list[str]:
        """Compute the effective tool list for a persona."""
        base = list(persona.tools_allow) if persona.tools_allow is not None else list(global_allow)
        if persona.tools_deny:
            base = [t for t in base if t not in persona.tools_deny]
        # Always include essential tools
        for essential in ("memory_search", "manage_tasks", "manage_personas"):
            if essential not in base:
                base.append(essential)
        return base

    def _load_persona(self, persona_dir: Path) -> Persona | None:
        """Load a persona from its directory."""
        json_path = persona_dir / "persona.json"
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return Persona(
                id=data["id"],
                name=data.get("name", data["id"]),
                description=data.get("description", ""),
                enabled=data.get("enabled", True),
                tools_allow=data.get("tools_allow"),
                tools_deny=data.get("tools_deny"),
                model_override=data.get("model_override"),
                cron_enabled=data.get("cron_enabled", True),
                created_at=data.get("created_at", 0),
                workspace_path=str(persona_dir),
                user_md_path=str(Path(self.root_workspace) / "USER.md"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load persona from {persona_dir}: {e}")
            return None

    def _save_persona(self, persona: Persona) -> None:
        """Write persona.json to disk."""
        persona_dir = Path(persona.workspace_path)
        data = {
            "id": persona.id,
            "name": persona.name,
            "description": persona.description,
            "enabled": persona.enabled,
            "tools_allow": persona.tools_allow,
            "tools_deny": persona.tools_deny,
            "model_override": persona.model_override,
            "cron_enabled": persona.cron_enabled,
            "created_at": persona.created_at,
        }
        json_path = persona_dir / "persona.json"
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
