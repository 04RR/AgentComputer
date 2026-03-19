"""Dynamic skill loader for agent_computer.

Loads auto-extracted Python skill files from memory/skills/ and registers
them as tools in the ToolRegistry, making them available to the agent.

Each skill file can contain multiple public async functions. Every public
async function found will be registered as a separate tool, using the
function name as the tool name.
"""

from __future__ import annotations
import importlib.util
import inspect
import logging
from pathlib import Path

from tool_registry import Tool, ToolParam, ToolRegistry

logger = logging.getLogger("agent_computer.skill_loader")

# Map Python type annotations to OpenAI parameter types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def load_skills(skills_dir: Path | str, registry: ToolRegistry) -> list[str]:
    """Load all .py skill files from a directory and register as tools.

    Returns a list of successfully registered skill names.
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []

    registered = []
    files_loaded = 0
    for skill_file in sorted(skills_dir.glob("*.py")):
        names = _load_skill_functions(skill_file, registry)
        if names:
            registered.extend(names)
            files_loaded += 1

    if registered:
        logger.info(f"Loaded {len(registered)} skill function(s) from {files_loaded} file(s): {', '.join(registered)}")
    return registered


def _load_skill_functions(skill_file: Path, registry: ToolRegistry) -> list[str]:
    """Load all public async functions from a skill file and register them as tools.

    Returns a list of successfully registered tool names.
    """
    module_name = skill_file.stem

    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location(f"skill_{module_name}", str(skill_file))
        if spec is None or spec.loader is None:
            logger.warning(f"Could not create module spec for {skill_file}")
            return []

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Collect all public async functions
        handlers = [
            (name, fn)
            for name, fn in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_") and inspect.iscoroutinefunction(fn)
        ]

        if not handlers:
            logger.warning(f"No public async function found in {skill_file}")
            return []

        module_doc = module.__doc__ or ""
        registered = []

        for fn_name, fn in handlers:
            # Skip if already registered
            if registry.get(fn_name):
                logger.debug(f"Skill {fn_name} already registered, skipping")
                continue

            # Prefer the function's own docstring; fall back to module docstring
            fn_doc = fn.__doc__ or ""
            if fn_doc.strip():
                description = fn_doc.strip().split("\n")[0]
            elif module_doc.strip():
                description = module_doc.strip().split("\n")[0]
            else:
                description = f"Skill: {fn_name}"
            description = f"[skill] {description}"

            # Build params from function signature
            sig = inspect.signature(fn)
            params = []
            for param_name, param in sig.parameters.items():
                annotation = param.annotation
                param_type = _TYPE_MAP.get(annotation, "string")
                required = param.default is inspect.Parameter.empty
                params.append(ToolParam(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter: {param_name}",
                    required=required,
                ))

            tool = Tool(
                name=fn_name,
                description=description,
                params=params,
                handler=fn,
            )
            registry.register(tool)
            logger.info(f"Registered skill: {fn_name} ({len(params)} params)")
            registered.append(fn_name)

        return registered

    except Exception as e:
        logger.error(f"Failed to load skill {skill_file}: {e}")
        return []
