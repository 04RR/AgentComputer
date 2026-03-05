"""PinchTab browser control tools.

Provides interactive browser automation via PinchTab (local CDP proxy).
These complement the Scrapling tools — use Scrapling for fast stateless
HTTP fetches and PinchTab for clicking, filling forms, and multi-step
browser workflows.
"""

from __future__ import annotations
import json
from pathlib import Path

import httpx

from tool_registry import Tool, ToolParam, ToolRegistry

# Module-level state — reused across tool calls
_client: httpx.AsyncClient | None = None
_instance_id: str | None = None


def _get_client(config) -> httpx.AsyncClient:
    """Return (or create) the shared httpx client for PinchTab."""
    global _client
    if _client is None:
        headers = {}
        if config.token:
            headers["Authorization"] = f"Bearer {config.token}"
        _client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=30,
        )
    return _client


async def _ensure_instance(config) -> str:
    """Return a cached instance ID, or start/reuse one from PinchTab."""
    global _instance_id
    if _instance_id is not None:
        return _instance_id

    client = _get_client(config)

    # Check for existing instances
    resp = await client.get("/instances")
    if resp.status_code == 200:
        data = resp.json()
        instances = data if isinstance(data, list) else data.get("instances", [])
        if instances:
            _instance_id = instances[0].get("id") or instances[0].get("instanceId")
            if _instance_id:
                return _instance_id

    # No existing instance — start a new one
    body = {
        "profileId": config.default_profile,
        "mode": "" if config.headless else "headed",
    }
    resp = await client.post("/instances/start", json=body)
    resp.raise_for_status()
    result = resp.json()
    _instance_id = result.get("id") or result.get("instanceId")
    return _instance_id


def register_pinchtab_tools(
    registry: ToolRegistry,
    workspace: str,
    allowed: list[str] | None = None,
    pinchtab_config=None,
) -> None:
    """Register all PinchTab browser tools."""

    config = pinchtab_config

    def _is_allowed(name: str) -> bool:
        return allowed is None or name in allowed

    # ─── browser_navigate ───

    if _is_allowed("browser_navigate"):
        async def browser_navigate(url: str, new_tab: bool = False) -> str:
            """Navigate the browser to a URL."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                resp = await client.post("/navigate", json={"url": url, "newTab": new_tab})
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_navigate",
            description="Navigate the browser to a URL. Returns tab ID, final URL, and page title.",
            params=[
                ToolParam("url", "string", "The URL to navigate to"),
                ToolParam("new_tab", "boolean", "Open in a new tab (default false)", required=False),
            ],
            handler=browser_navigate,
        ))

    # ─── browser_snapshot ───

    if _is_allowed("browser_snapshot"):
        async def browser_snapshot(selector: str | None = None, max_tokens: int = 4000) -> str:
            """Get an accessibility snapshot of the current page."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                params = {"filter": "interactive", "format": "compact"}
                if selector:
                    params["selector"] = selector
                if max_tokens != 4000:
                    params["maxTokens"] = str(max_tokens)
                resp = await client.get("/snapshot", params=params)
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return resp.text
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_snapshot",
            description="Get an accessibility tree snapshot of the page. Returns interactive elements with ref IDs (e0, e5, e12) for use with browser_click, browser_type, browser_fill.",
            params=[
                ToolParam("selector", "string", "CSS selector to scope the snapshot", required=False),
                ToolParam("max_tokens", "integer", "Max tokens in response (default 4000)", required=False),
            ],
            handler=browser_snapshot,
        ))

    # ─── browser_click ───

    if _is_allowed("browser_click"):
        async def browser_click(ref: str, wait_nav: bool = False) -> str:
            """Click an element by its ref ID from the snapshot."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                body = {"kind": "click", "ref": ref, "waitNav": wait_nav}
                resp = await client.post("/action", json=body)
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_click",
            description="Click an element by its ref ID from browser_snapshot.",
            params=[
                ToolParam("ref", "string", "Element ref ID (e.g. 'e5')"),
                ToolParam("wait_nav", "boolean", "Wait for navigation after click (default false)", required=False),
            ],
            handler=browser_click,
        ))

    # ─── browser_type ───

    if _is_allowed("browser_type"):
        async def browser_type(ref: str, text: str) -> str:
            """Type text into an element character by character."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                body = {"kind": "type", "ref": ref, "text": text}
                resp = await client.post("/action", json=body)
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_type",
            description="Type text into an element character by character. Use for inputs that need keystroke events.",
            params=[
                ToolParam("ref", "string", "Element ref ID (e.g. 'e5')"),
                ToolParam("text", "string", "Text to type"),
            ],
            handler=browser_type,
        ))

    # ─── browser_fill ───

    if _is_allowed("browser_fill"):
        async def browser_fill(ref: str, text: str) -> str:
            """Fill an input field instantly (clears existing value first)."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                body = {"kind": "fill", "ref": ref, "value": text}
                resp = await client.post("/action", json=body)
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_fill",
            description="Fill an input field instantly, replacing any existing value. Faster than browser_type.",
            params=[
                ToolParam("ref", "string", "Element ref ID (e.g. 'e5')"),
                ToolParam("text", "string", "Text to fill into the field"),
            ],
            handler=browser_fill,
        ))

    # ─── browser_press ───

    if _is_allowed("browser_press"):
        async def browser_press(key: str) -> str:
            """Press a keyboard key (e.g. Enter, Tab, Escape)."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                body = {"kind": "press", "key": key}
                resp = await client.post("/action", json=body)
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_press",
            description="Press a keyboard key. Examples: 'Enter', 'Tab', 'Escape', 'ArrowDown', 'Control+a'.",
            params=[
                ToolParam("key", "string", "Key to press (e.g. 'Enter', 'Tab')"),
            ],
            handler=browser_press,
        ))

    # ─── browser_scroll ───

    if _is_allowed("browser_scroll"):
        async def browser_scroll(direction: str) -> str:
            """Scroll the page up or down."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                scroll_y = -500 if direction == "up" else 500
                body = {"kind": "scroll", "scrollX": 0, "scrollY": scroll_y}
                resp = await client.post("/action", json=body)
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_scroll",
            description="Scroll the page up or down by ~500px.",
            params=[
                ToolParam("direction", "string", "Scroll direction", enum=["up", "down"]),
            ],
            handler=browser_scroll,
        ))

    # ─── browser_text ───

    if _is_allowed("browser_text"):
        async def browser_text() -> str:
            """Get the full text content of the current page."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                resp = await client.get("/text")
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                data = resp.json()
                text = data.get("text", "")
                truncated = len(text) > 10000
                if truncated:
                    text = text[:10000]
                return json.dumps({
                    "url": data.get("url", ""),
                    "title": data.get("title", ""),
                    "text": text,
                    "truncated": truncated,
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_text",
            description="Get the full text content of the current page (truncated to 10000 chars). For structured interaction, use browser_snapshot instead.",
            params=[],
            handler=browser_text,
        ))

    # ─── browser_screenshot ───

    if _is_allowed("browser_screenshot"):
        async def browser_screenshot(filename: str = "screenshot.jpg") -> str:
            """Take a screenshot of the current page."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                resp = await client.get("/screenshot", params={"raw": "true", "quality": "70"})
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                save_path = Path(workspace) / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(resp.content)
                return json.dumps({"status": "ok", "path": str(save_path), "bytes": len(resp.content)})
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_screenshot",
            description="Take a JPEG screenshot of the current page and save it to the workspace.",
            params=[
                ToolParam("filename", "string", "Filename to save as (default 'screenshot.jpg')", required=False),
            ],
            handler=browser_screenshot,
        ))

    # ─── browser_tabs ───

    if _is_allowed("browser_tabs"):
        async def browser_tabs() -> str:
            """List all open browser tabs."""
            try:
                await _ensure_instance(config)
                client = _get_client(config)
                resp = await client.get("/tabs")
                if resp.status_code >= 400:
                    return json.dumps({"error": f"PinchTab returned {resp.status_code}: {resp.text[:500]}"})
                return json.dumps(resp.json())
            except Exception as e:
                return json.dumps({"error": str(e)})

        registry.register(Tool(
            name="browser_tabs",
            description="List all open browser tabs with their IDs, URLs, and titles.",
            params=[],
            handler=browser_tabs,
        ))
