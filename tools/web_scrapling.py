"""Scrapling-based web fetching tools.

Provides three tiers of web fetching:
- web_fetch: Fast HTTP with browser TLS impersonation (AsyncFetcher)
- web_fetch_js: Full browser JS rendering (DynamicFetcher)
- web_fetch_stealth: Stealth anti-bot bypass (StealthyFetcher)
"""

from __future__ import annotations
import json

from tool_registry import Tool, ToolParam, ToolRegistry


def _extract_response(response, css_selector: str | None, max_chars: int) -> str:
    """Extract text from a Scrapling Response, optionally filtered by CSS selector."""
    if css_selector:
        elements = response.css(css_selector)
        text = "\n".join(str(el.get_all_text(separator=" ", strip=True)) for el in elements)
    else:
        text = str(response.get_all_text(separator="\n", strip=True))
    truncated = len(text) > max_chars
    return json.dumps({
        "status_code": response.status,
        "url": response.url,
        "body": text[:max_chars],
        "truncated": truncated,
    })


def register_scrapling_tools(registry: ToolRegistry, workspace: str, allowed: list[str] | None = None) -> None:
    """Register all Scrapling-based web fetching tools. If allowed is given, only register tools in the list."""

    # ─── web_fetch (AsyncFetcher) ───

    async def web_fetch(url: str, css_selector: str | None = None, max_chars: int = 10000) -> str:
        """Fetch a URL via HTTP with browser TLS impersonation."""
        try:
            from scrapling.fetchers import AsyncFetcher
            response = await AsyncFetcher.get(
                url,
                stealthy_headers=True,
                follow_redirects=True,
                timeout=15,
            )
            return _extract_response(response, css_selector, max_chars)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if allowed is None or "web_fetch" in allowed:
        registry.register(Tool(
            name="web_fetch",
            description="Fetch a URL via HTTP with browser TLS impersonation. Fast, no JS rendering. Best for reading pages, fetching data, API calls. For interactive tasks (clicking, filling forms, multi-step workflows), use the browser_* tools instead.",
            params=[
                ToolParam("url", "string", "The URL to fetch"),
                ToolParam("css_selector", "string", "CSS selector to extract specific elements", required=False),
                ToolParam("max_chars", "integer", "Max characters to return (default 10000)", required=False),
            ],
            handler=web_fetch,
        ))

    # ─── web_fetch_js (DynamicFetcher) ───

    async def web_fetch_js(url: str, css_selector: str | None = None, wait_seconds: int = 3, max_chars: int = 10000) -> str:
        """Fetch a URL with a full browser for JS rendering."""
        try:
            from scrapling.fetchers import DynamicFetcher
            response = await DynamicFetcher.async_fetch(
                url,
                headless=True,
                network_idle=True,
                wait=wait_seconds * 1000,
                timeout=30000,
            )
            return _extract_response(response, css_selector, max_chars)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if allowed is None or "web_fetch_js" in allowed:
        registry.register(Tool(
            name="web_fetch_js",
            description="Fetch a URL with a full browser for JS rendering. Use for dynamic sites (Reddit, YouTube, Google). Slower but renders JS content.",
            params=[
                ToolParam("url", "string", "The URL to fetch"),
                ToolParam("css_selector", "string", "CSS selector to extract specific elements", required=False),
                ToolParam("wait_seconds", "integer", "Seconds to wait for JS to render (default 3)", required=False),
                ToolParam("max_chars", "integer", "Max characters to return (default 10000)", required=False),
            ],
            handler=web_fetch_js,
        ))

    # ─── web_fetch_stealth (StealthyFetcher) ───

    async def web_fetch_stealth(url: str, css_selector: str | None = None, max_chars: int = 10000) -> str:
        """Fetch a URL with a stealth browser that bypasses bot detection."""
        try:
            from scrapling.fetchers import StealthyFetcher
            response = await StealthyFetcher.async_fetch(
                url,
                headless=True,
                network_idle=True,
                solve_cloudflare=True,
            )
            return _extract_response(response, css_selector, max_chars)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if allowed is None or "web_fetch_stealth" in allowed:
        registry.register(Tool(
            name="web_fetch_stealth",
            description="Fetch a URL with a stealth browser that bypasses Cloudflare and bot detection. Use when other fetchers get blocked.",
            params=[
                ToolParam("url", "string", "The URL to fetch"),
                ToolParam("css_selector", "string", "CSS selector to extract specific elements", required=False),
                ToolParam("max_chars", "integer", "Max characters to return (default 10000)", required=False),
            ],
            handler=web_fetch_stealth,
        ))
