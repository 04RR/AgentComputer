"""Verification-mode tools — Phase 1.

Three independent tools used by the /api/verify/raw endpoint:

- reverse_image_search: TinEye reverse-image search via pytineye.
- extract_image_metadata: local EXIF / GPS extraction via Pillow.
- fact_check_lookup: Google Fact Check Tools API claims:search.

Phase 1 registers these in the tool registry but excludes them from the
agent's allow list — they're called directly by the verification endpoint,
not by the agent loop. Phase 2 will flip them on for agent use.

All three tools follow the existing tool contract (async, return a JSON
string, structured error on failure). The blocking calls inside (pytineye
network I/O, Pillow image decode) run on a worker thread via
asyncio.to_thread so concurrent calls in asyncio.gather actually overlap.
"""

from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from tool_registry import Tool, ToolParam, ToolRegistry

logger = logging.getLogger("agent_computer.tools.verification")


# EXIF sub-IFD pointers (used as integer constants to avoid Pillow version
# differences in PIL.ExifTags.Base availability).
_EXIF_IFD_POINTER = 0x8769
_GPS_IFD_POINTER = 0x8825

# Substring matches (case-insensitive) flagged as AI-generator software.
_AI_GENERATOR_HINTS = (
    "stable diffusion", "midjourney", "dall-e", "dalle", "imagen",
    "flux", "comfyui", "automatic1111", "invokeai", "novelai",
)

# Stub responses for reverse_image_search. Shape matches the real TinEye
# contract exactly; "_stub": true at the top level marks them as fake so
# Phase 2 agent prompts and Phase 3 UI can badge stubbed runs distinctly.
_STUB_HIT_RESPONSE: dict = {
    "first_seen_date": "2018-04-02",
    "first_seen_url": "https://www.reuters.com/article/world-middle-east/syria-photo-2018-04-02",
    "first_seen_domain": "reuters.com",
    "total_matches": 47,
    "top_matches": [
        {
            "image_url": "https://www.reuters.com/resizer/photo-2018-04-02.jpg",
            "domain": "reuters.com",
            "score": 98.5,
            "earliest_crawl_date": "2018-04-02",
            "backlinks": [
                {
                    "page_url": "https://www.reuters.com/article/world-middle-east/syria-photo-2018-04-02",
                    "crawl_date": "2018-04-02",
                },
                {
                    "page_url": "https://www.reuters.com/news/picture/related-coverage-2018-04-03",
                    "crawl_date": "2018-04-03",
                },
            ],
        },
        {
            "image_url": "https://www.dpa.com/photo/2018/dpa-12345.jpg",
            "domain": "dpa.com",
            "score": 96.2,
            "earliest_crawl_date": "2018-04-05",
            "backlinks": [
                {
                    "page_url": "https://www.dpa.com/europe/2018/04/05/article",
                    "crawl_date": "2018-04-05",
                },
            ],
        },
        {
            "image_url": "https://apnews.com/images/2018/photo.jpg",
            "domain": "apnews.com",
            "score": 94.8,
            "earliest_crawl_date": "2018-04-10",
            "backlinks": [
                {
                    "page_url": "https://apnews.com/article/middle-east-news-2018-04-10",
                    "crawl_date": "2018-04-10",
                },
            ],
        },
        {
            "image_url": "https://www.afp.com/news/2018/photo.jpg",
            "domain": "afp.com",
            "score": 89.3,
            "earliest_crawl_date": "2018-09-12",
            "backlinks": [
                {
                    "page_url": "https://www.afp.com/news/2018/09/12/related-event",
                    "crawl_date": "2018-09-12",
                },
            ],
        },
        {
            "image_url": "https://www.gettyimages.com/photos/2019/stock.jpg",
            "domain": "gettyimages.com",
            "score": 82.1,
            "earliest_crawl_date": "2019-03-22",
            "backlinks": [
                {
                    "page_url": "https://www.gettyimages.com/photos/news-event-2019-03-22",
                    "crawl_date": "2019-03-22",
                },
            ],
        },
    ],
    "search_engine": "tineye",
    "_stub": True,
}

_STUB_MISS_RESPONSE: dict = {
    "first_seen_date": None,
    "first_seen_url": None,
    "first_seen_domain": None,
    "total_matches": 0,
    "top_matches": [],
    "search_engine": "tineye",
    "_stub": True,
}


# ─── Tool 1: reverse_image_search (TinEye) ───────────────────────────────

def _format_tineye_response(response, max_results: int) -> dict:
    """Convert a pytineye Response into the documented JSON shape."""
    matches = list(getattr(response, "matches", None) or [])
    matches.sort(key=lambda m: getattr(m, "score", 0) or 0, reverse=True)
    top = matches[:max_results]

    # Earliest crawl_date across ALL matches' backlinks (not just top_matches).
    # Lexical sort works for "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS UTC" alike;
    # filter empty/None first to avoid sorting None.
    all_dated_backlinks: list[tuple[str, str]] = []
    for m in matches:
        for bl in getattr(m, "backlinks", None) or []:
            cd = getattr(bl, "crawl_date", None)
            page_url = getattr(bl, "backlink", None)
            if cd and page_url:
                all_dated_backlinks.append((cd, page_url))

    first_seen_date: str | None = None
    first_seen_url: str | None = None
    first_seen_domain: str | None = None
    if all_dated_backlinks:
        all_dated_backlinks.sort(key=lambda x: x[0])
        first_seen_date, first_seen_url = all_dated_backlinks[0]
        try:
            first_seen_domain = urlparse(first_seen_url).netloc or None
        except Exception:
            first_seen_domain = None

    top_matches: list[dict] = []
    for m in top:
        bls = list(getattr(m, "backlinks", None) or [])
        match_dates = [getattr(b, "crawl_date", None) for b in bls]
        match_dates = [d for d in match_dates if d]
        earliest = min(match_dates) if match_dates else None
        bl_out = []
        for bl in bls[:5]:
            bl_out.append({
                "page_url": getattr(bl, "backlink", None),
                "crawl_date": getattr(bl, "crawl_date", None),
            })
        top_matches.append({
            "image_url": getattr(m, "image_url", None),
            "domain": getattr(m, "domain", None),
            "score": getattr(m, "score", None),
            "earliest_crawl_date": earliest,
            "backlinks": bl_out,
        })

    total_matches = getattr(response, "total_results", None)
    if total_matches is None:
        total_matches = len(matches)

    return {
        "first_seen_date": first_seen_date,
        "first_seen_url": first_seen_url,
        "first_seen_domain": first_seen_domain,
        "total_matches": total_matches,
        "top_matches": top_matches,
        "search_engine": "tineye",
    }


# ─── Tool 2: extract_image_metadata (Pillow EXIF) ────────────────────────

def _gps_to_decimal(value, ref) -> float | None:
    """Convert (deg, min, sec) rational tuple + N/S/E/W ref to decimal degrees."""
    if not value or not ref:
        return None
    try:
        d, m, s = float(value[0]), float(value[1]), float(value[2])
        decimal = d + m / 60.0 + s / 3600.0
        if str(ref).upper() in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def _parse_exif_datetime(s) -> datetime | None:
    """Parse 'YYYY:MM:DD HH:MM:SS' (EXIF spec format). Naive datetime."""
    if not s:
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y:%m:%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _extract_metadata_sync(image_path: str) -> dict:
    """Open the image and pull EXIF / GPS / dimensions. Runs on a worker thread."""
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(path) as img:
        dimensions = list(img.size)  # (width, height)
        exif = img.getexif()

    top_level: dict[int, object] = dict(exif) if exif else {}
    exif_ifd: dict[int, object] = {}
    gps_ifd: dict[int, object] = {}
    try:
        exif_ifd = dict(exif.get_ifd(_EXIF_IFD_POINTER)) if exif else {}
    except Exception:
        exif_ifd = {}
    try:
        gps_ifd = dict(exif.get_ifd(_GPS_IFD_POINTER)) if exif else {}
    except Exception:
        gps_ifd = {}

    has_exif = bool(top_level) or bool(exif_ifd) or bool(gps_ifd)

    # Resolve common fields by name.
    def _name_lookup(d: dict, name: str, source: dict = TAGS):
        for tag_id, val in d.items():
            if source.get(tag_id) == name:
                return val
        return None

    camera_make = _name_lookup(top_level, "Make") or _name_lookup(exif_ifd, "Make")
    camera_model = _name_lookup(top_level, "Model") or _name_lookup(exif_ifd, "Model")
    datetime_original = _name_lookup(exif_ifd, "DateTimeOriginal") or _name_lookup(top_level, "DateTime")
    software = _name_lookup(top_level, "Software") or _name_lookup(exif_ifd, "Software")

    gps_lat_raw = _name_lookup(gps_ifd, "GPSLatitude", GPSTAGS)
    gps_lat_ref = _name_lookup(gps_ifd, "GPSLatitudeRef", GPSTAGS)
    gps_lon_raw = _name_lookup(gps_ifd, "GPSLongitude", GPSTAGS)
    gps_lon_ref = _name_lookup(gps_ifd, "GPSLongitudeRef", GPSTAGS)
    gps_lat = _gps_to_decimal(gps_lat_raw, gps_lat_ref)
    gps_lon = _gps_to_decimal(gps_lon_raw, gps_lon_ref)

    # Stringify common fields (Pillow returns bytes for some, IFDRational for others).
    def _stringify(v):
        if v is None:
            return None
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8", errors="replace").strip("\x00 ")
            except Exception:
                return None
        return str(v).strip()

    camera_make_s = _stringify(camera_make)
    camera_model_s = _stringify(camera_model)
    datetime_original_s = _stringify(datetime_original)
    software_s = _stringify(software)

    # Anomalies.
    anomalies: list[dict] = []

    dt = _parse_exif_datetime(datetime_original_s)
    if dt is not None and dt > datetime.now():
        anomalies.append({
            "type": "datetime_in_future",
            "detail": f"DateTimeOriginal {dt.isoformat()} is in the future",
        })

    if software_s:
        sw_lower = software_s.lower()
        for hint in _AI_GENERATOR_HINTS:
            if hint in sw_lower:
                anomalies.append({
                    "type": "ai_generator_software",
                    "detail": f"Software field contains '{hint}': {software_s!r}",
                })
                break

    if not has_exif:
        anomalies.append({
            "type": "no_exif_at_all",
            "detail": "Image has zero EXIF tags (common for web images, weak signal)",
        })

    if (gps_lat is not None or gps_lon is not None) and not camera_make_s and not camera_model_s:
        anomalies.append({
            "type": "gps_without_camera",
            "detail": "GPS coordinates present but no camera make/model — atypical for real photos",
        })

    # raw_exif_keys: union of names from all three IFDs.
    names: set[str] = set()
    for d, source in ((top_level, TAGS), (exif_ifd, TAGS), (gps_ifd, GPSTAGS)):
        for tag_id in d.keys():
            names.add(source.get(tag_id, f"Unknown_{hex(tag_id)}"))

    return {
        "has_exif": has_exif,
        "has_c2pa": False,  # Phase 4 will add C2PA parsing.
        "exif_summary": {
            "camera_make": camera_make_s,
            "camera_model": camera_model_s,
            "datetime_original": datetime_original_s,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "software": software_s,
            "image_dimensions": dimensions,
        },
        "anomalies": anomalies,
        "raw_exif_keys": sorted(names),
    }


# ─── Tool 3: fact_check_lookup (Google Fact Check Tools) ─────────────────

_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _format_factcheck_response(query: str, payload: dict) -> dict:
    """Flatten Google's claims/claimReview shape into a flat list of matches."""
    matches: list[dict] = []
    for claim in payload.get("claims") or []:
        claim_text = claim.get("text") or ""
        claimant = claim.get("claimant")
        claim_date = claim.get("claimDate")
        for review in claim.get("claimReview") or []:
            publisher = review.get("publisher") or {}
            matches.append({
                "claim_text": claim_text,
                "claimant": claimant,
                "claim_date": claim_date,
                "rating": review.get("textualRating") or "",
                "publisher": publisher.get("name") or "",
                "publisher_site": publisher.get("site") or "",
                "review_url": review.get("url") or "",
                "review_title": review.get("title"),
                "review_date": review.get("reviewDate"),
                "language": review.get("languageCode") or "",
            })
    return {
        "query": query,
        "match_count": len(matches),
        "matches": matches,
    }


# ─── Registration ────────────────────────────────────────────────────────

def register_verification_tools(
    registry: ToolRegistry,
    workspace: str,
    verification_config,
) -> None:
    """Register the three verification tools in the shared tool registry.

    Phase 1: registration is unconditional — these are not gated by the
    agent's allow list. The /api/verify/raw endpoint calls them directly
    via registry.execute(); the agent loop never sees them because they're
    absent from config.agent.tools.allow.
    """
    api_url = verification_config.tineye_api_url
    tineye_key = verification_config.tineye_api_key
    tineye_stub_mode = verification_config.tineye_stub_mode
    factcheck_key = verification_config.google_factcheck_api_key

    workspace_root = Path(workspace).resolve()

    def _resolve_image_path(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = workspace_root / path
        return path.resolve()

    # ── reverse_image_search ──
    async def reverse_image_search(image_path: str, max_results: int = 10) -> str:
        # Stub mode short-circuits before everything else: no api_key needed,
        # no pytineye import, no image read. The image_path arg is ignored.
        if tineye_stub_mode == "hit":
            return json.dumps(_STUB_HIT_RESPONSE)
        if tineye_stub_mode == "miss":
            return json.dumps(_STUB_MISS_RESPONSE)
        if not tineye_key:
            return json.dumps({
                "error": "TinEye API key not configured",
                "tool": "reverse_image_search",
            })
        try:
            from pytineye import TinEyeAPIRequest
        except ImportError as e:
            return json.dumps({
                "error": f"pytineye not installed: {e}",
                "tool": "reverse_image_search",
            })
        try:
            path = _resolve_image_path(image_path)
            if not path.exists():
                return json.dumps({
                    "error": f"Image file not found: {image_path}",
                    "tool": "reverse_image_search",
                })
            image_bytes = await asyncio.to_thread(path.read_bytes)

            api = TinEyeAPIRequest(api_url=api_url, api_key=tineye_key)
            # search_data is synchronous; off-load to a worker thread so we
            # don't block the event loop while TinEye crawls.
            response = await asyncio.to_thread(api.search_data, data=image_bytes)

            return json.dumps(_format_tineye_response(response, max_results))
        except Exception as e:
            logger.warning(f"reverse_image_search failed: {e}")
            return json.dumps({"error": str(e), "tool": "reverse_image_search"})

    registry.register(Tool(
        name="reverse_image_search",
        description=(
            "Reverse-image search via TinEye. Given a local image path, returns the "
            "earliest crawl date (first-seen date and URL), top matching pages, and "
            "their backlinks. Useful for tracing where an image has appeared online "
            "and when it was first indexed."
        ),
        params=[
            ToolParam("image_path", "string", "Path to image file (relative to workspace or absolute)"),
            ToolParam("max_results", "integer", "Max top matches to return (default 10)", required=False),
        ],
        handler=reverse_image_search,
    ))

    # ── extract_image_metadata ──
    async def extract_image_metadata(image_path: str) -> str:
        try:
            path = _resolve_image_path(image_path)
            if not path.exists():
                return json.dumps({
                    "error": f"Image file not found: {image_path}",
                    "tool": "extract_image_metadata",
                })
            # Pillow's image decode + EXIF parse can take tens of ms on large
            # images; off-load to keep the event loop responsive.
            result = await asyncio.to_thread(_extract_metadata_sync, str(path))
            return json.dumps(result)
        except Exception as e:
            logger.warning(f"extract_image_metadata failed: {e}")
            return json.dumps({"error": str(e), "tool": "extract_image_metadata"})

    registry.register(Tool(
        name="extract_image_metadata",
        description=(
            "Read EXIF metadata, GPS coordinates, and image dimensions from a local "
            "image file. Flags anomalies like future-dated EXIF, AI-generator software "
            "signatures, missing EXIF entirely, or GPS without camera make/model. "
            "C2PA parsing is not yet implemented (Phase 4)."
        ),
        params=[
            ToolParam("image_path", "string", "Path to image file (relative to workspace or absolute)"),
        ],
        handler=extract_image_metadata,
    ))

    # ── fact_check_lookup ──
    async def fact_check_lookup(query: str, max_results: int = 10, language_code: str = "en") -> str:
        if not factcheck_key:
            return json.dumps({
                "error": "Google Fact Check API key not configured",
                "tool": "fact_check_lookup",
            })
        try:
            params = {
                "query": query,
                "languageCode": language_code,
                "pageSize": max_results,
                "key": factcheck_key,
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(_FACTCHECK_URL, params=params)
                if resp.status_code != 200:
                    return json.dumps({
                        "error": f"Google Fact Check returned HTTP {resp.status_code}: {resp.text[:300]}",
                        "tool": "fact_check_lookup",
                    })
                payload = resp.json()
            return json.dumps(_format_factcheck_response(query, payload))
        except Exception as e:
            logger.warning(f"fact_check_lookup failed: {e}")
            return json.dumps({"error": str(e), "tool": "fact_check_lookup"})

    registry.register(Tool(
        name="fact_check_lookup",
        description=(
            "Search Google's Fact Check Tools API for prior fact-check articles "
            "matching a claim. Returns flattened list of (claim, rating, publisher, "
            "review_url, review_date) tuples — one per ClaimReview."
        ),
        params=[
            ToolParam("query", "string", "Claim text to search for"),
            ToolParam("max_results", "integer", "Max claims to return (default 10)", required=False),
            ToolParam("language_code", "string", "BCP-47 language code (default 'en')", required=False),
        ],
        handler=fact_check_lookup,
    ))
