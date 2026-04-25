"""Smoke test for the Phase 1 verification tools.

Usage:
    python scripts/verify_tools_smoke.py [<image_path>]

If <image_path> is omitted, generates a 64x64 synthetic PNG in the system
temp dir and runs the metadata tool against it. The synthetic image has no
EXIF — extract_image_metadata should return has_exif=false and flag a
no_exif_at_all anomaly.

API-key-gated tools (reverse_image_search, fact_check_lookup) are skipped
with a printed note when the corresponding key is missing from config.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Allow running from the repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config  # noqa: E402
from tool_registry import ToolRegistry  # noqa: E402
from tools.verification import register_verification_tools  # noqa: E402


def _make_synthetic_image() -> str:
    """Create a 64x64 solid-color PNG. Returns its path."""
    from PIL import Image
    path = Path(tempfile.gettempdir()) / "verify_smoke_synthetic.png"
    Image.new("RGB", (64, 64), color=(120, 50, 200)).save(path, format="PNG")
    return str(path)


async def main(image_path: str) -> None:
    config = load_config(str(REPO_ROOT / "config.json"))
    registry = ToolRegistry()
    register_verification_tools(registry, config.agent.workspace, config.verification)

    print(f"=== extract_image_metadata ===")
    print(f"  image: {image_path}")
    result_str = await registry.execute("extract_image_metadata", {"image_path": image_path})
    print(json.dumps(json.loads(result_str), indent=2))
    print()

    print("=== reverse_image_search ===")
    if config.verification.tineye_api_key:
        result_str = await registry.execute(
            "reverse_image_search",
            {"image_path": image_path, "max_results": 5},
        )
        print(json.dumps(json.loads(result_str), indent=2))
    else:
        print("  SKIP: tineye_api_key not configured in config.json")
        print("  (TinEye sandbox key for shape-checking: "
              "'6mm60lsCNIB,FwOWjJqA80QZHh9BMwc-ber4u=t^' — always returns matches "
              "for the meloncat test image regardless of input.)")
    print()

    print("=== fact_check_lookup ===")
    if config.verification.google_factcheck_api_key:
        result_str = await registry.execute(
            "fact_check_lookup",
            {"query": "Earth is flat", "max_results": 3},
        )
        print(json.dumps(json.loads(result_str), indent=2))
    else:
        print("  SKIP: google_factcheck_api_key not configured in config.json")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"error: image not found: {path}", file=sys.stderr)
            sys.exit(1)
    else:
        path = _make_synthetic_image()
        print(f"(no image arg given — using synthetic: {path})\n")
    asyncio.run(main(path))
