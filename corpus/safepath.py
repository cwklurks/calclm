"""Path sanitizing for untrusted names (URLs, archive members)."""

from __future__ import annotations

import re

_SAFE = re.compile(r"[^A-Za-z0-9._+\-]")


def safe_relpath(name: str, max_parts: int = 12) -> str | None:
    """Map an untrusted slash path to a safe relative path, or None if it cannot be made safe.

    Drops empty, '.' and '..' components (never climbs), replaces odd characters with '_',
    and rejects absurdly deep paths.
    """
    parts = []
    for comp in name.replace("\\", "/").split("/"):
        if comp in ("", ".", ".."):
            continue
        comp = _SAFE.sub("_", comp)[:120].lstrip(".") or "_"
        parts.append(comp)
    if not parts or len(parts) > max_parts:
        return None
    return "/".join(parts)
