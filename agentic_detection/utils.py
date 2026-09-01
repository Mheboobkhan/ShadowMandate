"""Shared utility helpers used across the agentic_detection package."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Union


PathLike = Union[str, Path]


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load and parse a JSON file, raising a clear error if it's missing/invalid."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p}: {exc}") from exc


def save_json(data: Dict[str, Any], path: PathLike, indent: int = 2) -> None:
    """Write a dict to disk as pretty-printed JSON, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a numeric value into [lo, hi]."""
    return max(lo, min(hi, value))


# Very small, dependency-free key=value log line tokenizer.
# Handles: key=value, key="quoted value with spaces", key='quoted'
_KV_PATTERN = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_.]*)='
    r'(?:"(?P<dquoted>(?:[^"\\]|\\.)*)"'
    r"|'(?P<squoted>(?:[^'\\]|\\.)*)'"
    r"|(?P<bare>[^\s]+))"
)


def parse_kv_line(line: str) -> Dict[str, str]:
    """Parse a single key=value formatted log line into a dict of strings.

    Supports bare tokens (key=value), double-quoted values (key="a b c"),
    and single-quoted values (key='a b c'). Unmatched trailing text is ignored.
    """
    result: Dict[str, str] = {}
    for m in _KV_PATTERN.finditer(line):
        key = m.group("key")
        value = m.group("dquoted")
        if value is None:
            value = m.group("squoted")
        if value is None:
            value = m.group("bare")
        result[key] = value
    return result


def find_patterns(text: str, patterns: Iterable[str]) -> bool:
    """Case-insensitive substring search: True if any pattern is found in text."""
    if not text:
        return False
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def get_nested_field(obj: Dict[str, Any], field: str) -> Any:
    """Fetch a field from a dict, or from its nested 'action_details' dict."""
    if field in obj:
        return obj[field]
    details = obj.get("action_details")
    if isinstance(details, dict) and field in details:
        return details[field]
    return None
