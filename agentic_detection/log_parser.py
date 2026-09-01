"""
log_parser.py - Log Parsing & Event Extraction
================================================

Parses key=value formatted agent logs (and a few common fallback formats)
into a normalized list of event dicts that the rest of the pipeline can
consume.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .utils import PathLike, parse_kv_line


# Ordered list of (action_type, patterns-to-check-in-message) used to
# categorize a parsed event. Order matters: first match wins.
_ACTION_TYPE_RULES = [
    ("external_download", ["download", "downloaded", "downloading", ".exe", ".dmg", ".pkg", ".tar.gz", ".zip", "bundle"]),
    ("update_check", ["update available", "checking for update", "new update", "update_check"]),
    ("http_request", ["http://", "https://", "GET ", "POST ", "PUT ", "DELETE "]),
    ("dns_lookup", ["dns", "lookup", "resolve"]),
    ("file_write", ["wrote file", "writing", "saved to", "file_path"]),
    ("credential_access", ["password", "secret", "credential", "api_key", "apikey", "token="]),
    ("process_exec", ["exec", "spawn", "subprocess", "started process"]),
    ("connection", ["connect", "connection", "socket"]),
]

_URL_PATTERN = re.compile(r'(https?://[^\s"\'<>]+)', re.IGNORECASE)

# Regex for a generic timestamp at the start of a line, e.g.:
# time=2025-09-26T13:22:59.220-04:00
_TIME_PREFIX_PATTERN = re.compile(
    r'^\s*(?:time=)?(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)'
)


class AgentLogParser:
    """Parses agent log files into normalized event dictionaries.

    Each parsed event has (at minimum) the shape::

        {
            "timestamp": "2025-09-26T13:22:59.220-04:00",
            "message": "New update available at https://...",
            "action_type": "external_download",
            "action_details": {
                "url": "https://...",
                "has_external_url": True,
                ...
            },
            "raw": {...all key=value fields parsed from the line...},
            "raw_line": "the original log line",
        }
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_log_file(self, path: PathLike) -> List[Dict[str, Any]]:
        """Parse an entire log file and return the list of extracted events."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {p}")

        events: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                event = self.parse_log_line(line)
                if event is not None:
                    event["line_number"] = line_no
                    events.append(event)

        self.events = events
        return events

    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line into a normalized event dict.

        Auto-detects the line format: a JSON object (e.g. AWS Bedrock
        ModelInvocationLog ndjson) is parsed structurally; anything else
        falls back to the key=value tokenizer.
        """
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                doc = json.loads(stripped)
            except json.JSONDecodeError:
                doc = None
            if isinstance(doc, dict) and doc.get("schemaType") == "ModelInvocationLog":
                return self._parse_bedrock_invocation_log(doc, line)

        raw_fields = parse_kv_line(line)

        timestamp = raw_fields.get("time") or raw_fields.get("timestamp")
        if not timestamp:
            m = _TIME_PREFIX_PATTERN.match(line)
            if m:
                timestamp = m.group("ts")

        message = raw_fields.get("msg") or raw_fields.get("message") or line

        action_details = self._extract_action_details(raw_fields, message)
        action_type = self._categorize_action(message, action_details)

        event = {
            "timestamp": timestamp,
            "message": message,
            "action_type": action_type,
            "action_details": action_details,
            "raw": raw_fields,
            "raw_line": line,
        }
        return event

    def get_events_by_type(self, action_type: str) -> List[Dict[str, Any]]:
        """Return all parsed events matching a given action_type."""
        return [e for e in self.events if e.get("action_type") == action_type]

    def get_external_connections(self) -> List[Dict[str, Any]]:
        """Convenience filter: events that touched an external URL/host."""
        return [
            e for e in self.events
            if e.get("action_type") in ("external_download", "http_request", "update_check", "connection")
            or e.get("action_details", {}).get("has_external_url")
        ]

    def summarize(self) -> Dict[str, int]:
        """Return a count of events grouped by action_type."""
        counts: Dict[str, int] = {}
        for e in self.events:
            counts[e["action_type"]] = counts.get(e["action_type"], 0) + 1
        return counts

    # ------------------------------------------------------------------
    # AWS Bedrock ModelInvocationLog (ndjson) support
    # ------------------------------------------------------------------

    def _parse_bedrock_invocation_log(self, doc: Dict[str, Any], line: str) -> Dict[str, Any]:
        """Flatten one AWS Bedrock ModelInvocationLog record into a normalized event.

        Bedrock invocation logs are structured JSON, not key=value text, so
        prompt text, tool calls, token counts, and routing metadata are
        extracted directly from their nested paths rather than regexed out
        of a message string.
        """
        input_obj = doc.get("input") or {}
        output_obj = doc.get("output") or {}
        input_body = input_obj.get("inputBodyJson") or {}
        output_body = output_obj.get("outputBodyJson")

        prompt_text = self._extract_bedrock_prompt_text(input_body)
        tool_names = self._extract_bedrock_tool_names(output_body)

        input_token_count = input_obj.get("inputTokenCount")
        output_token_count = output_obj.get("outputTokenCount")
        ratio = None
        if isinstance(input_token_count, (int, float)) and isinstance(output_token_count, (int, float)):
            ratio = output_token_count / max(input_token_count, 1)

        identity = doc.get("identity") or {}
        metadata = input_body.get("metadata") or {}

        action_details = {
            "prompt_text": prompt_text,
            "tool_names": ", ".join(tool_names),
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "output_input_ratio": ratio,
            "region": doc.get("region"),
            "inference_region": doc.get("inferenceRegion"),
            "model_id": doc.get("modelId"),
            "identity_arn": identity.get("arn"),
            "user_id": metadata.get("user_id"),
            "operation": doc.get("operation"),
            "request_id": doc.get("requestId"),
        }

        return {
            "timestamp": doc.get("timestamp"),
            "message": prompt_text,
            "action_type": "tool_invocation" if tool_names else "model_invocation",
            "action_details": action_details,
            "raw": doc,
            "raw_line": line.rstrip("\n"),
        }

    @staticmethod
    def _extract_bedrock_prompt_text(input_body: Dict[str, Any]) -> str:
        chunks: List[str] = []
        for msg in input_body.get("messages", []) or []:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    chunks.append(block["text"])
        return " ".join(chunks)

    @staticmethod
    def _extract_bedrock_tool_names(output_body: Any) -> List[str]:
        names: List[str] = []
        if not isinstance(output_body, list):
            return names
        for block in output_body:
            if not isinstance(block, dict) or block.get("type") != "content_block_start":
                continue
            content_block = block.get("content_block") or {}
            tool_use = content_block.get("toolUse") or {}
            name = tool_use.get("name")
            if name:
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_action_details(self, raw_fields: Dict[str, str], message: str) -> Dict[str, Any]:
        details: Dict[str, Any] = {}

        url_match = _URL_PATTERN.search(message)
        url = raw_fields.get("url") or (url_match.group(1) if url_match else None)
        if url:
            details["url"] = url
            details["has_external_url"] = True
        else:
            details["has_external_url"] = False

        for candidate_key in ("file_path", "path", "file"):
            if candidate_key in raw_fields:
                details["file_path"] = raw_fields[candidate_key]
                break
        else:
            # try to spot a windows/unix path in the message
            path_match = re.search(r'([A-Za-z]:\\[^\s"]+|/(?:[\w.\-]+/)+[\w.\-]+)', message)
            if path_match:
                details["file_path"] = path_match.group(1)

        for extra_key in ("host", "domain", "ip", "port", "status", "level"):
            if extra_key in raw_fields:
                details[extra_key] = raw_fields[extra_key]

        return details

    def _categorize_action(self, message: str, details: Dict[str, Any]) -> str:
        lowered = (message or "").lower()
        for action_type, keywords in _ACTION_TYPE_RULES:
            if any(kw.lower() in lowered for kw in keywords):
                return action_type
        if details.get("has_external_url"):
            return "http_request"
        return "other"
