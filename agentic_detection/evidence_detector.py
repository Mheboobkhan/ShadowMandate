"""
evidence_detector.py - Evidence Normalization
================================================

Converts log events into binary evidence vectors (0/1 per evidence node)
and can persist them as a normalized CSV for auditing or batch inference.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from .log_parser import AgentLogParser
from .rule_engine import RuleEngine
from .utils import PathLike


class EvidenceDetector:
    """Bridges the RuleEngine's matching logic with a simple evidence API."""

    def __init__(self, parser: AgentLogParser, rule_engine: RuleEngine) -> None:
        self.parser = parser
        self.rule_engine = rule_engine

    def detect_evidence(self, events: List[Dict[str, Any]], behavior_id: str) -> Dict[str, int]:
        """Return the binary evidence vector for a behavior given parsed events."""
        return self.rule_engine.evaluate_logs(events, behavior_id)

    def build_evidence_csv(
        self,
        events: List[Dict[str, Any]],
        behavior_id: str,
        output_path: PathLike,
        agent_id: str = "agent-001",
    ) -> Dict[str, int]:
        """Compute evidence for a behavior and append a normalized row to a CSV.

        If the CSV already exists, the row is appended (creating any new
        evidence-node columns as needed); otherwise a new file is created.
        """
        evidence = self.detect_evidence(events, behavior_id)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["agent_id", "behavior"] + sorted(evidence.keys())
        row = {"agent_id": agent_id, "behavior": behavior_id, **evidence}

        existing_rows: List[Dict[str, Any]] = []
        if out_path.exists():
            with out_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                existing_fields = reader.fieldnames or []
                for field in existing_fields:
                    if field not in fieldnames:
                        fieldnames.append(field)

        existing_rows.append(row)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)

        return evidence

    def batch_detect(
        self,
        log_files: List[PathLike],
        behavior_id: str,
        agent_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run evidence detection across multiple agent log files.

        Returns a list of {"agent_id": ..., "evidence": {...}} dicts.
        """
        results = []
        agent_ids = agent_ids or [Path(p).stem for p in log_files]
        if len(agent_ids) != len(log_files):
            raise ValueError("agent_ids must be the same length as log_files")

        for log_file, agent_id in zip(log_files, agent_ids):
            events = self.parser.parse_log_file(log_file)
            evidence = self.detect_evidence(events, behavior_id)
            results.append({"agent_id": agent_id, "evidence": evidence})

        return results
