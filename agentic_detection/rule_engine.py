"""
rule_engine.py - Behavior Detection Rules
===========================================

Loads the security team's curated detection rules (dr_*.json files) and
evaluates parsed log events against them to produce a binary evidence
vector per behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import PathLike, load_json, find_patterns, get_nested_field


class RuleEngine:
    """Loads detection-rule configs and matches them against parsed events."""

    def __init__(self) -> None:
        # behavior_id -> detection rule config dict
        self.rules: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_rule_file(self, path: PathLike) -> str:
        """Load a single dr_*.json detection-rule file. Returns the behavior id."""
        config = load_json(path)
        behavior_id = config.get("behavior")
        if not behavior_id:
            raise ValueError(f"Rule file {path} is missing a 'behavior' key")
        self.rules[behavior_id] = config
        return behavior_id

    def load_rules_directory(self, directory: PathLike) -> List[str]:
        """Recursively load every dr_*.json file found under `directory`."""
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Rules directory not found: {root}")

        loaded: List[str] = []
        for rule_path in sorted(root.rglob("dr_*.json")):
            loaded.append(self.load_rule_file(rule_path))
        return loaded

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_logs(self, events: List[Dict[str, Any]], behavior_id: str) -> Dict[str, int]:
        """Evaluate parsed events against a behavior's evidence_mapping rules.

        Returns a dict of {evidence_node_name: 1 or 0}.
        """
        if behavior_id not in self.rules:
            raise KeyError(
                f"No detection rules loaded for behavior '{behavior_id}'. "
                f"Available: {list(self.rules.keys())}"
            )

        config = self.rules[behavior_id]
        evidence_mapping = config.get("evidence_mapping", {})

        evidence: Dict[str, int] = {node: 0 for node in evidence_mapping}
        matches: Dict[str, List[Dict[str, Any]]] = {node: [] for node in evidence_mapping}

        for node_name, node_config in evidence_mapping.items():
            condition = node_config.get("condition")
            patterns = node_config.get("patterns", [])
            search_fields = node_config.get("search_fields", ["message"])

            for event in events:
                if condition is not None:
                    matched = self._condition_matches(event, condition)
                else:
                    matched = self._event_matches(event, patterns, search_fields)
                if matched:
                    evidence[node_name] = 1
                    matches[node_name].append(event)

        self._last_matches = matches
        return evidence

    def get_last_matches(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the events that matched each evidence node in the last evaluate_logs call."""
        return getattr(self, "_last_matches", {})

    def get_behavior_config(self, behavior_id: str) -> Dict[str, Any]:
        return self.rules[behavior_id]

    def list_behaviors(self) -> List[str]:
        return list(self.rules.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _event_matches(
        self,
        event: Dict[str, Any],
        patterns: List[str],
        search_fields: List[str],
    ) -> bool:
        for field in search_fields:
            value = self._field_value(event, field)
            if value is None:
                continue
            if find_patterns(str(value), patterns):
                return True
        return False

    _OPERATORS = {
        "gt": lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
    }

    def _condition_matches(self, event: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluate a numeric/structural evidence condition against one event.

        Supports comparing a field to a literal `value`, or to another
        field via `compare_field` (e.g. detecting a mismatch between two
        fields on the same event). Either side being absent (None) means
        the condition cannot be evaluated and is treated as not matched,
        so behaviors observed only in some log formats don't false-positive
        on logs that simply lack that metadata.
        """
        operator = condition.get("operator", "eq")
        op_fn = self._OPERATORS.get(operator)
        if op_fn is None:
            raise ValueError(f"Unsupported condition operator: {operator!r}")

        left = self._field_value(event, condition["field"])
        if "compare_field" in condition:
            right = self._field_value(event, condition["compare_field"])
        else:
            right = condition.get("value")

        if left is None or right is None:
            return False

        # Try numeric comparison first (handles ints/floats/numeric strings);
        # fall back to the raw values (e.g. string equality) otherwise.
        try:
            left_cmp, right_cmp = float(left), float(right)
        except (TypeError, ValueError):
            left_cmp, right_cmp = left, right

        try:
            return bool(op_fn(left_cmp, right_cmp))
        except TypeError:
            return False

    def _field_value(self, event: Dict[str, Any], field: str) -> Any:
        value = event.get(field)
        if value is None:
            value = get_nested_field(event.get("action_details", {}), field)
        if value is None:
            value = event.get("raw", {}).get(field)
        return value
