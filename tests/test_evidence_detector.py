"""Tests for EvidenceDetector's CSV audit trail (now wired into
SessionAnalyzer's per-hypothesis run)."""

import csv
import json

from agentic_detection.evidence_detector import EvidenceDetector
from agentic_detection.rule_engine import RuleEngine


def test_build_evidence_csv_appends_rows_and_records_evidence(tmp_path):
    dr_path = tmp_path / "dr_x.json"
    dr_path.write_text(json.dumps({
        "behavior": "x",
        "evidence_mapping": {"node_a": {"patterns": ["hit"], "search_fields": ["message"]}},
    }))
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(dr_path)
    detector = EvidenceDetector(parser=None, rule_engine=engine)

    out_path = tmp_path / "output.csv"
    detector.build_evidence_csv([{"message": "a hit here"}], behavior_id, out_path, agent_id="a1")
    detector.build_evidence_csv([{"message": "nothing"}], behavior_id, out_path, agent_id="a2")

    with out_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["agent_id"] == "a1"
    assert rows[0]["node_a"] == "1"
    assert rows[1]["agent_id"] == "a2"
    assert rows[1]["node_a"] == "0"
