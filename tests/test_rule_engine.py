"""Tests for RuleEngine: pattern matching, numeric conditions, and the
field-lookup fallback chain (event -> action_details -> raw)."""

import json

import pytest

from agentic_detection.rule_engine import RuleEngine


def _write_rule(tmp_path, evidence_mapping, behavior="test_behavior"):
    path = tmp_path / "dr_test.json"
    path.write_text(json.dumps({"behavior": behavior, "evidence_mapping": evidence_mapping}))
    return path


def test_pattern_match_is_case_insensitive(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"patterns": ["danger"], "search_fields": ["message"]}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)
    evidence = engine.evaluate_logs([{"message": "this is DANGEROUS"}], behavior_id)
    assert evidence == {"node_a": 1}


def test_pattern_no_match_leaves_evidence_at_zero(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"patterns": ["danger"], "search_fields": ["message"]}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)
    evidence = engine.evaluate_logs([{"message": "all clear"}], behavior_id)
    assert evidence == {"node_a": 0}


def test_field_falls_back_to_action_details(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"patterns": ["danger"], "search_fields": ["custom_field"]}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)
    event = {"message": "x", "action_details": {"custom_field": "danger zone"}}
    evidence = engine.evaluate_logs([event], behavior_id)
    assert evidence == {"node_a": 1}


def test_field_falls_back_to_raw(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"patterns": ["danger"], "search_fields": ["custom_field"]}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)
    event = {"message": "x", "action_details": {}, "raw": {"custom_field": "danger zone"}}
    evidence = engine.evaluate_logs([event], behavior_id)
    assert evidence == {"node_a": 1}


def test_condition_gt_operator(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"condition": {"field": "count", "operator": "gt", "value": 10}}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)

    fires = engine.evaluate_logs([{"action_details": {"count": 20}}], behavior_id)
    assert fires == {"node_a": 1}

    no_fire = engine.evaluate_logs([{"action_details": {"count": 5}}], behavior_id)
    assert no_fire == {"node_a": 0}


def test_condition_with_missing_field_does_not_fire(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"condition": {"field": "count", "operator": "gt", "value": 10}}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)
    evidence = engine.evaluate_logs([{"action_details": {}}], behavior_id)
    assert evidence == {"node_a": 0}


def test_condition_compare_field(tmp_path):
    path = _write_rule(tmp_path, {
        "node_a": {"condition": {"field": "a", "compare_field": "b", "operator": "gt"}}
    })
    engine = RuleEngine()
    behavior_id = engine.load_rule_file(path)

    fires = engine.evaluate_logs([{"action_details": {"a": 10, "b": 5}}], behavior_id)
    assert fires == {"node_a": 1}

    no_fire = engine.evaluate_logs([{"action_details": {"a": 5, "b": 10}}], behavior_id)
    assert no_fire == {"node_a": 0}


def test_unknown_behavior_raises_key_error():
    engine = RuleEngine()
    with pytest.raises(KeyError):
        engine.evaluate_logs([], "nonexistent_behavior")


def test_load_rule_file_without_behavior_key_raises(tmp_path):
    path = tmp_path / "dr_bad.json"
    path.write_text(json.dumps({"evidence_mapping": {}}))
    engine = RuleEngine()
    with pytest.raises(ValueError):
        engine.load_rule_file(path)
