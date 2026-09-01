"""Tests for SessionAnalyzer: weighted-average combination and the
zero-session-weight fallback that used to raise ZeroDivisionError."""

import json

import pytest

from agentic_detection.session_analyzer import SessionAnalyzer


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _build_generic_behavior(root, behavior_id, pattern, weight, session_weight):
    node = f"{behavior_id}_node"
    dr_path = root / "hypotheses" / "generic" / behavior_id / f"dr_{behavior_id}.json"
    bn_path = root / "hypotheses" / "generic" / behavior_id / f"bn_{behavior_id}.json"
    _write_json(dr_path, {
        "behavior": behavior_id,
        "description": "test",
        "evidence_mapping": {node: {"patterns": [pattern], "search_fields": ["message"]}},
    })
    _write_json(bn_path, {
        "behavior": behavior_id,
        "evidence_nodes": [node],
        "cpd_parameters": {"base": 0.1, f"{node}_weight": weight},
    })
    return {
        "id": behavior_id,
        "name": behavior_id,
        "detection_rules": f"hypotheses/generic/{behavior_id}/dr_{behavior_id}.json",
        "bn_config": f"hypotheses/generic/{behavior_id}/bn_{behavior_id}.json",
        "session_weight": session_weight,
    }


def test_weighted_average_combines_fired_hypotheses(tmp_path):
    root = tmp_path / "agentic_detection"
    entry_a = _build_generic_behavior(root, "node_a", "trigger_a", weight=0.4, session_weight=1.0)
    entry_b = _build_generic_behavior(root, "node_b", "trigger_b", weight=0.7, session_weight=3.0)
    config_path = root / "config" / "hypothesis.json"
    _write_json(config_path, {
        "generic_behaviors": [entry_a, entry_b],
        "roles": [{"id": "test_role", "name": "Test Role", "mandate_behaviors": []}],
    })

    log_path = tmp_path / "session.log"
    log_path.write_text('msg="saw trigger_a and trigger_b in one line"\n')

    report = SessionAnalyzer(config_path).analyze(log_path, agent_id="test-agent", role_id="test_role")

    # node_a posterior = .1 + .4 = .5 (weight 1.0); node_b posterior = .1 + .7 = .8 (weight 3.0)
    # weighted average = (1*.5 + 3*.8) / (1+3) = 2.9 / 4 = .725
    assert report.overall_posterior == pytest.approx(0.725)
    assert report.overall_verdict == "DRIFT_DETECTED"


def test_zero_session_weight_fired_hypothesis_does_not_crash(tmp_path):
    """Regression test: previously, when every fired hypothesis had
    session_weight 0.0, dividing by weight_sum raised ZeroDivisionError."""
    root = tmp_path / "agentic_detection"
    entry = _build_generic_behavior(root, "zero_node", "trigger", weight=0.5, session_weight=0.0)
    config_path = root / "config" / "hypothesis.json"
    _write_json(config_path, {
        "generic_behaviors": [entry],
        "roles": [{"id": "test_role", "name": "Test Role", "mandate_behaviors": []}],
    })

    log_path = tmp_path / "session.log"
    log_path.write_text('msg="saw trigger here"\n')

    report = SessionAnalyzer(config_path).analyze(log_path, agent_id="test-agent", role_id="test_role")

    # base .1 + weight .5 = .6; falls back to the fired hypothesis's own
    # posterior instead of dividing by a zero weight sum.
    assert report.overall_posterior == pytest.approx(0.6)


def test_nothing_fired_falls_back_to_minimum_posterior(tmp_path):
    root = tmp_path / "agentic_detection"
    entry = _build_generic_behavior(root, "quiet_node", "trigger", weight=0.5, session_weight=1.0)
    config_path = root / "config" / "hypothesis.json"
    _write_json(config_path, {
        "generic_behaviors": [entry],
        "roles": [{"id": "test_role", "name": "Test Role", "mandate_behaviors": []}],
    })

    log_path = tmp_path / "session.log"
    log_path.write_text('msg="nothing interesting here"\n')

    report = SessionAnalyzer(config_path).analyze(log_path, agent_id="test-agent", role_id="test_role")
    assert report.overall_posterior == pytest.approx(0.1)  # base rate, nothing fired
    assert report.overall_verdict == "NO_DRIFT"


def test_unknown_role_raises_system_exit(tmp_path):
    config_path = tmp_path / "agentic_detection" / "config" / "hypothesis.json"
    _write_json(config_path, {"generic_behaviors": [], "roles": []})
    analyzer = SessionAnalyzer(config_path)
    with pytest.raises(SystemExit):
        analyzer.resolve_for_role("nonexistent_role")


def test_unknown_behavior_raises_system_exit(tmp_path):
    config_path = tmp_path / "agentic_detection" / "config" / "hypothesis.json"
    _write_json(config_path, {"generic_behaviors": [], "roles": []})
    analyzer = SessionAnalyzer(config_path)
    with pytest.raises(SystemExit):
        analyzer.behavior_spec("nonexistent_behavior")


def test_analyze_requires_exactly_one_of_role_or_behavior(tmp_path):
    config_path = tmp_path / "agentic_detection" / "config" / "hypothesis.json"
    _write_json(config_path, {"generic_behaviors": [], "roles": []})
    analyzer = SessionAnalyzer(config_path)
    log_path = tmp_path / "session.log"
    log_path.write_text("msg=\"x\"\n")

    with pytest.raises(ValueError):
        analyzer.analyze(log_path, agent_id="a")
    with pytest.raises(ValueError):
        analyzer.analyze(log_path, agent_id="a", role_id="r", behavior_id="b")
