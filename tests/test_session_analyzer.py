"""Tests for SessionAnalyzer: how per-hypothesis posteriors combine into one
session score, and the zero-session-weight fallback that used to raise
ZeroDivisionError."""

import json

import pytest

from agentic_detection.session_analyzer import SessionAnalyzer, combine_posteriors
from agentic_detection.verdict_generator import VerdictGenerator


def _result(posterior, weight, fired=True, hypothesis_id="h"):
    """A minimal HypothesisResult stand-in for combination-math tests."""
    from agentic_detection.session_analyzer import HypothesisResult

    verdict = VerdictGenerator().generate_verdict(
        agent_id="a", behavior_id=hypothesis_id, posterior_probability=posterior
    )
    return HypothesisResult(
        id=hypothesis_id,
        name=hypothesis_id,
        is_mandate=False,
        session_weight=weight,
        fired=fired,
        verdict=verdict,
    )


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


def test_corroborating_evidence_never_lowers_the_session_score():
    """Regression test for the dilution bug.

    The old weighted average let extra *true positives* drag the session down:
    a driving hypothesis at 0.93 plus four correct generic hits in the
    0.40-0.50 band averaged out to 0.65, below the driving signal itself. Under
    every supported strategy, adding a fired hypothesis must never reduce the
    score.
    """
    driving = _result(0.93, weight=2.5, hypothesis_id="mandate")
    corroborating = [
        _result(0.50, weight=1.0, hypothesis_id="generic_a"),
        _result(0.45, weight=1.0, hypothesis_id="generic_b"),
        _result(0.40, weight=1.0, hypothesis_id="generic_c"),
        _result(0.40, weight=1.0, hypothesis_id="generic_d"),
    ]

    for strategy in ("noisy_or", "max"):
        alone = combine_posteriors([driving], strategy=strategy)
        together = combine_posteriors([driving] + corroborating, strategy=strategy)
        assert together >= alone, f"{strategy} diluted the driving hypothesis"

    # The old behavior, kept available for comparison, still dilutes - which is
    # precisely why it is no longer the default.
    legacy = combine_posteriors([driving] + corroborating, strategy="weighted_average")
    assert legacy < driving.verdict.posterior_probability


def test_noisy_or_is_monotonic_in_added_evidence():
    base = [_result(0.6, weight=1.0, hypothesis_id="a")]
    previous = combine_posteriors(base, strategy="noisy_or")
    for i in range(4):
        base.append(_result(0.3, weight=1.0, hypothesis_id=f"extra_{i}"))
        current = combine_posteriors(base, strategy="noisy_or")
        assert current > previous
        previous = current
    assert previous < 1.0


def test_session_weight_suppresses_generic_noise_under_noisy_or():
    """session_weight is what keeps a mandate violation in charge.

    Under noisy-OR each hypothesis contributes with exponent w_i / w_max, so
    raising the mandate's weight shrinks every generic hypothesis's influence
    and pulls the session score toward the mandate's own posterior instead of
    letting a crowd of weak generic hits run it up.
    """
    generics = [_result(0.3, weight=1.0, hypothesis_id=f"g{i}") for i in range(3)]

    flat = combine_posteriors(
        [_result(0.8, weight=1.0, hypothesis_id="mandate")] + generics, strategy="noisy_or"
    )
    weighted = combine_posteriors(
        [_result(0.8, weight=2.5, hypothesis_id="mandate")] + generics, strategy="noisy_or"
    )

    assert weighted < flat, "session_weight did not suppress the generic hypotheses"
    assert weighted > 0.8, "session score fell below its driving hypothesis"


def test_unfired_hypotheses_are_excluded_from_the_combination():
    fired = _result(0.7, weight=1.0, hypothesis_id="fired")
    quiet = _result(0.05, weight=1.0, fired=False, hypothesis_id="quiet")
    assert combine_posteriors([fired, quiet], strategy="noisy_or") == pytest.approx(
        combine_posteriors([fired], strategy="noisy_or")
    )


def test_unknown_combination_strategy_is_rejected():
    with pytest.raises(ValueError):
        combine_posteriors([_result(0.5, 1.0)], strategy="average_of_vibes")


def test_combination_strategy_is_config_driven(tmp_path):
    root = tmp_path / "agentic_detection"
    entry_a = _build_generic_behavior(root, "node_a", "trigger_a", weight=0.4, session_weight=1.0)
    entry_b = _build_generic_behavior(root, "node_b", "trigger_b", weight=0.7, session_weight=3.0)
    log_path = tmp_path / "session.log"
    log_path.write_text('msg="saw trigger_a and trigger_b in one line"\n')

    scores = {}
    for strategy in ("noisy_or", "max", "weighted_average"):
        config_path = root / "config" / f"hypothesis_{strategy}.json"
        _write_json(config_path, {
            "session_combination": strategy,
            "generic_behaviors": [entry_a, entry_b],
            "roles": [{"id": "test_role", "name": "Test Role", "mandate_behaviors": []}],
        })
        report = SessionAnalyzer(config_path).analyze(
            log_path, agent_id="test-agent", role_id="test_role"
        )
        assert report.combination == strategy
        scores[strategy] = report.overall_posterior

    # node_a scores .1 + .4 = .5; node_b scores .1 + .7 = .8 accumulated, which
    # saturates to ~.757. Corroboration lifts noisy-OR above the strongest
    # single hypothesis; the average pulls it below.
    assert scores["max"] == pytest.approx(0.7574, abs=1e-4)
    assert scores["noisy_or"] > scores["max"] > scores["weighted_average"]


def test_unknown_combination_in_config_fails_at_load(tmp_path):
    config_path = tmp_path / "agentic_detection" / "config" / "hypothesis.json"
    _write_json(config_path, {
        "session_combination": "nonsense",
        "generic_behaviors": [],
        "roles": [],
    })
    with pytest.raises(ValueError):
        SessionAnalyzer(config_path)


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
