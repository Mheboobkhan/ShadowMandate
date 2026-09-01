"""Integration tests replaying the real sample data through SessionAnalyzer
and pinning the exact outcomes documented in the README / demonstrated in
the Quickstart. If these break, the README's own worked examples are wrong.
"""

import pytest

from agentic_detection.session_analyzer import SessionAnalyzer

from conftest import CONFIG_PATH, DATA_ROOT


def test_iam_investigator_session_flags_critical_mandate_violation():
    report = SessionAnalyzer(CONFIG_PATH).analyze(
        DATA_ROOT / "iam_investigator_session.ndjson",
        agent_id="iam-agent-01",
        role_id="iam_investigator",
    )

    mandate_result = next(
        r for r in report.hypothesis_results if r.id == "iam_investigator_mandate_violation"
    )
    assert mandate_result.fired is True
    assert mandate_result.verdict.posterior_probability == pytest.approx(1.0)
    assert mandate_result.verdict.risk_level == "CRITICAL"

    # The whole point of the demo: every *generic* hypothesis on this same
    # log stays well below the mandate hypothesis's CRITICAL verdict.
    generic_results = [r for r in report.hypothesis_results if not r.is_mandate]
    assert generic_results, "expected at least one generic hypothesis to run"
    for r in generic_results:
        assert r.verdict.risk_level in ("MINIMAL", "LOW", "MEDIUM"), (
            f"{r.id} unexpectedly reached {r.verdict.risk_level} on the generic-only pass"
        )

    assert report.overall_verdict == "DRIFT_DETECTED"
    assert report.overall_risk_level == "HIGH"


def test_external_connection_single_hypothesis_on_app_log():
    report = SessionAnalyzer(CONFIG_PATH).analyze(
        DATA_ROOT / "app.log",
        agent_id="ollama-test",
        behavior_id="external_connection",
    )
    result = report.hypothesis_results[0]
    assert result.verdict.posterior_probability == pytest.approx(0.82)
    assert result.verdict.risk_level == "CRITICAL"
