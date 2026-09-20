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
    # Two of the three mandate evidence nodes fire (credential reset and
    # harvesting), accumulating .05 + .55 + .50 + .20 = 1.30. The soft knee
    # maps that to 0.93 - high, and honestly short of certainty. The old hard
    # clamp reported exactly 1.00 here, indistinguishable from all three
    # nodes firing.
    assert mandate_result.verdict.posterior_probability == pytest.approx(0.9305, abs=1e-4)
    assert mandate_result.verdict.posterior_probability < 1.0
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
    assert report.overall_risk_level == "CRITICAL"

    # The session must not score below its own driving hypothesis. Under the
    # old weighted average the four correctly-fired generic hypotheses pulled
    # this session down to 0.65 - beneath the mandate violation driving it.
    assert report.overall_posterior >= mandate_result.verdict.posterior_probability
    assert report.combination == "noisy_or"

    # Default run: the Jev channel is off and nothing was sent anywhere.
    assert report.jev.status == "disabled"
    assert report.jev.questions_asked == 0


def test_external_connection_single_hypothesis_on_app_log():
    report = SessionAnalyzer(CONFIG_PATH).analyze(
        DATA_ROOT / "app.log",
        agent_id="ollama-test",
        behavior_id="external_connection",
    )
    result = report.hypothesis_results[0]
    # external_url_access + external_download: base .05 + .30 + .35 + .12 boost
    # = .82 accumulated, saturating to ~.769.
    assert result.verdict.posterior_probability == pytest.approx(0.7692, abs=1e-4)
    assert result.verdict.risk_level == "CRITICAL"
