"""Tests for BayesianNetworkEngine: base rate, weights, boosts, and the
manual/pgmpy backend equivalence the project claims in its docs."""

import pytest

from agentic_detection.bn_engine import BayesianNetworkEngine

from conftest import HYPOTHESES_ROOT

EXTERNAL_CONNECTION_BN = HYPOTHESES_ROOT / "generic/external_connection/bn_external_connection.json"
IAM_MANDATE_BN = HYPOTHESES_ROOT / "roles/iam_investigator/bn_iam_investigator_mandate_violation.json"


def test_no_evidence_returns_base_rate():
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    assert bn.infer_drift_probability({}) == pytest.approx(0.05)


def test_single_weight_applies():
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    posterior = bn.infer_drift_probability({"external_url_access": 1})
    assert posterior == pytest.approx(0.35)


def test_boost_requires_both_named_nodes_not_a_substring_match():
    """Regression test: a boost key like
    'external_url_access_and_external_download_boost' must require BOTH
    named nodes to be active. A prior bug matched it via substring
    containment, so firing external_url_access alone silently borrowed
    part of the boost meant for firing both nodes together."""
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    posterior = bn.infer_drift_probability({
        "external_url_access": 1,
        "external_download": 0,
    })
    assert posterior == pytest.approx(0.35)


def test_boost_applies_when_both_named_nodes_fire():
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    posterior = bn.infer_drift_probability({
        "external_url_access": 1,
        "external_download": 1,
    })
    # base .05 + .30 + .35 (weights) + .12 (boost) = .82
    assert posterior == pytest.approx(0.82)


def test_connection_rule_match_contributes_to_posterior():
    """Regression test: connection_rule_match was matched by the rule
    engine and reported as fired, but was missing from this bn config's
    evidence_nodes, so it never moved the posterior off the base rate."""
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    posterior = bn.infer_drift_probability({"connection_rule_match": 1})
    assert posterior > 0.05
    assert posterior == pytest.approx(0.20)


def test_all_boost_requires_every_evidence_node():
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)

    three_of_four = bn.explain({
        "external_url_access": 1,
        "external_download": 1,
        "dns_lookup_failure": 1,
        "connection_rule_match": 0,
    })
    all_four = bn.explain({
        "external_url_access": 1,
        "external_download": 1,
        "dns_lookup_failure": 1,
        "connection_rule_match": 1,
    })

    # Both saturate the clamp, so compare the raw (pre-clamp) probability
    # and which boosts actually applied instead of the clamped posterior.
    assert "all_boost" not in three_of_four["breakdown"]["boosts_applied"]
    assert "all_boost" in all_four["breakdown"]["boosts_applied"]
    assert all_four["breakdown"]["raw_probability"] > three_of_four["breakdown"]["raw_probability"]
    assert all_four["posterior_probability"] == pytest.approx(1.0)  # clamped


def test_iam_mandate_walkthrough_numbers():
    """Pins the exact numbers documented in the README's worked example."""
    bn = BayesianNetworkEngine(IAM_MANDATE_BN)
    assert bn.explain({})["posterior_probability"] == pytest.approx(0.05)
    assert bn.explain({"credential_reset_or_rotation": 1})["posterior_probability"] == pytest.approx(0.60)
    assert bn.explain({
        "credential_reset_or_rotation": 1,
        "credential_harvesting": 1,
    })["posterior_probability"] == pytest.approx(1.0)


def test_manual_and_pgmpy_backends_agree():
    """The module docstring promises the pgmpy and manual backends are
    mathematically equivalent. Skipped unless pgmpy is installed."""
    pytest.importorskip("pgmpy")
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    assert bn.backend == "pgmpy"

    combos = [
        {},
        {"external_url_access": 1},
        {"external_url_access": 1, "external_download": 1},
        {
            "external_url_access": 1,
            "external_download": 1,
            "dns_lookup_failure": 1,
            "connection_rule_match": 1,
        },
    ]
    for evidence in combos:
        full_evidence = {node: int(bool(evidence.get(node, 0))) for node in bn.evidence_nodes}
        manual = bn._infer_manual(full_evidence)
        via_pgmpy = bn._infer_pgmpy(full_evidence)
        assert manual == pytest.approx(via_pgmpy, abs=1e-6)
