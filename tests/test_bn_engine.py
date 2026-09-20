"""Tests for BayesianNetworkEngine: base rate, weights, boosts, and the
manual/pgmpy backend equivalence the project claims in its docs."""

import itertools
import json

import pytest

from agentic_detection.bn_engine import _PGMPY_AVAILABLE, BayesianNetworkEngine
from agentic_detection.config_validation import ConfigValidationError

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
    explained = bn.explain({
        "external_url_access": 1,
        "external_download": 1,
    })
    # base .05 + .30 + .35 (weights) + .12 (boost) = .82 accumulated score,
    # which sits above the 0.6 saturation knee and compresses to ~0.769.
    assert explained["breakdown"]["raw_score"] == pytest.approx(0.82)
    assert explained["posterior_probability"] == pytest.approx(0.7692, abs=1e-4)


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

    assert "all_boost" not in three_of_four["breakdown"]["boosts_applied"]
    assert "all_boost" in all_four["breakdown"]["boosts_applied"]
    assert all_four["breakdown"]["raw_score"] > three_of_four["breakdown"]["raw_score"]
    # The soft knee keeps these two distinguishable. Under the old hard clamp
    # both collapsed onto exactly 1.00 and the extra evidence was invisible.
    assert all_four["posterior_probability"] > three_of_four["posterior_probability"]
    assert all_four["posterior_probability"] < 1.0


def test_iam_mandate_walkthrough_numbers():
    """Pins the exact numbers documented in the README's worked example."""
    bn = BayesianNetworkEngine(IAM_MANDATE_BN)
    assert bn.explain({})["posterior_probability"] == pytest.approx(0.05)
    # Below the saturation knee, the posterior is still exactly base + weight.
    assert bn.explain({"credential_reset_or_rotation": 1})["posterior_probability"] == pytest.approx(0.60)
    two_nodes = bn.explain({
        "credential_reset_or_rotation": 1,
        "credential_harvesting": 1,
    })
    assert two_nodes["breakdown"]["raw_score"] == pytest.approx(1.30)
    assert two_nodes["posterior_probability"] == pytest.approx(0.9305, abs=1e-4)


def test_every_evidence_combination_scores_distinctly():
    """Regression test for the saturating-clamp bug.

    The hard clamp mapped four of these eight combinations onto exactly 1.00,
    so the tool reported certainty - and identical certainty - for two pieces
    of evidence and for three. Every combination must now be distinguishable,
    strictly ordered by how much evidence it carries, and short of certainty.
    """
    bn = BayesianNetworkEngine(IAM_MANDATE_BN)
    nodes = bn.evidence_nodes

    scored = {}
    for combo in itertools.product([0, 1], repeat=len(nodes)):
        posterior = bn.infer_drift_probability(dict(zip(nodes, combo)))
        assert posterior < 1.0, f"{combo} still reports certainty"
        scored[combo] = posterior

    assert len(set(scored.values())) == len(scored), "combinations collapsed onto shared values"

    # Superset of evidence must always score strictly higher than a subset.
    for a, pa in scored.items():
        for b, pb in scored.items():
            if a != b and all(x <= y for x, y in zip(a, b)):
                assert pb > pa, f"{b} did not outscore its subset {a}"


def test_soft_evidence_uses_the_closed_form_even_under_pgmpy():
    """Soft evidence cannot be expressed as hard evidence in a discrete BN
    query, so it must take the closed-form path regardless of backend."""
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    assert bn.is_binary({"external_url_access": 1}) is True
    assert bn.is_binary({"external_url_access": 0.5}) is False
    assert bn.infer_drift_probability({"external_url_access": 0.5}) == pytest.approx(0.20)


def test_soft_evidence_scales_a_weight_proportionally():
    """A Jev-style probability contributes a fraction of the node's weight,
    and reduces exactly to the binary behavior at 0 and 1."""
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    assert bn.infer_drift_probability({"external_url_access": 0.0}) == pytest.approx(0.05)
    assert bn.infer_drift_probability({"external_url_access": 0.5}) == pytest.approx(0.20)
    assert bn.infer_drift_probability({"external_url_access": 1.0}) == pytest.approx(0.35)


def test_soft_evidence_scales_a_boost_by_joint_probability():
    bn = BayesianNetworkEngine(EXTERNAL_CONNECTION_BN)
    explained = bn.explain({"external_url_access": 0.5, "external_download": 0.5})
    boost_key = "external_url_access_and_external_download_boost"
    # base .05 + .30*.5 + .35*.5 + .12*(.5*.5) = .405, below the knee
    assert explained["breakdown"]["boosts_applied"][boost_key] == pytest.approx(0.03)
    assert explained["posterior_probability"] == pytest.approx(0.405)


def test_invalid_config_is_rejected_at_load(tmp_path):
    """A mistyped weight key must fail loudly instead of silently scoring
    the hypothesis at its base rate forever."""
    bad = tmp_path / "bn_typo.json"
    bad.write_text(json.dumps({
        "behavior": "typo_demo",
        "evidence_nodes": ["password_patterns"],
        "cpd_parameters": {"base": 0.05, "password_pattern_weight": 0.4},
    }))
    with pytest.raises(ConfigValidationError) as exc:
        BayesianNetworkEngine(bad)
    assert "password_pattern_weight" in str(exc.value)
    assert "password_patterns_weight" in str(exc.value)  # suggestion


def test_manual_and_pgmpy_backends_agree():
    """The module docstring promises the pgmpy and manual backends are
    mathematically equivalent.

    Gated on _PGMPY_AVAILABLE rather than on pgmpy being importable: an
    installed-but-incompatible pgmpy (1.x on Python 3.9, say) is deliberately
    treated as absent, and this test must follow that decision rather than
    assert a backend the engine correctly refused to use.
    """
    if not _PGMPY_AVAILABLE:
        pytest.skip("pgmpy not available to this interpreter")
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
