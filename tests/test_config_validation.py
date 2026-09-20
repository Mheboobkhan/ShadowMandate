"""Tests for fail-fast config validation.

The bug class these exist for: every number that moves a posterior is typed
by a human into a bn_*.json file, and a mistyped key used to be skipped in
silence - the hypothesis then scored at its base rate forever with nothing in
the output to say why.
"""

import json

import pytest

from agentic_detection.bn_engine import BayesianNetworkEngine
from agentic_detection.config_validation import (
    ConfigValidationError,
    check_bn_config,
    validate_bn_config,
)

from conftest import AGENTIC_ROOT, CONFIG_PATH


def _config(**overrides):
    base = {
        "behavior": "demo",
        "evidence_nodes": ["alpha", "beta"],
        "cpd_parameters": {
            "base": 0.05,
            "alpha_weight": 0.3,
            "beta_weight": 0.4,
            "alpha_and_beta_boost": 0.1,
        },
    }
    base.update(overrides)
    return base


def test_a_clean_config_passes():
    assert validate_bn_config(_config()) == []


def test_mistyped_weight_key_is_caught_with_a_suggestion():
    errors = validate_bn_config(_config(cpd_parameters={
        "base": 0.05,
        "alph_weight": 0.3,      # typo
        "beta_weight": 0.4,
    }))
    joined = "\n".join(errors)
    assert "alph_weight" in joined
    assert "alpha_weight" in joined, "no 'did you mean' suggestion offered"


def test_boost_naming_an_unknown_node_is_caught():
    errors = validate_bn_config(_config(cpd_parameters={
        "base": 0.05,
        "alpha_weight": 0.3,
        "beta_weight": 0.4,
        "alpha_and_gamma_boost": 0.1,
    }))
    assert any("gamma" in e for e in errors)


def test_single_node_boost_is_caught():
    errors = validate_bn_config(_config(cpd_parameters={
        "base": 0.05,
        "alpha_weight": 0.3,
        "beta_weight": 0.4,
        "alpha_boost": 0.1,
    }))
    assert any("at least two" in e for e in errors)


def test_all_boost_is_accepted_in_any_spelling():
    for key in ("all_boost", "all_three_boost", "all_four_of_them_boost"):
        params = {"base": 0.05, "alpha_weight": 0.3, "beta_weight": 0.4, key: 0.1}
        assert validate_bn_config(_config(cpd_parameters=params)) == []


def test_unreachable_evidence_node_is_caught():
    """A node declared in evidence_nodes but named by no weight or boost can
    never move the posterior - the same class of silent dead rule as the
    historical connection_rule_match bug."""
    errors = validate_bn_config(_config(cpd_parameters={
        "base": 0.05,
        "alpha_weight": 0.3,
    }))
    assert any("beta" in e and "never move" in e for e in errors)


def test_unrecognized_parameter_is_caught():
    errors = validate_bn_config(_config(cpd_parameters={
        "base": 0.05,
        "alpha_weight": 0.3,
        "beta_weight": 0.4,
        "confidence_multiplier": 2.0,
    }))
    assert any("confidence_multiplier" in e and "never be read" in e for e in errors)


def test_non_numeric_parameter_is_caught():
    errors = validate_bn_config(_config(cpd_parameters={
        "base": "high",
        "alpha_weight": 0.3,
        "beta_weight": 0.4,
    }))
    assert any("must be a number" in e for e in errors)


@pytest.mark.parametrize("params,fragment", [
    ({"base": 1.5, "alpha_weight": 0.3, "beta_weight": 0.4}, "'base'"),
    ({"base": 0.05, "alpha_weight": 1.4, "beta_weight": 0.4}, "alpha_weight"),
    ({"base": 0.05, "saturation_knee": 1.0, "alpha_weight": 0.3, "beta_weight": 0.4}, "saturation_knee"),
    ({"base": 0.05, "evidence_fire_threshold": 0.0, "alpha_weight": 0.3, "beta_weight": 0.4}, "evidence_fire_threshold"),
])
def test_out_of_range_scalars_are_caught(params, fragment):
    errors = validate_bn_config(_config(cpd_parameters=params))
    assert any(fragment in e for e in errors)


def test_duplicate_evidence_nodes_are_caught():
    errors = validate_bn_config(_config(evidence_nodes=["alpha", "alpha", "beta"]))
    assert any("duplicate" in e for e in errors)


def test_missing_behavior_key_is_caught():
    errors = validate_bn_config({"evidence_nodes": [], "cpd_parameters": {}})
    assert any("behavior" in e for e in errors)


def test_check_raises_with_every_problem_listed():
    with pytest.raises(ConfigValidationError) as exc:
        check_bn_config(_config(cpd_parameters={
            "base": 0.05,
            "alph_weight": 0.3,
            "alpha_and_gamma_boost": 0.1,
        }), source="bn_demo.json")
    message = str(exc.value)
    assert "bn_demo.json" in message
    assert "alph_weight" in message
    assert "gamma" in message


def test_engine_refuses_to_load_an_invalid_config(tmp_path):
    path = tmp_path / "bn_bad.json"
    path.write_text(json.dumps(_config(cpd_parameters={
        "base": 0.05,
        "alpha_weight": 0.3,
        "beta_wieght": 0.4,
    })))
    with pytest.raises(ConfigValidationError):
        BayesianNetworkEngine(path)


def test_validation_can_be_bypassed_for_ad_hoc_experiments(tmp_path):
    """Escape hatch for someone sketching a rule at a REPL - but never the
    default, and never what the CLI does."""
    path = tmp_path / "bn_bad.json"
    path.write_text(json.dumps(_config(cpd_parameters={
        "base": 0.05,
        "alpha_weight": 0.3,
        "beta_wieght": 0.4,
    })))
    engine = BayesianNetworkEngine(path, validate=False)
    assert engine.behavior == "demo"


def test_every_shipped_bn_config_is_valid():
    """The whole catalog must pass its own validator."""
    config = json.loads(CONFIG_PATH.read_text())
    entries = list(config["generic_behaviors"])
    for role in config["roles"]:
        entries.extend(role["mandate_behaviors"])

    failures = {}
    for entry in entries:
        path = AGENTIC_ROOT / entry["bn_config"]
        errors = validate_bn_config(json.loads(path.read_text()), source=str(path))
        if errors:
            failures[entry["id"]] = errors
    assert not failures, f"shipped configs failed validation: {failures}"
