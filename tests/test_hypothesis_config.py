"""Config-lint tests over config/hypothesis.json and every dr_*.json/bn_*.json
it references. This is the test class that would have caught the
connection_rule_match bug (an evidence node matched by the rule engine but
missing from its bn config's evidence_nodes) before it shipped.
"""

import json

import pytest

from conftest import AGENTIC_ROOT, CONFIG_PATH


def _load(path):
    with open(path) as f:
        return json.load(f)


def _all_behavior_entries(config):
    entries = list(config.get("generic_behaviors", []))
    for role in config.get("roles", []):
        entries.extend(role.get("mandate_behaviors", []))
    return entries


@pytest.fixture(scope="module")
def config():
    return _load(CONFIG_PATH)


@pytest.fixture(scope="module")
def all_entries(config):
    return _all_behavior_entries(config)


def test_every_entry_has_a_unique_id(all_entries):
    ids = [e["id"] for e in all_entries]
    assert len(ids) == len(set(ids)), f"duplicate hypothesis ids: {ids}"


def test_every_referenced_rule_file_exists(all_entries):
    for entry in all_entries:
        dr_path = AGENTIC_ROOT / entry["detection_rules"]
        bn_path = AGENTIC_ROOT / entry["bn_config"]
        assert dr_path.exists(), f"{entry['id']}: missing {dr_path}"
        assert bn_path.exists(), f"{entry['id']}: missing {bn_path}"


def test_role_manifests_exist(config):
    for role in config.get("roles", []):
        manifest = role.get("manifest")
        if manifest:
            assert (AGENTIC_ROOT / manifest).exists(), f"{role['id']}: missing manifest {manifest}"


def test_evidence_nodes_match_between_dr_and_bn(all_entries):
    for entry in all_entries:
        dr = _load(AGENTIC_ROOT / entry["detection_rules"])
        bn = _load(AGENTIC_ROOT / entry["bn_config"])
        dr_nodes = set(dr.get("evidence_mapping", {}).keys())
        bn_nodes = set(bn.get("evidence_nodes", []))
        assert dr_nodes == bn_nodes, (
            f"{entry['id']}: dr evidence_mapping {dr_nodes} != bn evidence_nodes {bn_nodes}"
        )


def test_boost_keys_reference_real_evidence_nodes(all_entries):
    for entry in all_entries:
        bn = _load(AGENTIC_ROOT / entry["bn_config"])
        evidence_nodes = set(bn.get("evidence_nodes", []))
        for key in bn.get("cpd_parameters", {}):
            if not key.endswith("_boost") or key.startswith("all_"):
                continue
            required = set(key[: -len("_boost")].split("_and_"))
            assert required <= evidence_nodes, (
                f"{entry['id']}: boost key {key!r} references unknown node(s) "
                f"{required - evidence_nodes}"
            )


def test_behavior_ids_are_globally_unique(all_entries):
    behaviors = [_load(AGENTIC_ROOT / entry["detection_rules"])["behavior"] for entry in all_entries]
    assert len(behaviors) == len(set(behaviors)), f"duplicate behavior ids: {behaviors}"


def test_session_weight_is_non_negative(all_entries):
    for entry in all_entries:
        assert entry.get("session_weight", 1.0) >= 0, f"{entry['id']}: negative session_weight"
