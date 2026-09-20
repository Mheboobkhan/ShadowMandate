"""
config_validation.py - Fail-fast validation for human-authored configs
=======================================================================

Every number that moves a posterior in this project is typed by a human into
a ``bn_*.json`` file. That is the design (see README's human-in-the-loop
section) - but it also means a typo in a parameter key is a silent, permanent
downgrade of a detection: ``password_pattern_weight`` instead of
``password_patterns_weight`` is simply never read, and the hypothesis scores
at its base rate forever with nothing in the output to say so.

These validators turn that class of mistake into a loud error at load time.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Iterable, List, Set


class ConfigValidationError(ValueError):
    """Raised when a hypothesis config contains keys the engine cannot honor."""


# Scalar tuning knobs that are not tied to a specific evidence node.
_SCALAR_PARAMS = {
    "base",
    "saturation_knee",
    "evidence_fire_threshold",
}

_WEIGHT_SUFFIX = "_weight"
_BOOST_SUFFIX = "_boost"
_NODE_JOINER = "_and_"


def _suggest(bad_key: str, candidates: Iterable[str]) -> str:
    """Return a ' (did you mean X?)' fragment, or '' if nothing is close."""
    matches = difflib.get_close_matches(bad_key, list(candidates), n=1, cutoff=0.7)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def expected_parameter_keys(evidence_nodes: Iterable[str]) -> Set[str]:
    """The full set of per-node parameter keys the engine knows how to read.

    Interaction boosts are combinatorial, so this returns only the scalar and
    single-node keys; boost keys are validated structurally instead. The set is
    used to generate 'did you mean' suggestions for mistyped keys.
    """
    nodes = list(evidence_nodes)
    keys = set(_SCALAR_PARAMS)
    keys.update(f"{node}{_WEIGHT_SUFFIX}" for node in nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            keys.add(f"{a}{_NODE_JOINER}{b}{_BOOST_SUFFIX}")
    return keys


def validate_cpd_parameters(
    behavior: str,
    evidence_nodes: Iterable[str],
    cpd_parameters: Dict[str, Any],
) -> List[str]:
    """Check every cpd_parameters key against the declared evidence nodes.

    Returns a list of human-readable error strings (empty when the config is
    clean). Every key must be one of:

    * a scalar knob: ``base``, ``saturation_knee``, ``evidence_fire_threshold``
    * a per-node weight: ``<node>_weight``
    * an interaction boost: ``<node_a>_and_<node_b>[_and_...]_boost``
    * a global boost: ``all_<anything>_boost`` (applies when every node fires)
    """
    nodes = set(evidence_nodes)
    known = expected_parameter_keys(nodes)
    errors: List[str] = []

    for key, value in cpd_parameters.items():
        if not _is_number(value):
            errors.append(f"{behavior}: parameter {key!r} must be a number, got {value!r}")
            continue

        if key in _SCALAR_PARAMS:
            continue

        if key.startswith("all_") and key.endswith(_BOOST_SUFFIX):
            # Global "every node fired" boost - the middle is a free-form label
            # ("all_boost", "all_three_boost"), so there is nothing to check.
            continue

        if key.endswith(_BOOST_SUFFIX):
            required = key[: -len(_BOOST_SUFFIX)].split(_NODE_JOINER)
            unknown = [n for n in required if n not in nodes]
            if unknown:
                errors.append(
                    f"{behavior}: boost key {key!r} references undeclared evidence "
                    f"node(s) {unknown}{_suggest(key, known)}"
                )
            elif len(required) < 2:
                errors.append(
                    f"{behavior}: boost key {key!r} names only one node; a boost "
                    f"must join at least two with '{_NODE_JOINER}'"
                )
            continue

        if key.endswith(_WEIGHT_SUFFIX):
            node = key[: -len(_WEIGHT_SUFFIX)]
            if node not in nodes:
                errors.append(
                    f"{behavior}: weight key {key!r} does not match any declared "
                    f"evidence node{_suggest(key, known)}"
                )
            continue

        errors.append(
            f"{behavior}: unrecognized parameter {key!r} - it will never be read"
            f"{_suggest(key, known)}"
        )

    # A declared node with no weight can still be intentional (it may only
    # appear inside a boost), so only flag the case where it appears nowhere.
    for node in sorted(nodes):
        mentioned = any(node in key for key in cpd_parameters)
        if not mentioned:
            errors.append(
                f"{behavior}: evidence node {node!r} is declared but appears in no "
                f"weight or boost - it can never move the posterior"
            )

    return errors


def _is_number(value: Any) -> bool:
    """True for a real number. Booleans are excluded: `True` is an int in
    Python, and a `true` in a config file is a typo, not a weight of 1.0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_ranges(behavior: str, cpd_parameters: Dict[str, Any]) -> List[str]:
    """Check the scalar knobs sit in the ranges the engine assumes.

    Non-numeric values are skipped here; validate_cpd_parameters already
    reports them, and re-reporting a type error as a range error would just
    bury the real message.
    """
    errors: List[str] = []

    base = cpd_parameters.get("base", 0.05)
    if _is_number(base) and not 0.0 <= base <= 1.0:
        errors.append(f"{behavior}: 'base' must be in [0.0, 1.0], got {base}")

    knee = cpd_parameters.get("saturation_knee", None)
    if _is_number(knee) and not 0.0 < knee < 1.0:
        errors.append(f"{behavior}: 'saturation_knee' must be in (0.0, 1.0), got {knee}")

    thresh = cpd_parameters.get("evidence_fire_threshold", None)
    if _is_number(thresh) and not 0.0 < thresh <= 1.0:
        errors.append(
            f"{behavior}: 'evidence_fire_threshold' must be in (0.0, 1.0], got {thresh}"
        )

    for key, value in cpd_parameters.items():
        if not (key.endswith(_WEIGHT_SUFFIX) or key.endswith(_BOOST_SUFFIX)):
            continue
        if _is_number(value) and not 0.0 <= value <= 1.0:
            errors.append(f"{behavior}: {key!r} must be in [0.0, 1.0], got {value}")

    return errors


def validate_bn_config(config: Dict[str, Any], source: str = "<bn config>") -> List[str]:
    """Validate one parsed bn_*.json document. Returns a list of errors."""
    behavior = config.get("behavior")
    if not behavior:
        return [f"{source}: missing required 'behavior' key"]

    nodes = config.get("evidence_nodes")
    if not isinstance(nodes, list):
        return [f"{behavior}: 'evidence_nodes' must be a list, got {nodes!r}"]

    duplicates = {n for n in nodes if nodes.count(n) > 1}
    errors = [f"{behavior}: duplicate evidence node(s) {sorted(duplicates)}"] if duplicates else []

    params = config.get("cpd_parameters", {})
    if not isinstance(params, dict):
        return errors + [f"{behavior}: 'cpd_parameters' must be an object, got {params!r}"]

    errors += validate_cpd_parameters(behavior, nodes, params)
    errors += validate_ranges(behavior, params)
    return errors


def check_bn_config(config: Dict[str, Any], source: str = "<bn config>") -> None:
    """Validate a bn config and raise ConfigValidationError if anything is wrong."""
    errors = validate_bn_config(config, source=source)
    if errors:
        joined = "\n  - ".join(errors)
        raise ConfigValidationError(
            f"Invalid Bayesian Network config in {source}:\n  - {joined}"
        )
