"""
bn_engine.py - Bayesian Network Inference
============================================

Builds a small Bayesian Network per behavior (evidence nodes -> hypothesis
node) and computes the posterior probability of drift given observed
evidence.

Scoring is a two-step process:

1. **Accumulate** the human-authored numbers from the behavior's bn_*.json:
   a ``base`` rate, a ``<node>_weight`` for every evidence node that fired,
   and any ``_boost`` for evidence that co-occurred. This is unchanged - the
   security team's numbers are still the only inputs.

2. **Saturate** the accumulated score into [0, 1) with a soft knee rather
   than a hard clamp. A hard clamp collapsed every high-evidence combination
   onto exactly 1.00: for the IAM mandate hypothesis, four of the eight
   possible evidence combinations all returned "certainty", so the tool could
   not distinguish two keyword hits from three, and reported confidence 1.0
   either way. The soft knee leaves everything at or below
   ``saturation_knee`` (default 0.6) *numerically identical* to the old
   behavior, and compresses the region above it onto an asymptote that
   approaches but never reaches 1.0. Ordering is preserved, so more evidence
   always scores strictly higher than less.

Evidence may be binary (the pattern channel, always 0 or 1) or a probability
in [0, 1] (the optional Jev channel - see jev_channel.py). Soft evidence
scales a node's weight proportionally, and scales a boost by the product of
the probabilities of the nodes it names, so both reduce exactly to the
binary behavior when every value is 0 or 1.

If `pgmpy` is installed, *binary* inference is performed with a real
DiscreteBayesianNetwork + VariableElimination, whose CPD table is generated
from the same formula. Soft evidence cannot be expressed as hard evidence in
that query, so it always uses the closed-form path.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config_validation import check_bn_config
from .utils import PathLike, load_json, saturate

# The guards below catch Exception, not just ImportError, and that breadth is
# deliberate. pgmpy is an optional accelerator whose absence is a fully
# supported configuration, but an *installed* pgmpy can fail to import for
# reasons that are not ImportError - pgmpy 1.x uses `int | float` annotations
# at class-definition time, which raises TypeError on Python 3.9. A narrower
# guard let that take the whole package down on import, turning "the optional
# dependency is the wrong version" into "nothing runs at all".
try:
    from pgmpy.models import DiscreteBayesianNetwork as _PgmpyBNModel
except Exception:  # pragma: no cover - depends on the installed pgmpy
    try:
        from pgmpy.models import BayesianNetwork as _PgmpyBNModel  # type: ignore
    except Exception:
        _PgmpyBNModel = None

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    _PGMPY_AVAILABLE = _PgmpyBNModel is not None
except Exception:  # pragma: no cover - depends on the installed pgmpy
    TabularCPD = None  # type: ignore
    VariableElimination = None  # type: ignore
    _PGMPY_AVAILABLE = False


#: Score at/below which the accumulated probability passes through unchanged.
DEFAULT_SATURATION_KNEE = 0.6

#: Evidence probability at/above which a node counts as "fired".
DEFAULT_EVIDENCE_FIRE_THRESHOLD = 0.5


class BayesianNetworkEngine:
    """Computes P(drift | evidence) for a single behavior's Bayesian Network."""

    def __init__(self, bn_config_path: PathLike, validate: bool = True):
        self.config = load_json(bn_config_path)
        if validate:
            check_bn_config(self.config, source=str(bn_config_path))

        self.behavior: str = self.config["behavior"]
        self.evidence_nodes: List[str] = list(self.config["evidence_nodes"])
        self.hypothesis_node: str = self.config.get(
            "hypothesis_node", f"{self.behavior}_drift"
        )
        self.cpd_parameters: Dict[str, float] = dict(self.config.get("cpd_parameters", {}))

        self.saturation_knee: float = float(
            self.cpd_parameters.get("saturation_knee", DEFAULT_SATURATION_KNEE)
        )
        self.evidence_fire_threshold: float = float(
            self.cpd_parameters.get(
                "evidence_fire_threshold", DEFAULT_EVIDENCE_FIRE_THRESHOLD
            )
        )

        self.backend = "pgmpy" if _PGMPY_AVAILABLE else "manual"
        self._model = None
        self._inference = None
        if self.backend == "pgmpy":
            self._build_pgmpy_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_evidence(self, evidence: Mapping[str, Any]) -> Dict[str, float]:
        """Coerce an evidence mapping to {node: probability} over all declared nodes.

        Nodes absent from `evidence` are treated as 0.0 (not detected). Values
        are clamped into [0, 1], so the binary 0/1 the pattern channel emits
        passes through untouched.
        """
        normalized: Dict[str, float] = {}
        for node in self.evidence_nodes:
            raw = evidence.get(node, 0.0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 1.0 if raw else 0.0
            normalized[node] = max(0.0, min(1.0, value))
        return normalized

    def is_binary(self, evidence: Mapping[str, Any]) -> bool:
        """True when every evidence value is exactly 0 or 1."""
        return all(v in (0.0, 1.0) for v in self.normalize_evidence(evidence).values())

    def fired_nodes(self, evidence: Mapping[str, Any]) -> List[str]:
        """Nodes whose evidence probability meets the fire threshold."""
        normalized = self.normalize_evidence(evidence)
        return [
            node for node, value in normalized.items()
            if value >= self.evidence_fire_threshold
        ]

    def infer_drift_probability(self, evidence: Mapping[str, Any]) -> float:
        """Compute the posterior P(drift=1 | evidence).

        `evidence` maps evidence-node names (a subset of, or equal to,
        self.evidence_nodes) to either 0/1 or a probability in [0, 1]. Nodes
        not present are treated as 0 (not observed / not detected).
        """
        full_evidence = self.normalize_evidence(evidence)

        if self.backend == "pgmpy" and self.is_binary(full_evidence):
            return self._infer_pgmpy({k: int(v) for k, v in full_evidence.items()})
        return self._infer_manual(full_evidence)

    def explain(self, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a breakdown of how the posterior was computed, for auditability."""
        full_evidence = self.normalize_evidence(evidence)
        posterior, breakdown = self._compute_manual_formula(full_evidence, explain=True)
        return {
            "behavior": self.behavior,
            "backend": self.backend,
            "evidence": full_evidence,
            "posterior_probability": posterior,
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Manual (dependency-free) inference
    # ------------------------------------------------------------------

    def _infer_manual(self, full_evidence: Dict[str, float]) -> float:
        posterior, _ = self._compute_manual_formula(full_evidence, explain=False)
        return posterior

    def _compute_manual_formula(
        self, full_evidence: Dict[str, float], explain: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        params = self.cpd_parameters
        base = params.get("base", 0.05)
        score = base
        breakdown: Dict[str, Any] = {
            "base": base,
            "weights_applied": {},
            "boosts_applied": {},
        }

        # Per-node weights: "<node>_weight", scaled by the node's probability.
        for node in self.evidence_nodes:
            p = full_evidence.get(node, 0.0)
            if p <= 0.0:
                continue
            weight_key = f"{node}_weight"
            if weight_key in params:
                contribution = params[weight_key] * p
                score += contribution
                if explain:
                    breakdown["weights_applied"][weight_key] = round(contribution, 6)

        # Pairwise / named interaction boosts: a param key of the form
        # "<node1>_and_<node2>_..._boost" is applied when every evidence node
        # named in it is active, scaled by the product of their probabilities.
        # "all_*_boost" keys are handled separately below.
        for key, value in params.items():
            if not key.endswith("_boost") or key.startswith("all_"):
                continue
            node_part = key[: -len("_boost")]
            required_nodes = node_part.split("_and_")
            if not all(n in self.evidence_nodes for n in required_nodes):
                continue
            joint = 1.0
            for n in required_nodes:
                joint *= full_evidence.get(n, 0.0)
            if joint > 0.0:
                score += value * joint
                if explain:
                    breakdown["boosts_applied"][key] = round(value * joint, 6)

        # Global "all evidence present" boost, e.g. "all_three_boost" / "all_boost",
        # scaled by the joint probability of every declared node.
        all_boost_keys = [k for k in params if k.startswith("all_") and k.endswith("_boost")]
        if all_boost_keys and self.evidence_nodes:
            joint = 1.0
            for node in self.evidence_nodes:
                joint *= full_evidence.get(node, 0.0)
            if joint > 0.0:
                for key in all_boost_keys:
                    score += params[key] * joint
                    if explain:
                        breakdown["boosts_applied"][key] = round(params[key] * joint, 6)

        posterior = saturate(score, knee=self.saturation_knee)
        if explain:
            breakdown["raw_score"] = round(score, 6)
            breakdown["saturation_knee"] = self.saturation_knee
            breakdown["saturated_probability"] = round(posterior, 6)
            # Kept under its historical name so existing tooling that reads
            # the breakdown keeps working.
            breakdown["raw_probability"] = round(score, 6)
        return posterior, (breakdown if explain else None)

    # ------------------------------------------------------------------
    # pgmpy-backed inference
    # ------------------------------------------------------------------

    def _build_pgmpy_model(self) -> None:
        edges = [(node, self.hypothesis_node) for node in self.evidence_nodes]
        model = _PgmpyBNModel(edges) if edges else _PgmpyBNModel()
        if not edges:
            model.add_node(self.hypothesis_node)

        # Uniform, non-informative priors for evidence nodes: they are always
        # supplied as hard evidence at query time, so their marginals never
        # actually influence the posterior.
        evidence_cpds = [
            TabularCPD(variable=node, variable_card=2, values=[[0.5], [0.5]])
            for node in self.evidence_nodes
        ]

        hypothesis_cpd = self._build_hypothesis_cpd()

        model.add_cpds(*evidence_cpds, hypothesis_cpd)
        model.check_model()

        self._model = model
        self._inference = VariableElimination(model)

    def _build_hypothesis_cpd(self):
        """Build the hypothesis node's TabularCPD from cpd_parameters.

        For every combination of evidence-node states, evaluate the weighted
        formula (same math as the manual backend) to get P(drift=1 | combo),
        then lay those out in pgmpy's expected column order.
        """
        n = len(self.evidence_nodes)
        combos = list(itertools.product([0, 1], repeat=n)) if n > 0 else [()]

        drift_true_row: List[float] = []
        drift_false_row: List[float] = []

        for combo in combos:
            combo_evidence = {node: float(v) for node, v in zip(self.evidence_nodes, combo)}
            p_drift, _ = self._compute_manual_formula(combo_evidence, explain=False)
            drift_true_row.append(p_drift)
            drift_false_row.append(1.0 - p_drift)

        evidence_card = [2] * n
        return TabularCPD(
            variable=self.hypothesis_node,
            variable_card=2,
            values=[drift_false_row, drift_true_row],
            evidence=self.evidence_nodes if n > 0 else None,
            evidence_card=evidence_card if n > 0 else None,
        )

    def _infer_pgmpy(self, full_evidence: Dict[str, int]) -> float:
        if not self.evidence_nodes:
            base, _ = self._compute_manual_formula({}, explain=False)
            return base
        try:
            result = self._inference.query(
                variables=[self.hypothesis_node],
                evidence=full_evidence,
                show_progress=False,
            )
            # state 1 == "drift"
            return float(result.values[1])
        except Exception:
            # If pgmpy inference fails for any reason, fall back to the
            # equivalent closed-form computation rather than crashing.
            return self._infer_manual({k: float(v) for k, v in full_evidence.items()})
