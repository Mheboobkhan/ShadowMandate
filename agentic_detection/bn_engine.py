"""
bn_engine.py - Bayesian Network Inference
============================================

Builds a small Bayesian Network per behavior (evidence nodes -> hypothesis
node) and computes the posterior probability of drift given observed
evidence.

If `pgmpy` is installed, inference is performed with a real
DiscreteBayesianNetwork + VariableElimination. The hypothesis node's CPD
table is generated from the human-tunable `cpd_parameters` in the behavior's
bn_*.json config (base rate, per-node weights, and interaction boosts).

If `pgmpy` is not available, the engine falls back to evaluating the exact
same weighted formula directly (no external dependency), so the module
still works out of the box.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

from .utils import PathLike, load_json, clamp

try:
    from pgmpy.models import DiscreteBayesianNetwork as _PgmpyBNModel
except ImportError:  # pragma: no cover - older pgmpy versions
    try:
        from pgmpy.models import BayesianNetwork as _PgmpyBNModel  # type: ignore
    except ImportError:
        _PgmpyBNModel = None

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    _PGMPY_AVAILABLE = _PgmpyBNModel is not None
except ImportError:
    TabularCPD = None  # type: ignore
    VariableElimination = None  # type: ignore
    _PGMPY_AVAILABLE = False


class BayesianNetworkEngine:
    """Computes P(drift | evidence) for a single behavior's Bayesian Network."""

    def __init__(self, bn_config_path: PathLike):
        self.config = load_json(bn_config_path)
        self.behavior: str = self.config["behavior"]
        self.evidence_nodes: List[str] = list(self.config["evidence_nodes"])
        self.hypothesis_node: str = self.config.get(
            "hypothesis_node", f"{self.behavior}_drift"
        )
        self.cpd_parameters: Dict[str, float] = dict(self.config.get("cpd_parameters", {}))

        self.backend = "pgmpy" if _PGMPY_AVAILABLE else "manual"
        self._model = None
        self._inference = None
        if self.backend == "pgmpy":
            self._build_pgmpy_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_drift_probability(self, evidence: Dict[str, int]) -> float:
        """Compute the posterior P(drift=1 | evidence).

        `evidence` should map evidence-node names (a subset of, or equal to,
        self.evidence_nodes) to 0/1. Nodes not present are treated as 0
        (not observed / not detected).
        """
        full_evidence = {node: int(bool(evidence.get(node, 0))) for node in self.evidence_nodes}

        if self.backend == "pgmpy":
            return self._infer_pgmpy(full_evidence)
        return self._infer_manual(full_evidence)

    def explain(self, evidence: Dict[str, int]) -> Dict[str, Any]:
        """Return a breakdown of how the posterior was computed, for auditability."""
        full_evidence = {node: int(bool(evidence.get(node, 0))) for node in self.evidence_nodes}
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

    def _infer_manual(self, full_evidence: Dict[str, int]) -> float:
        posterior, _ = self._compute_manual_formula(full_evidence, explain=False)
        return posterior

    def _compute_manual_formula(
        self, full_evidence: Dict[str, int], explain: bool = False
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        params = self.cpd_parameters
        base = params.get("base", 0.05)
        prob = base
        breakdown: Dict[str, Any] = {"base": base, "weights_applied": {}, "boosts_applied": {}}

        active_nodes = [n for n in self.evidence_nodes if full_evidence.get(n)]

        # Per-node weights: "<node>_weight"
        for node in active_nodes:
            weight_key = f"{node}_weight"
            if weight_key in params:
                prob += params[weight_key]
                if explain:
                    breakdown["weights_applied"][weight_key] = params[weight_key]

        # Pairwise / named interaction boosts: a param key of the form
        # "<node1>_and_<node2>_..._boost" is applied when every evidence node
        # named in it (joined by "_and_") is active. "all_*_boost" keys are
        # handled separately below.
        for key, value in params.items():
            if not key.endswith("_boost") or key.startswith("all_"):
                continue
            node_part = key[: -len("_boost")]
            required_nodes = node_part.split("_and_")
            if not all(n in self.evidence_nodes for n in required_nodes):
                continue
            if all(full_evidence.get(n) for n in required_nodes):
                prob += value
                if explain:
                    breakdown["boosts_applied"][key] = value

        # Global "all evidence present" boost, e.g. "all_three_boost" / "all_boost"
        all_boost_keys = [k for k in params if k.startswith("all_") and k.endswith("_boost")]
        if active_nodes and len(active_nodes) == len(self.evidence_nodes):
            for key in all_boost_keys:
                prob += params[key]
                if explain:
                    breakdown["boosts_applied"][key] = params[key]

        clamped = clamp(prob, 0.0, 1.0)
        if explain:
            breakdown["raw_probability"] = prob
            breakdown["clamped_probability"] = clamped
        return clamped, (breakdown if explain else None)

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
            combo_evidence = dict(zip(self.evidence_nodes, combo))
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
            return self._infer_manual(full_evidence)
