"""
session_analyzer.py - Role-Aware Session Analysis
=====================================================

Runs every applicable hypothesis - the full generic catalog plus a role's
mandate rules - against one agent log in a single pass, and combines the
individual per-hypothesis posteriors into one session-level drift verdict.

This is where the project's central thesis lives: a generic rule set alone
can't tell that an IAM-investigation agent resetting a credential is out of
bounds - it's the role's mandate hypothesis that catches it. Mandate
hypotheses are weighted more heavily than generic ones when combining scores
(config-driven `session_weight`, not hardcoded), so a mandate breach still
dominates the overall verdict even when a generic rule also fires alongside it.

Combining the scores
--------------------
The original combination was a weighted *average* over fired hypotheses, which
had a serious flaw: corroborating evidence lowered the verdict. On the sample
IAM session the mandate hypothesis scored at the top of the range, four
generic hypotheses correctly fired in the 0.40-0.50 band, and the average
dragged the session down below its own driving signal. More true positives
produced a calmer verdict - backwards for a detector.

The default is now a weighted noisy-OR, the standard Bayesian idiom for
"several independent causes of the same effect": the session score is never
below the strongest fired hypothesis, and additional fired hypotheses can only
raise it. `session_weight` becomes the exponent on each cause's survival term,
normalized against the heaviest fired hypothesis, so a mandate violation at
weight 2.5 still dominates a generic hit at 1.0.

The strategy is config-driven (`session_combination` in hypothesis.json), for
the same reason every other number here is: it is a judgment call about how
your team wants evidence to add up, and it belongs in your git history rather
than in this file. See docs/scoring.md for the trade-offs, including the
independence assumption noisy-OR makes and where this catalog violates it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .log_parser import AgentLogParser
from .rule_engine import RuleEngine
from .bn_engine import BayesianNetworkEngine
from .evidence_detector import EvidenceDetector
from .jev_channel import JevChannel, JevOutcome, JevQuestion, build_questions, merge_evidence
from .verdict_generator import Verdict, VerdictGenerator, risk_level_for
from .utils import PathLike, load_json

#: How per-hypothesis posteriors are combined into one session score.
COMBINATION_STRATEGIES = ("noisy_or", "max", "weighted_average")
DEFAULT_COMBINATION = "noisy_or"


@dataclass
class HypothesisSpec:
    """A resolved, ready-to-run hypothesis: rule file + BN config + how much it
    should count toward the overall session score."""

    id: str
    name: str
    dr_path: Path
    bn_path: Path
    session_weight: float = 1.0
    is_mandate: bool = False
    output_csv_path: Optional[Path] = None


@dataclass
class HypothesisResult:
    """One hypothesis's outcome within a session run."""

    id: str
    name: str
    is_mandate: bool
    session_weight: float
    fired: bool
    verdict: Verdict
    pattern_evidence: Dict[str, int] = field(default_factory=dict)
    jev_evidence: Dict[str, float] = field(default_factory=dict)

    def jev_only_nodes(self) -> List[str]:
        """Nodes the Jev channel raised that the pattern channel missed."""
        return sorted(
            node for node, p in self.jev_evidence.items()
            if p > 0.5 and not self.pattern_evidence.get(node)
        )

    def to_dict(self) -> Dict[str, Any]:
        d = self.verdict.to_dict()
        d.update(
            {
                "hypothesis_id": self.id,
                "hypothesis_name": self.name,
                "mandate_violation": self.is_mandate,
                "session_weight": self.session_weight,
                "fired": self.fired,
            }
        )
        if self.jev_evidence:
            d["evidence_channels"] = {
                "patterns": dict(self.pattern_evidence),
                "jev": {k: round(v, 4) for k, v in self.jev_evidence.items()},
                "jev_only_nodes": self.jev_only_nodes(),
            }
        return d


@dataclass
class SessionReport:
    """The consolidated result of analyzing one agent log against a set of hypotheses."""

    agent_id: str
    role_id: Optional[str]
    objective: Optional[str]
    overall_posterior: float
    overall_verdict: str
    overall_risk_level: str
    threshold: float
    hypothesis_results: List[HypothesisResult] = field(default_factory=list)
    combination: str = DEFAULT_COMBINATION
    jev: JevOutcome = field(default_factory=JevOutcome)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role_id": self.role_id,
            "objective": self.objective,
            "overall_posterior_probability": round(self.overall_posterior, 4),
            "overall_verdict": self.overall_verdict,
            "overall_risk_level": self.overall_risk_level,
            "threshold": self.threshold,
            "combination": self.combination,
            "evidence_channels": {
                "patterns": {"status": "ok"},
                "jev": self.jev.to_dict(),
            },
            "hypothesis_results": [r.to_dict() for r in self.hypothesis_results],
            "generated_at": self.generated_at,
        }


# ----------------------------------------------------------------------
# Combination strategies
# ----------------------------------------------------------------------

def combine_posteriors(
    results: Sequence[HypothesisResult],
    strategy: str = DEFAULT_COMBINATION,
) -> float:
    """Combine fired hypotheses into one session-level posterior.

    Every strategy here satisfies one property the old weighted average did
    not: adding another fired hypothesis never *lowers* the session score.
    """
    if strategy not in COMBINATION_STRATEGIES:
        raise ValueError(
            f"Unknown session_combination {strategy!r}; expected one of {list(COMBINATION_STRATEGIES)}"
        )

    fired = [r for r in results if r.fired]
    if not fired:
        # Nothing fired: fall back to the (base-rate) posterior of whatever ran.
        return min((r.verdict.posterior_probability for r in results), default=0.05)

    posteriors = [r.verdict.posterior_probability for r in fired]
    weights = [r.session_weight for r in fired]
    max_weight = max(weights)

    if strategy == "max" or max_weight <= 0:
        # All weights zero means every fired hypothesis is logged but excluded
        # from weighting; the fired signal must still not be dropped.
        return max(posteriors)

    if strategy == "weighted_average":
        weight_sum = sum(weights)
        if weight_sum <= 0:
            return max(posteriors)
        return sum(w * p for w, p in zip(weights, posteriors)) / weight_sum

    # noisy_or: 1 - PROD (1 - p_i) ** (w_i / w_max)
    survival = 1.0
    for weight, posterior in zip(weights, posteriors):
        exponent = weight / max_weight
        survival *= (1.0 - posterior) ** exponent
    return 1.0 - survival


class SessionAnalyzer:
    """Resolves hypotheses from config/hypothesis.json and runs them against a log."""

    def __init__(
        self,
        hypothesis_config_path: PathLike,
        jev_channel: Optional[JevChannel] = None,
    ):
        self.hypothesis_config_path = Path(hypothesis_config_path)
        self.config = load_json(self.hypothesis_config_path)
        # hypothesis.json lives in agentic_detection/config/; the paths inside
        # it (e.g. "hypotheses/generic/...") are relative to agentic_detection/.
        self.package_root = self.hypothesis_config_path.resolve().parent.parent
        self.jev_channel = jev_channel or JevChannel(enabled=False)
        self.combination = self.config.get("session_combination", DEFAULT_COMBINATION)
        if self.combination not in COMBINATION_STRATEGIES:
            raise ValueError(
                f"{self.hypothesis_config_path}: session_combination "
                f"{self.combination!r} must be one of {list(COMBINATION_STRATEGIES)}"
            )

    # ------------------------------------------------------------------
    # Hypothesis resolution
    # ------------------------------------------------------------------

    def _spec_from_entry(self, entry: Dict[str, Any], is_mandate: bool) -> HypothesisSpec:
        output_csv = entry.get("output_csv")
        return HypothesisSpec(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            dr_path=self.package_root / entry["detection_rules"],
            bn_path=self.package_root / entry["bn_config"],
            session_weight=float(entry.get("session_weight", 1.0)),
            is_mandate=is_mandate,
            output_csv_path=self.package_root / output_csv if output_csv else None,
        )

    def generic_specs(self) -> List[HypothesisSpec]:
        return [self._spec_from_entry(e, is_mandate=False) for e in self.config.get("generic_behaviors", [])]

    def _all_behavior_entries(self) -> List[Tuple[Dict[str, Any], bool]]:
        entries = [(e, False) for e in self.config.get("generic_behaviors", [])]
        for role in self.config.get("roles", []):
            entries += [(e, True) for e in role.get("mandate_behaviors", [])]
        return entries

    def behavior_spec(self, behavior_id: str) -> HypothesisSpec:
        for entry, is_mandate in self._all_behavior_entries():
            if entry["id"] == behavior_id:
                return self._spec_from_entry(entry, is_mandate=is_mandate)
        available = [e["id"] for e, _ in self._all_behavior_entries()]
        raise SystemExit(f"Unknown behavior '{behavior_id}'. Available behaviors: {available}")

    def _find_role(self, role_id: str) -> Dict[str, Any]:
        for role in self.config.get("roles", []):
            if role["id"] == role_id:
                return role
        available = [r["id"] for r in self.config.get("roles", [])]
        raise SystemExit(f"Unknown role '{role_id}'. Available roles: {available}")

    def resolve_for_role(self, role_id: str) -> Tuple[str, List[HypothesisSpec]]:
        """Return (objective, specs) for the generic catalog + this role's mandate rules."""
        role = self._find_role(role_id)
        objective = role.get("name", role_id)
        manifest_path = role.get("manifest")
        if manifest_path:
            manifest = load_json(self.package_root / manifest_path)
            objective = manifest.get("objective", objective)
        specs = self.generic_specs()
        specs += [self._spec_from_entry(e, is_mandate=True) for e in role.get("mandate_behaviors", [])]
        return objective, specs

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _collect_jev_questions(
        self, dr_configs: Dict[str, Dict[str, Any]]
    ) -> List[JevQuestion]:
        """Gather every opted-in evidence question across every hypothesis.

        Jev answers all questions in a request in one parallel pass, so asking
        the whole catalog at once costs roughly one round trip rather than one
        per hypothesis.
        """
        questions: List[JevQuestion] = []
        for dr_config in dr_configs.values():
            questions.extend(build_questions(dr_config))
        return questions

    def _run_one(
        self,
        events: List[Dict[str, Any]],
        spec: HypothesisSpec,
        agent_id: str,
        threshold: float,
        jev_probabilities: Optional[Dict[str, float]] = None,
    ) -> HypothesisResult:
        rule_engine = RuleEngine()
        behavior_id = rule_engine.load_rule_file(spec.dr_path)
        pattern_evidence = rule_engine.evaluate_logs(events, behavior_id)
        if spec.output_csv_path:
            EvidenceDetector(parser=None, rule_engine=rule_engine).build_evidence_csv(
                events, behavior_id, spec.output_csv_path, agent_id=agent_id
            )

        jev_probabilities = jev_probabilities or {}
        evidence = merge_evidence(pattern_evidence, jev_probabilities)

        bn = BayesianNetworkEngine(spec.bn_path)
        posterior = bn.infer_drift_probability(evidence)
        verdict_gen = VerdictGenerator(threshold=threshold)
        verdict = verdict_gen.generate_verdict(
            agent_id=agent_id,
            behavior_id=behavior_id,
            posterior_probability=posterior,
            evidence=evidence,
        )
        # A node counts as fired once its evidence probability clears the
        # BN's threshold - identical to `any(evidence)` for binary patterns,
        # and meaningful for soft Jev probabilities.
        fired = bool(bn.fired_nodes(evidence))
        return HypothesisResult(
            id=spec.id,
            name=spec.name,
            is_mandate=spec.is_mandate,
            session_weight=spec.session_weight,
            fired=fired,
            verdict=verdict,
            pattern_evidence=pattern_evidence,
            jev_evidence={k: v for k, v in jev_probabilities.items() if v > 0},
        )

    def analyze(
        self,
        log_file: PathLike,
        agent_id: str,
        role_id: Optional[str] = None,
        behavior_id: Optional[str] = None,
        threshold: float = 0.5,
    ) -> SessionReport:
        """Parse `log_file` once and run it against every applicable hypothesis.

        Exactly one of `role_id` (generic catalog + that role's mandate rules)
        or `behavior_id` (a single hypothesis, for ad-hoc testing) must be given.
        """
        if not role_id and not behavior_id:
            raise ValueError("analyze() requires either role_id or behavior_id")
        if role_id and behavior_id:
            raise ValueError("analyze() accepts only one of role_id or behavior_id")

        parser = AgentLogParser()
        events = parser.parse_log_file(log_file)

        if behavior_id:
            objective = None
            specs = [self.behavior_spec(behavior_id)]
        else:
            objective, specs = self.resolve_for_role(role_id)

        # One Jev call for the whole session, before any hypothesis runs.
        dr_configs = {spec.id: load_json(spec.dr_path) for spec in specs}
        jev_outcome = self.jev_channel.evaluate(
            events, self._collect_jev_questions(dr_configs)
        )

        results = [
            self._run_one(
                events,
                spec,
                agent_id,
                threshold,
                jev_probabilities=jev_outcome.for_behavior(
                    dr_configs[spec.id].get("behavior", spec.id)
                ),
            )
            for spec in specs
        ]

        overall_posterior = combine_posteriors(results, strategy=self.combination)
        overall_risk_level = risk_level_for(overall_posterior)
        overall_verdict = "DRIFT_DETECTED" if overall_posterior >= threshold else "NO_DRIFT"

        return SessionReport(
            agent_id=agent_id,
            role_id=role_id,
            objective=objective,
            overall_posterior=overall_posterior,
            overall_verdict=overall_verdict,
            overall_risk_level=overall_risk_level,
            threshold=threshold,
            hypothesis_results=results,
            combination=self.combination,
            jev=jev_outcome,
        )
