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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .log_parser import AgentLogParser
from .rule_engine import RuleEngine
from .bn_engine import BayesianNetworkEngine
from .evidence_detector import EvidenceDetector
from .verdict_generator import Verdict, VerdictGenerator, risk_level_for
from .utils import PathLike, load_json


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
            "hypothesis_results": [r.to_dict() for r in self.hypothesis_results],
            "generated_at": self.generated_at,
        }


class SessionAnalyzer:
    """Resolves hypotheses from config/hypothesis.json and runs them against a log."""

    def __init__(self, hypothesis_config_path: PathLike):
        self.hypothesis_config_path = Path(hypothesis_config_path)
        self.config = load_json(self.hypothesis_config_path)
        # hypothesis.json lives in agentic_detection/config/; the paths inside
        # it (e.g. "hypotheses/generic/...") are relative to agentic_detection/.
        self.package_root = self.hypothesis_config_path.resolve().parent.parent

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

    def _run_one(self, events: List[Dict[str, Any]], spec: HypothesisSpec, agent_id: str, threshold: float) -> HypothesisResult:
        rule_engine = RuleEngine()
        behavior_id = rule_engine.load_rule_file(spec.dr_path)
        evidence = rule_engine.evaluate_logs(events, behavior_id)
        if spec.output_csv_path:
            EvidenceDetector(parser=None, rule_engine=rule_engine).build_evidence_csv(
                events, behavior_id, spec.output_csv_path, agent_id=agent_id
            )
        bn = BayesianNetworkEngine(spec.bn_path)
        posterior = bn.infer_drift_probability(evidence)
        verdict_gen = VerdictGenerator(threshold=threshold)
        verdict = verdict_gen.generate_verdict(
            agent_id=agent_id,
            behavior_id=behavior_id,
            posterior_probability=posterior,
            evidence=evidence,
        )
        fired = any(evidence.values())
        return HypothesisResult(
            id=spec.id,
            name=spec.name,
            is_mandate=spec.is_mandate,
            session_weight=spec.session_weight,
            fired=fired,
            verdict=verdict,
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

        results = [self._run_one(events, spec, agent_id, threshold) for spec in specs]

        fired_results = [r for r in results if r.fired]
        weight_sum = sum(r.session_weight for r in fired_results)
        if fired_results and weight_sum > 0:
            weighted_sum = sum(r.session_weight * r.verdict.posterior_probability for r in fired_results)
            overall_posterior = weighted_sum / weight_sum
        elif fired_results:
            # All fired hypotheses have session_weight 0 (logged but excluded
            # from the weighted average): fall back to the highest of their
            # posteriors so the fired signal isn't silently dropped.
            overall_posterior = max(r.verdict.posterior_probability for r in fired_results)
        else:
            # Nothing fired: fall back to the (base-rate) posterior of whatever ran.
            overall_posterior = min((r.verdict.posterior_probability for r in results), default=0.05)

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
        )
