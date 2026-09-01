"""
verdict_generator.py - Verdict & Risk Assessment
===================================================

Converts a posterior drift probability into an actionable verdict: a risk
level, a pass/fail determination against a threshold, and a human-readable
recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# (lower_bound_inclusive, upper_bound_exclusive, label)
_RISK_BANDS = [
    (0.0, 0.4, "MINIMAL"),
    (0.4, 0.5, "LOW"),
    (0.5, 0.6, "MEDIUM"),
    (0.6, 0.75, "HIGH"),
    (0.75, 1.0 + 1e-9, "CRITICAL"),
]

_RECOMMENDATIONS = {
    "MINIMAL": "No action needed. Behavior is consistent with expected agent activity.",
    "LOW": "Monitor behavior. No immediate action required, but continue logging.",
    "MEDIUM": "Review in context. Correlate with other signals before deciding on action.",
    "HIGH": "High priority review. Escalate to security team for manual investigation.",
    "CRITICAL": "Immediate action required. Block {behavior} activities and investigate agent.",
}


def risk_level_for(posterior: float) -> str:
    """Map a posterior probability in [0, 1] to a risk-level label."""
    for lo, hi, label in _RISK_BANDS:
        if lo <= posterior < hi:
            return label
    return "CRITICAL" if posterior >= 0.75 else "MINIMAL"


@dataclass
class Verdict:
    agent_id: str
    behavior_id: str
    posterior_probability: float
    verdict: str  # "DRIFT_DETECTED" or "NO_DRIFT"
    risk_level: str
    confidence: float
    threshold: float
    recommendation: str
    evidence: Dict[str, int] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "behavior": self.behavior_id,
            "posterior_probability": round(self.posterior_probability, 4),
            "verdict": self.verdict,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "threshold": self.threshold,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "generated_at": self.generated_at,
        }


class VerdictGenerator:
    """Turns a posterior probability into a Verdict object."""

    def __init__(self, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def generate_verdict(
        self,
        agent_id: str,
        behavior_id: str,
        posterior_probability: float,
        evidence: Optional[Dict[str, int]] = None,
    ) -> Verdict:
        posterior = max(0.0, min(1.0, posterior_probability))
        risk_level = risk_level_for(posterior)
        is_drift = posterior >= self.threshold

        # Confidence: how far the posterior sits from the decision boundary,
        # normalized so a posterior at 0 or 1 yields full confidence and a
        # posterior right at the threshold yields ~0 confidence.
        if is_drift:
            span = max(1.0 - self.threshold, 1e-9)
            confidence = (posterior - self.threshold) / span
        else:
            span = max(self.threshold, 1e-9)
            confidence = (self.threshold - posterior) / span
        confidence = max(0.0, min(1.0, confidence))

        recommendation = _RECOMMENDATIONS[risk_level].format(behavior=behavior_id)

        return Verdict(
            agent_id=agent_id,
            behavior_id=behavior_id,
            posterior_probability=posterior,
            verdict="DRIFT_DETECTED" if is_drift else "NO_DRIFT",
            risk_level=risk_level,
            confidence=confidence,
            threshold=self.threshold,
            recommendation=recommendation,
            evidence=evidence or {},
        )
