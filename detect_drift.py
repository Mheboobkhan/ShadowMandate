#!/usr/bin/env python3
"""
detect_drift.py - Agentic Behavioral Drift Detection CLI
============================================================

Two modes:

  Role-aware session scan (primary workflow) - runs the full generic
  hypothesis catalog plus a role's mandate rules in one pass, and combines
  them into one session-level drift verdict:

    python detect_drift.py <log_file> --agent-id <agent_id> --role <role_id> [--output verdict.json]

  Single-hypothesis mode (ad-hoc testing of one rule in isolation):

    python detect_drift.py <log_file> --agent-id <agent_id> --behavior <behavior_id> [--output verdict.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_detection import SessionAnalyzer, SessionReport, HypothesisResult
from agentic_detection.utils import save_json

PACKAGE_ROOT = Path(__file__).resolve().parent / "agentic_detection"
DEFAULT_HYPOTHESIS_CONFIG = PACKAGE_ROOT / "config" / "hypothesis.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze agent logs for behavioral drift using a Bayesian Network."
    )
    parser.add_argument("log_file", help="Path to the agent log file to analyze")
    parser.add_argument("--agent-id", default="agent-001", help="Agent identifier")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--role",
        help="Agent role id (e.g. iam_investigator). Runs the full generic "
        "hypothesis catalog plus this role's mandate rules.",
    )
    mode.add_argument(
        "--behavior",
        help="Run a single hypothesis id in isolation (ad-hoc rule testing).",
    )

    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection threshold 0.0-1.0"
    )
    parser.add_argument("--output", default=None, help="Save verdict/report JSON to file")
    parser.add_argument(
        "--hypothesis-config",
        default=str(DEFAULT_HYPOTHESIS_CONFIG),
        help="Path to config/hypothesis.json",
    )
    return parser.parse_args()


def print_behavior_mode_summary(report: SessionReport) -> None:
    """Single-hypothesis mode: prints the same step-by-step summary the tool
    has always printed for one behavior run."""
    result = report.hypothesis_results[0]
    verdict = result.verdict

    print("[2/4] Loading rules and detecting evidence...")
    print(f"  [OK] Evidence: {verdict.evidence}")
    print()
    print("[3/4] Running Bayesian Network inference...")
    print(f"  [OK] Posterior P(drift|evidence): {verdict.posterior_probability:.2f}")
    print()
    print("[4/4] Generating verdict...")
    print(f"  [OK] Verdict: {verdict.verdict}")
    print(f"  [OK] Risk Level: {verdict.risk_level}")
    print()

    print("=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    print(f"Agent ID: {verdict.agent_id}")
    print(f"Behavior: {verdict.behavior_id}")
    print(f"Verdict: {verdict.verdict}")
    print(f"Risk Level: {verdict.risk_level}")
    print(f"Posterior Probability: {verdict.posterior_probability:.2f}")
    print(f"Confidence: {verdict.confidence:.2f}")
    print()
    print("Recommendation:")
    print(f"  {verdict.recommendation}")
    print()


def print_role_mode_summary(report: SessionReport) -> None:
    """Role-aware mode: prints every hypothesis that ran, then the combined
    session verdict."""
    print(f"Role: {report.role_id}")
    print(f"Objective: {report.objective}")
    print()

    n_mandate = sum(1 for r in report.hypothesis_results if r.is_mandate)
    n_generic = len(report.hypothesis_results) - n_mandate
    print(f"[1/2] Ran {len(report.hypothesis_results)} hypotheses ({n_generic} generic + {n_mandate} mandate)...")
    print()
    header = f"  {'HYPOTHESIS':<48} {'FIRED':<6} {'POSTERIOR':<10} {'RISK':<9} {'MANDATE':<8} {'WEIGHT'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(report.hypothesis_results, key=lambda r: r.verdict.posterior_probability, reverse=True):
        print(
            f"  {r.id:<48} {('yes' if r.fired else 'no'):<6} "
            f"{r.verdict.posterior_probability:<10.2f} {r.verdict.risk_level:<9} "
            f"{('YES' if r.is_mandate else '-'):<8} {r.session_weight}"
        )
    print()

    print("[2/2] Combining into session verdict (weighted average of fired hypotheses)...")
    print()

    print("=" * 70)
    print("SESSION VERDICT SUMMARY")
    print("=" * 70)
    print(f"Agent ID: {report.agent_id}")
    print(f"Role: {report.role_id}")
    print(f"Overall Verdict: {report.overall_verdict}")
    print(f"Overall Risk Level: {report.overall_risk_level}")
    print(f"Overall Posterior Probability: {report.overall_posterior:.2f}")
    print(f"Threshold: {report.threshold}")

    fired = [r for r in report.hypothesis_results if r.fired]
    if fired:
        driving = max(fired, key=lambda r: r.session_weight * r.verdict.posterior_probability)
        tag = "mandate violation" if driving.is_mandate else "generic hypothesis"
        print(
            f"Driving hypothesis: {driving.id} ({tag}, posterior "
            f"{driving.verdict.posterior_probability:.2f}, weight {driving.session_weight})"
        )
    print()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("AGENTIC BEHAVIORAL DRIFT DETECTION")
    print("=" * 70)
    print()

    analyzer = SessionAnalyzer(args.hypothesis_config)

    if args.behavior:
        print("[1/4] Parsing logs and resolving hypothesis...")
        report = analyzer.analyze(
            log_file=args.log_file,
            agent_id=args.agent_id,
            behavior_id=args.behavior,
            threshold=args.threshold,
        )
        print()
        print_behavior_mode_summary(report)
    else:
        report = analyzer.analyze(
            log_file=args.log_file,
            agent_id=args.agent_id,
            role_id=args.role,
            threshold=args.threshold,
        )
        print_role_mode_summary(report)

    if args.output:
        save_json(report.to_dict(), args.output)
        print(f"Verdict saved to: {args.output}")


if __name__ == "__main__":
    main()
