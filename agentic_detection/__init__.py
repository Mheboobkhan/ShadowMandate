"""
Agentic Behavioral Drift Detection Engine
==========================================

A behavioral drift detection system for LLM agents that uses Bayesian
Networks to compute posterior probabilities of suspicious behavior based
on a security team's curated detection rules.

Pipeline:
    Agent Logs -> Log Parser -> Rule Matcher -+
                                              |-> Evidence Detector
                  (optional) Jev Channel -----+
               -> BN Engine -> Verdict Generator -> Session Combination
"""

from .log_parser import AgentLogParser
from .rule_engine import RuleEngine
from .evidence_detector import EvidenceDetector
from .bn_engine import BayesianNetworkEngine
from .verdict_generator import VerdictGenerator, Verdict
from .session_analyzer import (
    SessionAnalyzer,
    SessionReport,
    HypothesisSpec,
    HypothesisResult,
    combine_posteriors,
    COMBINATION_STRATEGIES,
)
from .jev_channel import JevChannel, JevOutcome, JevQuestion
from .config_validation import ConfigValidationError, validate_bn_config

__all__ = [
    "AgentLogParser",
    "RuleEngine",
    "EvidenceDetector",
    "BayesianNetworkEngine",
    "VerdictGenerator",
    "Verdict",
    "SessionAnalyzer",
    "SessionReport",
    "HypothesisSpec",
    "HypothesisResult",
    "combine_posteriors",
    "COMBINATION_STRATEGIES",
    "JevChannel",
    "JevOutcome",
    "JevQuestion",
    "ConfigValidationError",
    "validate_bn_config",
]

__version__ = "1.1.0"
