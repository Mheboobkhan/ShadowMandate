"""
Agentic Behavioral Drift Detection Engine
==========================================

A behavioral drift detection system for LLM agents that uses Bayesian
Networks to compute posterior probabilities of suspicious behavior based
on a security team's curated detection rules.

Pipeline:
    Agent Logs -> Log Parser -> Rule Matcher -> Evidence Detector
               -> BN Engine -> Verdict Generator
"""

from .log_parser import AgentLogParser
from .rule_engine import RuleEngine
from .evidence_detector import EvidenceDetector
from .bn_engine import BayesianNetworkEngine
from .verdict_generator import VerdictGenerator, Verdict
from .session_analyzer import SessionAnalyzer, SessionReport, HypothesisSpec, HypothesisResult

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
]

__version__ = "1.0.0"
