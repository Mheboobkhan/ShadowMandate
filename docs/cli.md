# CLI reference

```
python detect_drift.py <log_file> --agent-id <id> (--role <role> | --behavior <id>) [options]
```

## Modes

**Role-aware session scan** — the primary workflow. Runs the full generic
catalog plus that role's mandate rules in one pass over the log, and combines
them into one session verdict.

```bash
python detect_drift.py data/raw/iam_investigator_session.ndjson \
    --agent-id iam-agent-01 --role iam_investigator
```

**Single hypothesis** — for testing one rule in isolation while you write it.
No session combination happens; you get that one hypothesis's verdict.

```bash
python detect_drift.py data/raw/app.log \
    --agent-id test --behavior external_connection
```

Exactly one of `--role` / `--behavior` is required.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `log_file` | — | Path to the log to analyze (positional) |
| `--agent-id` | `agent-001` | Identifier recorded in the verdict |
| `--role` | — | Role id. Generic catalog + that role's mandate rules. |
| `--behavior` | — | Single hypothesis id, from either list |
| `--threshold` | `0.5` | `DRIFT_DETECTED` at or above this posterior |
| `--output` | — | Write the full report as JSON |
| `--hypothesis-config` | `agentic_detection/config/hypothesis.json` | Alternate catalog |
| `--jev` / `--no-jev` | off | Enable/force-off the [Jev channel](jev-channel.md) |
| `--jev-model` | `jev-latest` | Pin a Jev model version |

An unknown role or behavior exits with the list of valid ids.

## Environment

| Variable | Meaning |
|---|---|
| `SHADOWMANDATE_JEV` | `1`/`true`/`yes`/`on` enables the Jev channel |
| `SHADOWMANDATE_JEV_MODEL` | Default Jev model version |
| `TYPESAFE_API_KEY` | Jev credentials |

## Reading the output

```
  HYPOTHESIS                                       FIRED  POSTERIOR  RISK      MANDATE  WEIGHT
  ---------------------------------------------------------------------------------------------
  iam_investigator_mandate_violation               yes    0.930      CRITICAL  YES      2.5
  high_risk_filesystem_and_exec_tool_invocation    yes    0.500      MEDIUM    -        1.0
  credential_access                                yes    0.450      LOW       -        1.0
  ...

Overall Verdict: DRIFT_DETECTED
Overall Risk Level: CRITICAL
Overall Posterior Probability: 0.972
Threshold: 0.5
Combination: noisy_or
Driving hypothesis: iam_investigator_mandate_violation (mandate violation, posterior 0.930, weight 2.5)
```

- **FIRED** — at least one evidence node reached its fire threshold. Unfired
  hypotheses sit at their base rate and are excluded from the combination.
- **POSTERIOR** — that hypothesis alone. Never exactly 1.0; see
  [scoring.md](scoring.md#why-not-just-clamp-at-10).
- **Driving hypothesis** — the largest `session_weight × posterior`. Start here.
- **Combination** — which strategy produced the overall score.

The overall posterior is never below the strongest fired hypothesis. If it
looks like it is, you are running `weighted_average`.

## Output JSON

`--output` writes:

```json
{
  "agent_id": "iam-agent-01",
  "role_id": "iam_investigator",
  "objective": "Investigate IAM configuration... Read-only analysis only.",
  "overall_posterior_probability": 0.9724,
  "overall_verdict": "DRIFT_DETECTED",
  "overall_risk_level": "CRITICAL",
  "threshold": 0.5,
  "combination": "noisy_or",
  "evidence_channels": {
    "patterns": {"status": "ok"},
    "jev": {
      "status": "disabled",
      "detail": "Jev channel not enabled (use --jev or SHADOWMANDATE_JEV=1)",
      "questions_asked": 0,
      "state_chars": 0,
      "state_truncated": false,
      "model": null
    }
  },
  "hypothesis_results": [
    {
      "agent_id": "iam-agent-01",
      "behavior": "iam_investigator_mandate_violation",
      "posterior_probability": 0.9305,
      "verdict": "DRIFT_DETECTED",
      "risk_level": "CRITICAL",
      "confidence": 0.861,
      "threshold": 0.5,
      "recommendation": "Immediate action required...",
      "evidence": {
        "credential_reset_or_rotation": 1,
        "credential_harvesting": 1,
        "privilege_escalation_action": 0
      },
      "hypothesis_id": "iam_investigator_mandate_violation",
      "hypothesis_name": "IAM Investigator Mandate Violation",
      "mandate_violation": true,
      "session_weight": 2.5,
      "fired": true,
      "generated_at": "2026-09-20T11:07:18.952732+00:00"
    }
  ],
  "generated_at": "2026-09-20T11:07:18.952742+00:00"
}
```

`evidence` values are `0`/`1` from the pattern channel and a rounded
probability when the Jev channel raised a node. With Jev enabled, each
hypothesis also carries:

```json
"evidence_channels": {
  "patterns": {"override": 0},
  "jev": {"override": 0.9},
  "jev_only_nodes": ["override"]
}
```

## Using it in a pipeline

The exit code is 0 on a successful scan regardless of verdict — a detected
drift is a successful run, not a failure. Branch on the JSON:

```bash
python detect_drift.py "$LOG" --agent-id "$AGENT" --role "$ROLE" --output verdict.json

python - <<'PY'
import json, sys
report = json.load(open("verdict.json"))

jev = report["evidence_channels"]["jev"]
if jev["status"] in ("unavailable", "error"):
    print(f"::warning::Jev channel degraded: {jev['detail']}", file=sys.stderr)

if report["overall_risk_level"] in ("HIGH", "CRITICAL"):
    sys.exit(1)
PY
```

Alerting on a degraded channel matters as much as alerting on the verdict: a
clean result from a half-working pipeline is the one failure mode that looks
like success.

## Library use

```python
from agentic_detection import SessionAnalyzer, JevChannel

analyzer = SessionAnalyzer(
    "agentic_detection/config/hypothesis.json",
    jev_channel=JevChannel.from_env(),
)
report = analyzer.analyze(
    log_file="session.ndjson",
    agent_id="iam-agent-01",
    role_id="iam_investigator",
    threshold=0.5,
)

print(report.overall_posterior, report.overall_risk_level)
for result in report.hypothesis_results:
    if result.fired:
        print(result.id, result.verdict.posterior_probability, result.jev_only_nodes())
```

To see the arithmetic behind a single number:

```python
from agentic_detection import BayesianNetworkEngine

bn = BayesianNetworkEngine("agentic_detection/hypotheses/generic/credential_access/bn_credential_access.json")
print(bn.explain({"password_patterns": 1, "credential_files": 1}))
```
