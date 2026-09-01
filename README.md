# ShadowMandate

**Your AI agents are shadow users.** They hold IAM privilege, call tools,
touch credentials, and route across regions - with none of the accountability
a human identity would carry. ShadowMandate is a Bayesian-network drift
detector DevSecOps teams write, own, and run against their own agent logs: it
scores how far an agent's observed behavior has drifted from (a) known-bad
generic patterns and (b) *that specific agent's declared mandate* - and it
takes your detection rules as input, not a vendor's fixed model.

The motivating case: an IAM-investigation agent resetting or harvesting
credentials isn't inherently unusual-looking on its own - a generic rule set
stays quiet on it. It's only a violation because it's outside *that agent's*
declared objective. A shadow user with root and no accountability doing
exactly that, unnoticed, is the failure mode this project targets.

## Related work

This isn't the first project in the space, and it's worth knowing the
neighbors before you evaluate it:

- **[dedrift](https://dedrift.ai/)** - open-source silent behavioral drift
  detection for AI agents via a canary-suite + statistical-signature approach.
- **[Agent Threat Rules (ATR)](https://github.com/Agent-Threat-Rule/agent-threat-rules)**,
  **AgentSigma**, **[agentshield-ai/sigma-ai](https://github.com/agentshield-ai/sigma-ai)** -
  Sigma-style, YAML-based detection-rule standards for LLM/agent tool-call
  activity, each already positioned as "Sigma for AI agents."
- **Armo** has written about per-agent behavioral baselines ("defining
  normal") as a drift signal, conceptually close to the mandate idea here.

What's different here: rules are scored through a **Bayesian network** (a
human-tunable CPD - base rate + per-evidence weights + interaction boosts,
auditable via `BayesianNetworkEngine.explain()`) rather than pure statistical
or embedding-based drift, and generic + role-mandate rules share **one
identical schema** (`dr_*.json` detection rules + `bn_*.json` BN config)
instead of a separate DSL for each tier. It's a narrower, more specific tool
than the projects above, not a replacement for any of them.

## How it works

```
agent log (key=value or AWS Bedrock ModelInvocationLog ndjson)
    -> AgentLogParser        parse into normalized events
    -> RuleEngine             match events against a hypothesis's evidence_mapping -> 0/1 evidence vector
    -> BayesianNetworkEngine  evidence vector -> P(drift | evidence)  (pgmpy if installed, else an
                               equivalent dependency-free closed-form formula)
    -> VerdictGenerator       posterior -> risk band + recommendation
    -> SessionAnalyzer        runs every applicable hypothesis (generic catalog + a role's
                               mandate rules) in one pass and combines fired hypotheses into
                               one session verdict (weighted average, mandate rules weighted
                               higher via a config-driven session_weight)
```

Every rule is a JSON pair your security team writes, reads, and diffs in code
review - not a trained model, not a vendor black box. No hard dependencies are
required: `pgmpy` is optional (see `requirements.txt`); without it,
`bn_engine.py` falls back to a manual formula that is mathematically
equivalent.

## Quickstart

```bash
git clone <this-repo>
cd ShadowMandate

# Role-aware scan: runs all generic hypotheses + the iam_investigator mandate
# rules against a sample session containing both legitimate read-only IAM
# investigation calls and two mandate-violating ones (password reset,
# bulk credential-report download).
python3 detect_drift.py data/raw/iam_investigator_session.ndjson \
    --agent-id iam-agent-01 --role iam_investigator

# Single-hypothesis mode: test one rule in isolation
python3 detect_drift.py data/raw/app.log \
    --agent-id ollama-test --behavior external_connection
```

The role-aware run above lands on `DRIFT_DETECTED / HIGH`, driven by the
`iam_investigator_mandate_violation` hypothesis (posterior 1.00) - while every
individual *generic* hypothesis in the same run stays at MEDIUM or below.
That gap is the point: this is the case a generic-only ruleset misses, and a
role-mandate one catches.


## Writing a new generic hypothesis

Add a folder under `hypotheses/generic/<id>/` with a `dr_<id>.json` (evidence
nodes - either substring `patterns` over named `search_fields`, or a numeric/
field-compare `condition`) and a `bn_<id>.json` (base rate + per-node weights +
interaction boosts), then register it in `config/hypothesis.json` under
`generic_behaviors`.

Example: a `bulk_data_export` hypothesis with one pattern-matched node and one
numeric-condition node.

`hypotheses/generic/bulk_data_export/dr_bulk_data_export.json`:

```json
{
  "behavior": "bulk_data_export",
  "description": "Detects an agent exporting data in bulk rather than reading it in place.",
  "references": ["internal"],
  "evidence_mapping": {
    "export_command": {
      "description": "Agent invoked a bulk export/dump command",
      "patterns": ["export-table", "dump database", "bulk-export", "--output csv"],
      "search_fields": ["message", "tool_names"]
    },
    "large_row_count": {
      "description": "Result set size is far beyond a normal interactive query",
      "condition": {"field": "row_count", "operator": "gt", "value": 100000}
    }
  }
}
```

`hypotheses/generic/bulk_data_export/bn_bulk_data_export.json`:

```json
{
  "behavior": "bulk_data_export",
  "evidence_nodes": ["export_command", "large_row_count"],
  "hypothesis_node": "bulk_data_export_drift",
  "cpd_parameters": {
    "base": 0.05,
    "export_command_weight": 0.35,
    "large_row_count_weight": 0.30,
    "export_command_and_large_row_count_boost": 0.20
  }
}
```

Interaction-boost keys must be named `<node1>_and_<node2>_boost`, joining the
exact `evidence_nodes` names the boost requires - that's how `bn_engine.py`
knows which nodes to check together.

Register it in `config/hypothesis.json`:

```json
{
  "id": "bulk_data_export",
  "name": "Bulk Data Export Detection",
  "detection_rules": "hypotheses/generic/bulk_data_export/dr_bulk_data_export.json",
  "bn_config": "hypotheses/generic/bulk_data_export/bn_bulk_data_export.json",
  "output_csv": "hypotheses/generic/bulk_data_export/output.csv",
  "session_weight": 1.0
}
```

## Writing a new role (defining a shadow user's mandate)

Add `hypotheses/roles/<role_id>/role.json` describing the agent's objective
and in/out-of-scope actions - i.e. what this specific IAM principal is
actually accountable for doing - plus one or more `dr_*.json`/`bn_*.json`
mandate hypotheses in the same folder, then register the role and its
`mandate_behaviors` in `config/hypothesis.json`. See `roles/iam_investigator/`
for a complete example.

Example: a `backup_operator` role whose mandate is "take/verify backups,
never restore or delete one" - one mandate hypothesis per out-of-scope
bullet, in the same order, so the two stay traceable to each other.

`hypotheses/roles/backup_operator/role.json`:

```json
{
  "id": "backup_operator",
  "name": "Backup Operator Agent",
  "objective": "Create and verify scheduled backups. Never restore, delete, or move backup data.",
  "in_scope": [
    "Triggering scheduled backup jobs",
    "Verifying backup integrity/checksums"
  ],
  "out_of_scope": [
    "Restoring or overwriting live data from a backup",
    "Deleting backup snapshots or archives"
  ],
  "owner": "security-team",
  "notes": "This list is descriptive only - it is not read by the engine. Enforcement is the evidence_mapping in dr_backup_operator_mandate_violation.json (one evidence node per out_of_scope bullet, in the same order); update both together."
}
```

`hypotheses/roles/backup_operator/dr_backup_operator_mandate_violation.json`:

```json
{
  "behavior": "backup_operator_mandate_violation",
  "description": "Detects a backup agent restoring/overwriting data or deleting backups - out of scope for a create-and-verify-only role.",
  "references": ["internal"],
  "note": "Each evidence node below is the executable enforcement of the corresponding bullet in role.json's out_of_scope list (same order). Keep both in sync when either changes.",
  "evidence_mapping": {
    "restore_or_overwrite": {
      "description": "Agent restored a backup onto live data",
      "patterns": ["restore-backup", "overwrite live", "restore-database"],
      "search_fields": ["message", "tool_names"]
    },
    "backup_deletion": {
      "description": "Agent deleted a backup snapshot or archive",
      "patterns": ["delete-snapshot", "rm -rf /backups", "delete-backup"],
      "search_fields": ["message", "tool_names"]
    }
  }
}
```

`hypotheses/roles/backup_operator/bn_backup_operator_mandate_violation.json`:

```json
{
  "behavior": "backup_operator_mandate_violation",
  "evidence_nodes": ["restore_or_overwrite", "backup_deletion"],
  "hypothesis_node": "backup_operator_mandate_violation_drift",
  "cpd_parameters": {
    "base": 0.05,
    "restore_or_overwrite_weight": 0.45,
    "backup_deletion_weight": 0.45,
    "restore_or_overwrite_and_backup_deletion_boost": 0.10
  }
}
```

Register the role in `config/hypothesis.json` under `roles`:

```json
{
  "id": "backup_operator",
  "name": "Backup Operator Agent",
  "manifest": "hypotheses/roles/backup_operator/role.json",
  "mandate_behaviors": [
    {
      "id": "backup_operator_mandate_violation",
      "name": "Backup Operator Mandate Violation",
      "detection_rules": "hypotheses/roles/backup_operator/dr_backup_operator_mandate_violation.json",
      "bn_config": "hypotheses/roles/backup_operator/bn_backup_operator_mandate_violation.json",
      "output_csv": "hypotheses/roles/backup_operator/output.csv",
      "session_weight": 2.5
    }
  ]
}
```

Then run it the same way as the built-in role:

```bash
python3 detect_drift.py <log_file> --agent-id backup-agent-01 --role backup_operator
```

## Data

Sample logs under `data/raw/` are drawn from Splunk's publicly available
attack-data dataset, used here for demonstration purposes only.

## License

MIT - see [LICENSE](LICENSE).