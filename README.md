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

## Writing a new role (defining a shadow user's mandate)

Add `hypotheses/roles/<role_id>/role.json` describing the agent's objective
and in/out-of-scope actions - i.e. what this specific IAM principal is
actually accountable for doing - plus one or more `dr_*.json`/`bn_*.json`
mandate hypotheses in the same folder, then register the role and its
`mandate_behaviors` in `config/hypothesis.json`. See `roles/iam_investigator/`
for a complete example.

## Data

Sample logs under `data/raw/` are drawn from Splunk's publicly available
attack-data dataset, used here for demonstration purposes only.

## License

MIT - see [LICENSE](LICENSE).