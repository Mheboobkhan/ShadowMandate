# Writing rules

Every hypothesis is two files. The detection rule says **what to look for**;
the BN config says **how much it matters**. Keeping them separate is deliberate:
the first is a claim about your logs, the second is a claim about your risk
tolerance, and they get reviewed by different people for different reasons.

---

## `dr_*.json` — the detection rule

```json
{
  "behavior": "credential_access",
  "description": "Detects agents reading, referencing, or handling credential material.",
  "references": ["internal", "OWASP-LLM02"],
  "evidence_mapping": {
    "password_patterns": {
      "description": "Message references passwords or secrets",
      "patterns": ["password", "secret", "api_key", "apikey"],
      "search_fields": ["message"]
    },
    "credential_files": {
      "description": "Access to files that commonly store credentials",
      "patterns": [".pem", ".key", ".env", "id_rsa", "credentials.json"],
      "search_fields": ["file_path"],
      "jev": false
    }
  }
}
```

| Key | Required | Meaning |
|---|---|---|
| `behavior` | yes | Globally unique id. Must match the `behavior` in the paired `bn_*.json`. |
| `description` | no | What the hypothesis is for. Humans only. |
| `references` | no | Free-form provenance — OWASP ids, ticket numbers, internal docs. |
| `evidence_mapping` | yes | One entry per evidence node. Keys must exactly equal the `evidence_nodes` list in the BN config. |

### Evidence node: pattern matching

```json
"coercive_language": {
  "description": "User is pressuring or threatening the assistant into compliance",
  "patterns": ["you must comply", "have no choice", "i will sue"],
  "search_fields": ["message"]
}
```

Case-insensitive substring match. The node is 1 if **any** pattern is found in
**any** `search_fields` value on **any** event in the session; otherwise 0.
There is no count and no severity — one hit and five hundred hits are the same
number.

`search_fields` resolves in a fixed order: top-level event key →
`action_details` → `raw`. So `message`, `tool_names`, `file_path`,
`prompt_text`, and raw log keys all work without qualification. See
[log-formats.md](log-formats.md) for what's available.

> **Write patterns that can't fire by accident.** `"rotate"` matches "rotate
> the log file". `"secret"` matches "no secret sauce here". Prefer the longest
> phrase that still catches the real case — `"reset-password"` over
> `"reset"` — and remember that a pattern list cannot cover paraphrase. That
> limit is what the [Jev channel](jev-channel.md) exists to address.

### Evidence node: numeric and structural conditions

```json
"high_output_to_input_ratio": {
  "description": "Output tokens far exceed input tokens",
  "condition": {"field": "output_input_ratio", "operator": "gt", "value": 20}
}
```

```json
"region_mismatch": {
  "description": "Inference ran in a different region than the request",
  "condition": {"field": "region", "operator": "ne", "compare_field": "inference_region"}
}
```

| Operator | Meaning |
|---|---|
| `gt` `gte` `lt` `lte` | Numeric comparison |
| `eq` `ne` | Equality; falls back to string comparison for non-numeric values |

Use `value` to compare against a literal, or `compare_field` to compare two
fields on the same event. **If either side is missing, the condition does not
match** — this is why a rule about Bedrock token counts doesn't false-positive
on a plaintext log that has no token counts at all.

A node has either `patterns` or `condition`, not both.

### Evidence node: `jev`

Controls the optional [natural-language channel](jev-channel.md).

| Value | Effect |
|---|---|
| *(omitted)* | Pattern nodes opt in automatically, using `description` as the question. Condition nodes are always skipped. |
| `false` | Opt out. Use when an exact test is strictly better than a judgment — file extensions, literal syntax. |
| `{"instructions": "..."}` | Opt in with custom wording, when the question you want asked differs from the documentation string. |

---

## `bn_*.json` — the scoring config

```json
{
  "behavior": "credential_access",
  "evidence_nodes": ["password_patterns", "credential_files"],
  "hypothesis_node": "credential_access_drift",
  "cpd_parameters": {
    "base": 0.05,
    "password_patterns_weight": 0.40,
    "credential_files_weight": 0.50,
    "password_patterns_and_credential_files_boost": 0.20
  }
}
```

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `base` | 0–1 | 0.05 | Suspicion with no evidence at all |
| `<node>_weight` | 0–1 | — | What this node adds on its own |
| `<a>_and_<b>[_and_<c>]_boost` | 0–1 | — | Extra when all named nodes fire together |
| `all_*_boost` | 0–1 | — | Extra when *every* declared node fires. The middle word is a free-form label (`all_boost`, `all_three_boost`). |
| `saturation_knee` | 0–1 excl. | 0.6 | Where compression starts. See [scoring.md](scoring.md#accumulate-then-saturate). |
| `evidence_fire_threshold` | 0–1 | 0.5 | Evidence value at which a node counts as fired |

`hypothesis_node` is cosmetic unless you use the pgmpy backend; it defaults to
`<behavior>_drift`.

### Validation is not optional

Loading a `bn_*.json` validates it, and an invalid one raises rather than
scoring quietly:

```
ConfigValidationError: Invalid Bayesian Network config in bn_credential_access.json:
  - credential_access: weight key 'password_pattern_weight' does not match any
    declared evidence node (did you mean 'password_patterns_weight'?)
```

Caught at load time:

- a weight or boost naming a node that isn't declared, with a suggestion
- a declared node that no weight or boost mentions — it could never move the score
- a boost naming only one node
- a parameter the engine has no way to read
- a non-numeric value, or one outside its range

This exists because the failure it prevents is invisible. A mistyped weight key
used to be skipped in silence: the hypothesis scored at its base rate forever,
and nothing in the output said why.

---

## Adding a generic hypothesis

1. `mkdir agentic_detection/hypotheses/generic/<behavior_id>/`
2. Write `dr_<behavior_id>.json` and `bn_<behavior_id>.json`. The
   `evidence_mapping` keys and the `evidence_nodes` list must match exactly.
3. Register it in `agentic_detection/config/hypothesis.json` under
   `generic_behaviors`:

```json
{
  "id": "my_behavior",
  "name": "My Behavior Detection",
  "detection_rules": "hypotheses/generic/my_behavior/dr_my_behavior.json",
  "bn_config": "hypotheses/generic/my_behavior/bn_my_behavior.json",
  "output_csv": "hypotheses/generic/my_behavior/output.csv",
  "session_weight": 1.0
}
```

4. Test it in isolation against a log you know should trip it, and one you know
   shouldn't:

```bash
python detect_drift.py my_session.log --agent-id test --behavior my_behavior
```

5. Run `pytest`. `tests/test_hypothesis_config.py` lints the whole catalog —
   unique ids, files that exist, dr/bn node agreement, boosts referencing real
   nodes.

## Adding a role

A role encodes what one agent was *scoped* to do, so that behavior which looks
unremarkable generically becomes a violation in context.

1. Write `hypotheses/roles/<role_id>/role.json` — the human-readable mandate:

```json
{
  "id": "iam_investigator",
  "name": "IAM Investigation Agent",
  "objective": "Investigate IAM configuration... Read-only analysis only.",
  "in_scope": ["Listing/reading IAM users, roles, groups, and policies"],
  "out_of_scope": ["Resetting, rotating, or creating credentials"],
  "owner": "security-team"
}
```

`in_scope` / `out_of_scope` are **documentation — the engine never reads
them.** Enforcement is the evidence nodes in the mandate rule file. Keep one
evidence node per `out_of_scope` bullet, in the same order, and update both in
the same commit. The shipped `role.json` says exactly this in its `notes`
field; that convention is the only thing keeping the prose and the executable
rule from drifting apart.

2. Write `dr_<role_id>_mandate_violation.json` and its `bn_` pair, exactly as
   for a generic hypothesis.

3. Register the role:

```json
{
  "id": "iam_investigator",
  "name": "IAM Investigation Agent",
  "manifest": "hypotheses/roles/iam_investigator/role.json",
  "mandate_behaviors": [{
    "id": "iam_investigator_mandate_violation",
    "name": "IAM Investigator Mandate Violation",
    "detection_rules": "hypotheses/roles/iam_investigator/dr_iam_investigator_mandate_violation.json",
    "bn_config": "hypotheses/roles/iam_investigator/bn_iam_investigator_mandate_violation.json",
    "output_csv": "hypotheses/roles/iam_investigator/output.csv",
    "session_weight": 2.5
  }]
}
```

The `session_weight` above 1.0 is the point: it is how a mandate breach stays
in charge of the session verdict when generic rules fire alongside it.

---

## Tuning weights

Some discipline that will save you from tuning in circles:

**Change one number per commit, with the reason in the message.** These numbers
are the product. A diff that moves six weights at once is unreviewable.

**Keep single-node scores under the knee.** A node whose weight alone pushes
`base + weight` past 0.6 is claiming that one substring match is near-conclusive.
It rarely is.

**Reserve the top of the range for combinations.** Boosts are how you say "these
two together mean something the parts don't" — that is where high scores should
come from.

**Pin the numbers in a test.** `tests/test_golden_scenarios.py` replays real
sample logs and asserts exact posteriors. If you change a weight and no test
fails, the weight wasn't doing anything.

**Re-read `.explain()` after tuning.** It shows every term that fired. If a term
you expected is absent, you have a node/key mismatch, not a tuning problem.
