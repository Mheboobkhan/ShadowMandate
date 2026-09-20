# The Jev evidence channel

**Status: optional, off by default.** Nothing in this page happens unless you
pass `--jev` or set `SHADOWMANDATE_JEV=1`.

---

## What problem it solves

The pattern channel asks: *does this log line contain one of these substrings?*
That is exact, offline, free, and auditable — and blind to paraphrase.

```
"ignore previous instructions"              → matches, node fires
"disregard everything you were told before" → matches nothing, scores 0.05
```

Both are prompt injection. A phrase list cannot enumerate a paraphrase space,
and the hypotheses where this hurts most are precisely the semantic ones:
`possible_prompt_injection`, `hostile_prompt_sentiment`,
`hidden_context_exposure`, `sensitive_data_in_prompts`, and mandate violations
phrased in prose rather than CLI syntax.

## What Jev is

[TypeSafe AI's](https://typesafe.ai) "System One" model: unstructured state in,
typed calibrated probabilities out. It does not generate text. A `Noul`
question returns a probability in [0, 1] — exactly the shape the evidence layer
wants — and every question in a request is answered in one parallel pass, so
the whole catalog costs roughly one round trip.

| | |
|---|---|
| Latency | 70–500 ms per call |
| Context | 32k tokens |
| Pricing | $0.042 / M input tokens; output free |
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` (hosted only; no on-prem option is documented) |

## Read this before enabling it

**Data leaves your machine.** The channel sends selected log content to
`api.typesafe.ai`. For this tool that content includes raw prompts, IAM ARNs,
tool arguments, file paths, and credential-adjacent text — the material a
security tool auditing privileged agents is specifically pointed at. This is
the reason the default is off, and it is an operator decision, not an
implementation detail. If your agent logs cannot leave your boundary, do not
enable this channel; the pattern channel is the supported offline configuration,
not a degraded one.

**It changes what the README claims.** The project's thesis is that every score
traces back to a decision a person made rather than a model that learned one.
With this channel on, the *observation* step is a closed-weights hosted model's
judgment. The defense is real but narrower than the original claim: a human
still authors the question, the criteria, the weights, the boosts, the
combination strategy, and the threshold, and still makes every decision about
what to do. What Jev replaces is the keyword list — which was itself only a
brittle proxy for the sentence the human actually wrote. Decide whether you
find that persuasive before turning it on, not after.

**Pin the model version.** `jev-latest` means your effective detection
thresholds move when TypeSafe ships a revision — the exact silent drift this
project promises not to have. Use `--jev-model jev-1.13.0` or
`SHADOWMANDATE_JEV_MODEL`, and treat a version bump as a change that needs the
golden scenarios re-run.

---

## Enabling it

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY="sk-..."

python detect_drift.py session.ndjson \
    --agent-id iam-agent-01 --role iam_investigator \
    --jev --jev-model jev-1.13.0
```

| Control | Effect |
|---|---|
| `--jev` | Enable for this run |
| `--no-jev` | Force off, overriding the environment |
| `SHADOWMANDATE_JEV=1` | Enable by default in this shell |
| `--jev-model` / `SHADOWMANDATE_JEV_MODEL` | Pin a model version |
| `TYPESAFE_API_KEY` | Credentials |

## How it plugs in

Each pattern evidence node already carries a `description` written as the
judgment a human wants made:

```json
"credential_reset_or_rotation": {
  "description": "Agent reset, rotated, or created a credential instead of just reading about it",
  "patterns": ["reset-password", "rotate", "create-access-key"]
}
```

That description *is* the question. It becomes
`Noul(instructions="Agent reset, rotated, or created a credential…")`, so the
whole catalog works with no rule-file edits. Per-node control is documented in
[writing-rules.md](writing-rules.md#evidence-node-jev).

Nodes are **skipped** when:

- they use a numeric `condition` — token counts, region mismatches, and prompt
  sizes are exact comparisons on structured fields, and a probabilistic model
  can only make them worse;
- they set `"jev": false` — used in the shipped catalog for `credential_files`
  (a `.pem` / `id_rsa` extension test) and `dangerous_sink_invocation` (literal
  `eval(` / `os.system(` syntax), where an exact match is simply correct.

Roughly a third of the shipped catalog is better off without this channel. That
is the intended outcome, not a gap.

### Merging the two channels

```python
evidence[node] = max(pattern_value, jev_probability)
```

Deliberately asymmetric. A pattern hit is exact and is trusted outright; Jev can
raise a node the patterns missed but can never talk one down. Enabling the
channel is therefore monotonic — nothing that fires today stops firing, so you
can turn it on without re-validating your existing detections.

The resulting probability flows through the normal scoring path as soft
evidence: a node at 0.9 contributes 90% of its weight. See
[scoring.md](scoring.md#soft-evidence).

### Batching and context

All questions for a session go in one request (chunked at 16, purely as a guard
against a pathological catalog). The 32k context is a real constraint for
Bedrock ndjson sessions, so events are compacted field-by-field and, if still
oversized, dropped **from the middle** — keeping the opening, where the agent's
objective usually sits, and the tail, where its final actions are. The count of
dropped events is included in the payload so the model is not silently reasoning
over a gap, and `state_truncated` is recorded in the report.

---

## When it fails

A security scan must not die because an integration is down, and must not
report a clean verdict from a half-working pipeline. Every failure degrades to
patterns-only and is recorded:

| `status` | Cause |
|---|---|
| `disabled` | Not enabled. Nothing was sent. |
| `ok` | Answers returned |
| `no_questions` | Enabled, but no node opted in |
| `unavailable` | SDK not installed, or no API key |
| `error` | The call failed — timeout, 429, 5xx, malformed response |

```
Evidence channels: patterns + jev  [!!] jev: typesafe-sdk is not installed
  WARNING: scored on pattern matching alone - paraphrases were not checked
```

and in the JSON:

```json
"evidence_channels": {
  "patterns": {"status": "ok"},
  "jev": {"status": "error", "detail": "TimeoutError: ...", "model": "jev-1.13.0"}
}
```

If you run this in CI or a pipeline, **alert on `jev.degraded`**. A silent
fallback is the one way this channel can hurt you.

## Reading the results

Findings the model caught that the patterns missed are called out explicitly:

```
Raised by the Jev channel alone (no pattern matched):
  - possible_prompt_injection: instruction_override_phrase
```

and per-hypothesis in the JSON under `evidence_channels.jev_only_nodes`. These
are the rows to review first — both because they are the channel's whole value
and because they are where an uncalibrated question will show up as a false
positive.

## Cost

31 questions across the shipped catalog, one batched call per session. At
$0.042/M input tokens with output free, a session of a few thousand tokens is a
fraction of a cent. The cost that matters is not per-session — it is that
per-*event* questioning, if you ever restructure toward it, multiplies by N and
stops being negligible.

## Evaluating it before you trust it

Both channels run side by side, so measure rather than assume:

1. Run your golden scenarios with `--no-jev` and with `--jev`, saving both
   `--output` files.
2. Diff the per-node evidence. Every difference is a node Jev raised.
3. Read each one. A paraphrase the patterns missed is the win; anything else is
   a question that needs rewording in the rule file.
4. Only then consider lowering a `session_weight` or widening a threshold on the
   strength of the new coverage.
