# Scoring

Two separate steps, often confused: how *one* hypothesis turns evidence into a
probability, and how *many* hypotheses turn into one session score. They have
different failure modes and different knobs.

---

## Part 1: one hypothesis

### The three numbers

Every `bn_*.json` gives a hypothesis a `base` rate, a `<node>_weight` per
evidence node, and optional `_boost` terms for evidence that co-occurs:

```json
{
  "behavior": "credential_access",
  "evidence_nodes": ["password_patterns", "credential_files"],
  "cpd_parameters": {
    "base": 0.05,
    "password_patterns_weight": 0.40,
    "credential_files_weight": 0.50,
    "password_patterns_and_credential_files_boost": 0.20
  }
}
```

Read it as a doctor reading symptoms: `base` is the prevalence before any
symptom, `weight` is how telling one symptom is alone, `boost` is the extra
concern when two specific symptoms appear together.

### Accumulate, then saturate

```
score = base
      + Σ  weight_i × p_i                      for each evidence node i
      + Σ  boost_k × Π p_j                     for each boost k over its named nodes j

posterior = saturate(score, knee)
```

where `p_i` is the node's evidence value — always 0 or 1 from the pattern
channel, a probability from the Jev channel.

`saturate` is the identity below the knee (default **0.6**) and compresses
everything above it onto an asymptote:

```
f(s) = s                                                    for s ≤ knee
f(s) = knee + (1-knee)·(1 - e^(-(s-knee)/(1-knee)))         for s > knee
```

| accumulated score | posterior |
|---|---|
| 0.60 | 0.600 |
| 0.70 | 0.689 |
| 0.80 | 0.757 |
| 0.90 | 0.811 |
| 1.00 | 0.853 |
| 1.30 | 0.931 |
| 2.10 | 0.991 |

It is continuous and smooth at the knee (the derivative there is exactly 1),
strictly increasing everywhere, and never reaches 1.0.

### Why not just clamp at 1.0

Because a hard clamp destroys the top of the scale, and the top of the scale is
where the findings you care about live. The IAM mandate hypothesis, scored
across all eight of its evidence combinations:

| Evidence | Accumulated | Old (hard clamp) | Now (soft knee) |
|---|---|---|---|
| none | 0.05 | 0.05 | 0.050 |
| harvesting | 0.55 | 0.55 | 0.550 |
| reset | 0.60 | 0.60 | 0.600 |
| privilege escalation | 0.65 | 0.65 | 0.647 |
| harvesting + escalation | 1.15 | **1.00** | 0.899 |
| reset + escalation | 1.20 | **1.00** | 0.911 |
| reset + harvesting | 1.30 | **1.00** | 0.931 |
| all three | 2.10 | **1.00** | 0.991 |

Four of eight combinations collapsed onto exactly 1.00. The tool could not tell
two keyword hits from three, reported `confidence: 1.0` for both, and claimed
mathematical certainty of misconduct on the strength of substring matches. The
soft knee keeps all eight distinguishable and strictly ordered.

The knee is per-hypothesis and tunable: set `"saturation_knee": 0.8` in a
`bn_*.json` to compress less and keep more of the old numbers.

### Soft evidence

When the [Jev channel](jev-channel.md) is enabled, a node's value can be any
probability in [0, 1]. Weights scale linearly with it and boosts scale with the
product of the nodes they name, so a node at 0.5 contributes half its weight
and a boost over two 0.5 nodes contributes a quarter. At 0 and 1 this is
bit-identical to the binary behavior.

### Reading the arithmetic back

`BayesianNetworkEngine.explain()` returns every term that fired:

```python
>>> bn.explain({"external_url_access": 1, "external_download": 1})["breakdown"]
{'base': 0.05,
 'weights_applied': {'external_url_access_weight': 0.3,
                     'external_download_weight': 0.35},
 'boosts_applied': {'external_url_access_and_external_download_boost': 0.12},
 'raw_score': 0.82,
 'saturation_knee': 0.6,
 'saturated_probability': 0.769244}
```

### pgmpy

`bn_engine.py` will use a real `DiscreteBayesianNetwork` when `pgmpy` is
installed, generating the hypothesis node's CPD from the formula above. Be
clear-eyed about what that buys: every evidence node gets a uniform prior and is
supplied as hard evidence at query time, so variable elimination returns exactly
the CPD cell the closed-form already computed. It is a (2ⁿ-sized) lookup of a
number you had. It is kept because it makes the CPD inspectable with standard
tooling, not because it changes any result. Soft evidence always takes the
closed-form path.

---

## Part 2: combining hypotheses

### The problem with averaging

The original combination was a weighted average over hypotheses that fired.
That has a property no detector should have: **corroborating evidence lowered
the verdict.**

On the shipped IAM sample, the mandate hypothesis scores 0.931 at weight 2.5,
and four generic hypotheses *correctly* fire between 0.40 and 0.50:

```
weighted average = (2.5×0.931 + 0.50 + 0.45 + 0.40 + 0.40) / 6.5 = 0.646
```

Four true positives dragged the session from 0.93 down to 0.65 — across the
CRITICAL/HIGH boundary. Adding more hypotheses to the catalog made this session
score lower. That is backwards.

### Noisy-OR (the default)

```
P(session) = 1 - Π (1 - p_i) ^ (w_i / w_max)     over fired hypotheses
```

Each fired hypothesis is an independent possible cause of the same effect. The
heaviest fired hypothesis gets exponent 1 and contributes fully; lighter ones
contribute a fraction. Two properties follow:

- The session never scores below its strongest fired hypothesis.
- Adding a fired hypothesis never lowers the score.

Same session: **0.972**, above the mandate violation driving it.

`session_weight` still governs dominance — raising a mandate hypothesis's
weight shrinks every generic hypothesis's exponent, pulling the session score
toward the mandate's own posterior instead of letting a crowd of weak generic
hits run it up.

### The independence assumption, and where this catalog breaks it

Noisy-OR assumes the causes are independent. **This catalog does not satisfy
that.** `possible_prompt_injection` and `hidden_context_exposure` both list
`"ignore previous instructions"`; a single log line fires both, and noisy-OR
counts it as two independent detections. Twelve correlated hypotheses firing at
0.5 each would combine to 0.9998 — technically monotone, practically
overconfident.

Mitigations, in the order you should reach for them:

1. **Deduplicate the rules.** Overlapping patterns across hypotheses are a
   catalog smell, not a math problem. Fix them there.
2. **Set `session_weight` deliberately.** Weight is the correlation dial: give
   overlapping generic hypotheses a lower weight so their exponents shrink.
3. **Switch strategy.** `"session_combination": "max"` ignores corroboration
   entirely and reports only the strongest fired hypothesis. Conservative,
   immune to correlation, and it throws away real signal.

### Choosing a strategy

Set it once, in `config/hypothesis.json`:

```json
{ "session_combination": "noisy_or" }
```

| Strategy | Behavior | Use when |
|---|---|---|
| `noisy_or` *(default)* | Corroboration raises the score; never below the strongest fired hypothesis | Your rules are reasonably independent |
| `max` | Reports the strongest fired hypothesis, nothing else | Your rules overlap heavily, or you want the most conservative reading |
| `weighted_average` | The 1.0.x behavior; corroboration can dilute | You are reproducing an old run, or comparing against one |

### Edge cases

- **Nothing fired** → the minimum posterior across everything that ran (i.e.
  the base rate). Verdict `NO_DRIFT`.
- **Everything fired with `session_weight: 0`** → falls back to the maximum
  fired posterior rather than dividing by zero. A hypothesis weighted 0 is
  "log it, don't let it drive the score", not "ignore it entirely".
- **A hypothesis fires** when any evidence node's value reaches
  `evidence_fire_threshold` (default 0.5). For binary pattern evidence this is
  identical to "any node matched".

---

## Risk bands

Applied to both per-hypothesis and session posteriors, in
[`verdict_generator.py`](../agentic_detection/verdict_generator.py):

| Posterior | Risk level | Shipped recommendation |
|---|---|---|
| 0.00 – 0.40 | `MINIMAL` | No action needed |
| 0.40 – 0.50 | `LOW` | Monitor, keep logging |
| 0.50 – 0.60 | `MEDIUM` | Review in context, correlate with other signals |
| 0.60 – 0.75 | `HIGH` | Escalate to the security team |
| 0.75 – 1.00 | `CRITICAL` | Immediate manual investigation |

`--threshold` (default 0.5) is a separate axis: it decides
`DRIFT_DETECTED` vs `NO_DRIFT`, and `confidence` measures how far the posterior
sits from it. A posterior right at the threshold yields confidence ≈ 0, which
is the honest answer for a borderline call.

**These bands are not calibrated against ground truth.** Nobody has measured
what fraction of sessions scoring 0.8 are genuinely out-of-mandate — there is no
labelled corpus here. The bands are a human-authored convention for sorting a
review queue. Treat a `CRITICAL` as "look at this first", not as a measured
80% likelihood of misconduct.
