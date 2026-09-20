# Upgrading from 1.0.x

Three behavior changes and one new optional feature. Nothing requires a config
edit, but **your pinned numbers will move** and two of the changes can shift a
session across a risk band.

---

## 1. Posteriors above 0.6 are lower, and none of them is 1.0

Scores at or below the `saturation_knee` (default 0.6) are **bit-identical**.
Above it, a soft knee replaces the hard clamp.

| Accumulated | Was | Now |
|---|---|---|
| ≤ 0.60 | unchanged | unchanged |
| 0.82 | 0.82 | 0.769 |
| 1.15 | 1.00 | 0.899 |
| 1.30 | 1.00 | 0.931 |
| 2.10 | 1.00 | 0.991 |

**Why:** the clamp collapsed every high-evidence combination onto exactly 1.00.
Four of the IAM mandate hypothesis's eight combinations were indistinguishable,
all reporting certainty. See
[scoring.md](scoring.md#why-not-just-clamp-at-10).

**What to check:** anything that previously landed between 0.75 and 0.82
accumulated was `CRITICAL` and may now be `HIGH`. Grep your golden tests for
`approx(1.0)`.

**To minimize the change:** set `"saturation_knee": 0.8` in a `bn_*.json` to
compress only the region that was actually saturating. You cannot recover the
old hard clamp, and you shouldn't want to.

## 2. Session scores combine with noisy-OR, not a weighted average

Session posteriors will generally go **up**, and will never again fall below
the strongest fired hypothesis. The shipped IAM sample moves from 0.65 to
0.972 — and from `HIGH` to `CRITICAL`.

**Why:** the average let true positives dilute the verdict. Four correctly
fired generic hypotheses dragged that session below the mandate violation
driving it. See [scoring.md](scoring.md#the-problem-with-averaging).

**To keep the old behavior,** in `config/hypothesis.json`:

```json
{ "session_combination": "weighted_average" }
```

It is supported and tested, kept for reproducing old runs. Understand that it
still dilutes.

**If your rules overlap heavily,** consider `"max"` instead — noisy-OR assumes
independent causes, and a catalog with shared patterns across hypotheses does
not satisfy that. The trade-offs are laid out in
[scoring.md](scoring.md#the-independence-assumption-and-where-this-catalog-breaks-it).

## 3. Invalid BN configs now raise instead of scoring quietly

`BayesianNetworkEngine` validates `cpd_parameters` at load. A mistyped weight
key, a boost naming an undeclared node, a declared node no parameter mentions,
or an out-of-range value is now a `ConfigValidationError` with a suggestion:

```
credential_access: weight key 'password_pattern_weight' does not match any
declared evidence node (did you mean 'password_patterns_weight'?)
```

**Why:** the old behavior was to skip the key silently. A typo meant the
hypothesis scored at its base rate forever with nothing in the output to say so.

**If a config of yours now fails to load, it was already broken** — it just
wasn't telling you. Fix the key; the error names it. For a throwaway experiment
at a REPL, `BayesianNetworkEngine(path, validate=False)` bypasses the check. The
CLI never does.

`tests/test_config_validation.py::test_every_shipped_bn_config_is_valid` runs
the validator over the whole catalog, so add your own configs to a similar test.

## 4. New: the optional Jev evidence channel

Off by default; no behavior change unless you enable it. Read
[jev-channel.md](jev-channel.md) — particularly the data-egress section —
before turning it on.

---

## Also changed

- **Evidence values are floats.** `Verdict.evidence` is `Dict[str, float]`.
  JSON output still renders `0`/`1` for binary pattern evidence and a rounded
  probability only where the Jev channel raised a node, so parsers that expect
  integers keep working for offline runs.
- **Posteriors print to 3 decimals**, not 2. Two decimals rounded 0.9997 to
  "1.00", reintroducing the appearance of certainty the knee exists to remove.
- **`SessionReport` gained** `combination` and `jev`, and `to_dict()` gained
  `combination` and `evidence_channels`. Existing keys are unchanged.
- **`explain()` breakdowns gained** `raw_score`, `saturation_knee`, and
  `saturated_probability`. `raw_probability` is still present as an alias for
  `raw_score`.

## Upgrade checklist

1. Run `pytest`. Failures are your pinned numbers, and every failure above
   maps to one of the four changes.
2. Re-run your golden logs and diff the reports rather than eyeballing them.
3. Re-check any alerting threshold keyed on the session posterior — noisy-OR
   raises it.
4. Decide on `session_combination` explicitly and commit it, rather than
   inheriting the default by accident.
