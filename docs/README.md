# ShadowMandate documentation

The [README](../README.md) makes the case for the project. These pages are for
people who have decided to run it, extend it, or argue with its numbers.

| Page | Read it when |
|---|---|
| [architecture.md](architecture.md) | You want to know what happens between a log line and a verdict, and which module owns which decision. |
| [scoring.md](scoring.md) | You want to understand — or change — how evidence becomes a probability, and how 13 hypotheses become one session score. |
| [writing-rules.md](writing-rules.md) | You are adding a hypothesis, adding a role, or tuning existing weights. Full `dr_*.json` / `bn_*.json` schema reference. |
| [log-formats.md](log-formats.md) | Your logs aren't being parsed, or you need to know which fields your rules can match on. |
| [cli.md](cli.md) | You want the flag reference and the output JSON schema. |
| [jev-channel.md](jev-channel.md) | You are considering enabling the optional Jev evidence channel. Read the privacy section before you do. |
| [upgrading.md](upgrading.md) | You are coming from 1.0.x and your pinned numbers changed. |

## Orientation in one paragraph

ShadowMandate reads an agent's session log and answers one question: *did this
agent do things outside what it was supposed to be doing?* It answers it by
running a catalog of **hypotheses** against the log. Each hypothesis is two
JSON files your team writes: a detection rule (`dr_*.json`) saying what
evidence to look for, and a Bayesian network config (`bn_*.json`) saying how
much each piece of evidence matters. A **role** adds mandate hypotheses on top
of the generic catalog — the rules that encode what *this particular agent*
was scoped to do. Nothing is learned from data; every number is one a person
typed and a reviewer approved.

## The shortest possible tour

```bash
# Score one session against a role's mandate plus the whole generic catalog
python detect_drift.py data/raw/iam_investigator_session.ndjson \
    --agent-id iam-agent-01 --role iam_investigator

# Test one hypothesis in isolation while you're writing it
python detect_drift.py data/raw/app.log \
    --agent-id test --behavior external_connection

# Run the tests (they pin every documented number in these pages)
pip install -r requirements-dev.txt && pytest
```

No dependencies are required for any of that. `pgmpy` and `typesafe-sdk` are
both optional and both off by default.
