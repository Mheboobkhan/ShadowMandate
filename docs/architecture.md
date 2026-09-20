# Architecture

## The pipeline

One agent log goes in; one session verdict comes out. Everything in between is
six small modules, none of which knows anything about the others' internals.

```
                                 agent log file
                                       |
                              [ log_parser.py ]
                      normalizes each line into an event dict
                                       |
                                  events[]
                                       |
              +------------------------+------------------------+
              |                                                 |
     [ rule_engine.py ]                              [ jev_channel.py ]
   substring / numeric matching                   natural-language questions
   per evidence node -> 0 or 1                    per evidence node -> 0..1
   always runs, offline, free                     opt-in, one API call, off by default
              |                                                 |
              +------------------------+------------------------+
                                       |
                             merge_evidence(max)
                                       |
                            {node: probability} per hypothesis
                                       |
                              [ bn_engine.py ]
                    base + weights + boosts, then soft saturation
                                       |
                          posterior per hypothesis
                                       |
                          [ verdict_generator.py ]
                     risk band, threshold, confidence, recommendation
                                       |
                          [ session_analyzer.py ]
                   combine_posteriors() over fired hypotheses
                                       |
                               SessionReport
```

## Who owns which decision

| Module | Owns | Explicitly does not own |
|---|---|---|
| [`log_parser.py`](../agentic_detection/log_parser.py) | Turning bytes into events: timestamps, action types, extracted URLs/paths, Bedrock JSON flattening | Any notion of what is suspicious |
| [`rule_engine.py`](../agentic_detection/rule_engine.py) | Matching a rule file's `evidence_mapping` against events | How much a match is worth |
| [`jev_channel.py`](../agentic_detection/jev_channel.py) | Asking a model the same questions in natural language, and degrading safely when it can't | Ever lowering a pattern hit |
| [`bn_engine.py`](../agentic_detection/bn_engine.py) | Turning an evidence vector into one probability | Which evidence is present |
| [`config_validation.py`](../agentic_detection/config_validation.py) | Refusing to load a config whose numbers can't be read | Guessing what the author meant |
| [`verdict_generator.py`](../agentic_detection/verdict_generator.py) | Probability → risk band, verdict, confidence, recommendation text | Combining hypotheses |
| [`session_analyzer.py`](../agentic_detection/session_analyzer.py) | Resolving which hypotheses apply, running them, combining them | Any per-hypothesis math |
| [`evidence_detector.py`](../agentic_detection/evidence_detector.py) | Persisting evidence vectors to CSV for audit/batch use | Anything on the scoring path |

## The event shape

Everything downstream of the parser sees this dict:

```python
{
    "timestamp":   "2025-09-26T13:22:59.220-04:00",
    "message":     "New update available at https://...",
    "action_type": "external_download",
    "action_details": {"url": "https://...", "has_external_url": True, ...},
    "raw":         {...},   # parsed key=value fields, or the whole JSON document
    "raw_line":    "the original log line",
    "line_number": 42,
}
```

A rule's `search_fields` resolve against this dict in a fixed order — top-level
key, then `action_details`, then `raw` — so a rule can name `message`,
`tool_names`, or `inputTokenCount` without caring which layer it came from. See
[log-formats.md](log-formats.md) for the full field inventory.

## Hypothesis resolution

`config/hypothesis.json` is the only entry point into the rule catalog:

```
hypothesis.json
├── session_combination            how per-hypothesis scores become one score
├── generic_behaviors[]            the catalog that runs against every agent
│   └── {id, detection_rules, bn_config, output_csv, session_weight}
└── roles[]
    ├── {id, name, manifest}
    └── mandate_behaviors[]        this role's out-of-scope rules
        └── {id, detection_rules, bn_config, session_weight}
```

`--role X` runs every `generic_behaviors` entry plus role `X`'s
`mandate_behaviors`. `--behavior Y` runs exactly one entry from either list,
which is what you want while writing a rule.

`session_weight` is the only place the mandate/generic distinction becomes a
number. Generic hypotheses ship at `1.0`; the IAM mandate hypothesis ships at
`2.5`, so a mandate breach dominates the session score even when generic rules
fire alongside it. See [scoring.md](scoring.md#combining-hypotheses) for what
that weight actually does under each combination strategy.

## Design commitments

These are properties the test suite enforces, not just intentions.

**Zero required dependencies.** The whole pipeline runs on the standard
library. `pgmpy` and `typesafe-sdk` are optional, and their absence is a
supported configuration, not a degraded one.

**Nothing is learned.** There is no training step, no weight fitting, no
feedback loop from past verdicts. Changing what "suspicious" means requires a
commit.

**Enabling a channel is monotonic.** Turning on Jev can raise an evidence node
but never lower one, so a detection that fires today still fires tomorrow.

**A degraded run says so.** If the Jev channel was requested and didn't
deliver, the CLI prints a warning and the JSON records the status. A clean
verdict from a half-working pipeline is worse than no verdict.

**The scan does not die on an integration.** Any failure inside the Jev channel
is caught, recorded, and stepped over.
