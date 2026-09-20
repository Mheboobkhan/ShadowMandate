# Log formats and available fields

[`log_parser.py`](../agentic_detection/log_parser.py) auto-detects the format
per line, so a file may mix them. Every format normalizes to the same event
dict, and that dict is what your rules' `search_fields` and `condition.field`
resolve against.

---

## Supported formats

### AWS Bedrock `ModelInvocationLog` (ndjson)

A line starting with `{` that parses as JSON with
`"schemaType": "ModelInvocationLog"` is parsed structurally — prompt text, tool
calls, token counts, and routing metadata come from their nested paths rather
than being regexed out of a string. This is the richest input and the one the
sample data under [`data/raw/`](../data/raw) uses.

### `key=value` text

The fallback for everything else:

```
time=2025-09-26T13:22:59.220-04:00 level=INFO msg="New update available at https://example.com/x.dmg"
```

Bare tokens, `"double quoted"`, and `'single quoted'` values are all handled.
Unmatched trailing text is ignored, so a partially-structured line still yields
whatever fields it does have.

### Plain text

A line with no recognizable `key=value` pairs still becomes an event: the whole
line is the `message`, a leading ISO-8601 timestamp is extracted if present,
and URLs and file paths are pulled out of the text.

---

## The event dict

```python
{
    "timestamp":   str | None,
    "message":     str,
    "action_type": str,
    "action_details": {...},
    "raw":         {...},
    "raw_line":    str,
    "line_number": int,
}
```

### Field resolution order

A rule naming `"search_fields": ["tool_names"]` doesn't need to know where
`tool_names` lives. Lookup tries, in order:

1. the top-level event key
2. `action_details`
3. `raw`

The first non-`None` wins. This is why `message`, `prompt_text`,
`input_token_count`, and a raw log's `level` field are all addressable the same
way.

### `action_type`

Assigned by first keyword match against `message`, in this order — order
matters, the first match wins:

| `action_type` | Triggered by |
|---|---|
| `external_download` | `download`, `.exe`, `.dmg`, `.pkg`, `.tar.gz`, `.zip`, `bundle` |
| `update_check` | `update available`, `checking for update`, `update_check` |
| `http_request` | `http://`, `https://`, `GET `, `POST `, `PUT `, `DELETE ` |
| `dns_lookup` | `dns`, `lookup`, `resolve` |
| `file_write` | `wrote file`, `writing`, `saved to`, `file_path` |
| `credential_access` | `password`, `secret`, `credential`, `api_key`, `token=` |
| `process_exec` | `exec`, `spawn`, `subprocess`, `started process` |
| `connection` | `connect`, `connection`, `socket` |
| `other` | nothing matched (and no URL was found) |

Bedrock records bypass this entirely: they are `tool_invocation` when the model
emitted a tool call, `model_invocation` otherwise.

> `action_type` is a coarse convenience bucket with the same substring-matching
> weaknesses as any pattern list. Prefer matching on `message`, `tool_names`, or
> a structured field over branching on `action_type`.

---

## `action_details` by source

### From Bedrock records

| Field | Source |
|---|---|
| `prompt_text` | concatenated user-role text blocks from `input.inputBodyJson.messages` |
| `tool_names` | comma-joined `toolUse.name` values from output content blocks |
| `input_token_count` / `output_token_count` | `input.inputTokenCount` / `output.outputTokenCount` |
| `output_input_ratio` | output ÷ max(input, 1) — for `excessive_use_of_tokens` |
| `region` / `inference_region` | `region` / `inferenceRegion` — compare the two for cross-region abuse |
| `model_id` | `modelId` |
| `identity_arn` | `identity.arn` |
| `user_id` | `input.inputBodyJson.metadata.user_id` |
| `operation` | `operation` |
| `request_id` | `requestId` |

`message` is set to `prompt_text`, so a rule searching `message` sees the
prompt.

### From key=value and plain text

| Field | Source |
|---|---|
| `url` | a `url=` field, or the first URL found in the message |
| `has_external_url` | whether either of the above produced a URL |
| `file_path` | a `file_path=` / `path=` / `file=` field, or a Windows/Unix path found in the message |
| `host` `domain` `ip` `port` `status` `level` | passed through when present as `key=value` |

---

## Matching on what isn't there

`condition` nodes treat a missing field as **not matched**, never as zero. A
rule comparing `input_token_count` against a threshold will not fire on a
plaintext log that has no token counts, rather than firing on an implicit 0.
This is what lets one catalog run over mixed log sources without generating
noise from the sources that lack the metadata.

## Adding a format

Extend `AgentLogParser.parse_log_line` with a detector and a
`_parse_<format>` method that returns the normalized dict. Populate
`action_details` with named fields rather than leaving data in `raw` — rules
that name a field read far better than rules that dig through a nested blob,
and a named field is what makes a hypothesis portable across log sources.
