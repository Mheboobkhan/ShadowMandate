"""Tests for AgentLogParser: key=value parsing and Bedrock ModelInvocationLog
structural parsing."""

import json

from agentic_detection.log_parser import AgentLogParser


def test_parse_kv_line_extracts_timestamp_message_and_url():
    parser = AgentLogParser()
    line = 'time=2025-01-01T00:00:00Z msg="Downloading file from https://example.com/bad.exe"'
    event = parser.parse_log_line(line)
    assert event["timestamp"] == "2025-01-01T00:00:00Z"
    assert event["action_type"] == "external_download"
    assert event["action_details"]["has_external_url"] is True
    assert event["action_details"]["url"] == "https://example.com/bad.exe"


def test_categorize_credential_access():
    parser = AgentLogParser()
    event = parser.parse_log_line('msg="reading password from vault"')
    assert event["action_type"] == "credential_access"


def test_categorize_other_when_nothing_matches():
    parser = AgentLogParser()
    event = parser.parse_log_line('msg="just a routine status update"')
    assert event["action_type"] == "other"


def test_bedrock_invocation_log_parsed_structurally():
    doc = {
        "schemaType": "ModelInvocationLog",
        "timestamp": "2025-01-01T00:00:00Z",
        "region": "us-east-1",
        "input": {
            "inputTokenCount": 100,
            "inputBodyJson": {"messages": [{"role": "user", "content": "hello there"}]},
        },
        "output": {"outputTokenCount": 20, "outputBodyJson": []},
        "identity": {"arn": "arn:aws:sts::123456789012:assumed-role/x/y"},
    }
    line = json.dumps(doc)
    parser = AgentLogParser()
    event = parser.parse_log_line(line)

    assert event["message"] == "hello there"
    assert event["action_type"] == "model_invocation"
    assert event["action_details"]["input_token_count"] == 100
    assert event["action_details"]["output_token_count"] == 20
    assert event["action_details"]["output_input_ratio"] == 20 / 100


def test_bedrock_invocation_log_with_tool_use_is_categorized_as_tool_invocation():
    doc = {
        "schemaType": "ModelInvocationLog",
        "timestamp": "2025-01-01T00:00:00Z",
        "input": {"inputBodyJson": {"messages": [{"role": "user", "content": "list my files"}]}},
        "output": {
            "outputBodyJson": [
                {"type": "content_block_start", "content_block": {"toolUse": {"name": "list_files"}}}
            ]
        },
    }
    parser = AgentLogParser()
    event = parser.parse_log_line(json.dumps(doc))
    assert event["action_type"] == "tool_invocation"


def test_parse_log_file_skips_blank_lines_and_tracks_line_numbers(tmp_path):
    path = tmp_path / "sample.log"
    path.write_text('msg="hello"\n\n\nmsg="world"\n')
    parser = AgentLogParser()
    events = parser.parse_log_file(path)
    assert len(events) == 2
    assert events[0]["line_number"] == 1
    assert events[1]["line_number"] == 4


def test_parse_log_file_missing_raises():
    parser = AgentLogParser()
    try:
        parser.parse_log_file("/nonexistent/path.log")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
