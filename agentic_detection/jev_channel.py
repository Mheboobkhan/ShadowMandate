"""
jev_channel.py - Optional second evidence channel (TypeSafe AI's Jev)
=======================================================================

The pattern channel in rule_engine.py asks "does this log line contain one of
these substrings?". That is exact, offline, auditable, and free - and it is
blind to paraphrase. ``"ignore previous instructions"`` is in the injection
pattern list; ``"disregard everything you were told before"`` is not, and
scores zero.

This module adds an opt-in second channel that asks the same question in
natural language instead. Jev is TypeSafe AI's "System One" model: state in,
typed calibrated probabilities out. A ``Noul`` question returns a probability
in [0, 1] rather than a bit, which is exactly the shape the evidence layer
wants, and every question in a request is answered in one parallel pass, so
all of a session's evidence nodes cost roughly one round trip.

Design constraints this module holds to:

* **Off by default.** No import of the SDK, no network call, and no behavior
  change unless the operator explicitly enables the channel.
* **Additive, never subtractive.** The pattern channel still runs. Evidence is
  combined as ``max(pattern, jev)``, so enabling Jev can only raise a node's
  value - a detection that fires today still fires.
* **Never fails the scan.** A missing SDK, missing key, timeout, or API error
  degrades to patterns-only and is *recorded in the report*, so a human
  reading the verdict can see the channel did not run rather than silently
  trusting a weaker result.
* **The human still authors the question.** The instruction text lives in the
  same ``dr_*.json`` file, in git, reviewed like any other rule change.

Note on data egress: enabling this channel sends the selected log content -
which may include raw prompts, ARNs, and credential-adjacent text - to
``api.typesafe.ai``. That is a deliberate operator decision, which is why the
default is off. See docs/jev-channel.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Characters of state to send. Jev's context window is 32k tokens; at a
#: conservative ~4 chars/token this leaves generous room for the questions.
DEFAULT_MAX_STATE_CHARS = 60_000

#: Per-event message truncation, applied before the global state budget.
DEFAULT_MAX_MESSAGE_CHARS = 2_000

#: Questions per request. All questions in a request answer in parallel, so
#: batching is nearly free; this only guards against a pathological catalog.
DEFAULT_MAX_QUESTIONS_PER_CALL = 16

#: Pin the model version. Leaving this floating means detection thresholds
#: shift underneath you when TypeSafe ships a new revision - the exact silent
#: drift this project promises not to have.
DEFAULT_MODEL = "jev-latest"

ENV_ENABLE = "SHADOWMANDATE_JEV"
ENV_MODEL = "SHADOWMANDATE_JEV_MODEL"
ENV_API_KEY = "TYPESAFE_API_KEY"


@dataclass(frozen=True)
class JevQuestion:
    """One natural-language evidence question, bound to a behavior's node."""

    behavior_id: str
    node: str
    instructions: str

    @property
    def key(self) -> str:
        """The key used in the Jev request/response payload."""
        return f"{self.behavior_id}__{self.node}"


@dataclass
class JevOutcome:
    """What the channel did, including why it did nothing.

    `status` is one of:
      ``disabled``     - operator did not enable the channel
      ``ok``           - answers returned
      ``unavailable``  - SDK not installed, or no API key
      ``no_questions`` - enabled, but no node opted in
      ``error``        - the call failed; scan continued on patterns alone
    """

    status: str = "disabled"
    detail: str = "Jev channel not enabled"
    probabilities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    questions_asked: int = 0
    state_chars: int = 0
    state_truncated: bool = False
    model: Optional[str] = None

    @property
    def ran(self) -> bool:
        return self.status == "ok"

    @property
    def degraded(self) -> bool:
        """True when the channel was asked for but did not deliver answers."""
        return self.status in ("unavailable", "error")

    def for_behavior(self, behavior_id: str) -> Dict[str, float]:
        """The {node: probability} answers belonging to one behavior."""
        return {
            node: p for (bid, node), p in self.probabilities.items()
            if bid == behavior_id
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "questions_asked": self.questions_asked,
            "state_chars": self.state_chars,
            "state_truncated": self.state_truncated,
            "model": self.model,
        }


# ----------------------------------------------------------------------
# Question construction
# ----------------------------------------------------------------------

def build_questions(dr_config: Dict[str, Any]) -> List[JevQuestion]:
    """Derive the Jev questions for one dr_*.json detection-rule config.

    A node opts in by default when it is a *pattern* node, because its
    ``description`` is already written as the judgment a human wants made.
    A node is skipped when:

    * it uses a numeric/structural ``condition`` (token counts, region
      mismatches) - those are exact comparisons on structured fields and a
      probabilistic model can only make them worse;
    * it sets ``"jev": false`` - an explicit opt-out for nodes where an exact
      test is correct (e.g. matching a ``.pem`` file extension);
    * no instruction text can be derived.

    An explicit ``"jev": {"instructions": "..."}`` overrides the description
    when the rule author wants to phrase the question differently from the
    documentation string.
    """
    behavior_id = dr_config.get("behavior")
    if not behavior_id:
        return []

    questions: List[JevQuestion] = []
    for node, node_config in (dr_config.get("evidence_mapping") or {}).items():
        if not isinstance(node_config, dict):
            continue

        jev_config = node_config.get("jev", None)
        if jev_config is False:
            continue
        if node_config.get("condition") is not None and not isinstance(jev_config, dict):
            continue

        instructions = None
        if isinstance(jev_config, dict):
            instructions = jev_config.get("instructions")
        if not instructions:
            instructions = node_config.get("description")
        if not instructions:
            continue

        questions.append(
            JevQuestion(behavior_id=behavior_id, node=node, instructions=instructions)
        )
    return questions


# ----------------------------------------------------------------------
# State construction
# ----------------------------------------------------------------------

_STATE_FIELDS = (
    "tool_names",
    "file_path",
    "url",
    "prompt_text",
    "identity_arn",
    "model_id",
    "region",
    "inference_region",
    "operation",
)


def build_state(
    events: Sequence[Dict[str, Any]],
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> Tuple[Dict[str, Any], bool]:
    """Compact parsed events into a Jev `state` payload.

    Returns ``(state, truncated)``. Log sessions routinely exceed Jev's 32k
    context, so events are compacted field-by-field and then, if still over
    budget, dropped from the middle - keeping the start and end of the session,
    which is where an agent's objective and its final actions both live. The
    number dropped is reported inside the state itself so the model is not
    silently reasoning over a gap.
    """
    compacted: List[Dict[str, Any]] = []
    for event in events:
        details = event.get("action_details") or {}
        entry: Dict[str, Any] = {
            "ts": event.get("timestamp"),
            "action": event.get("action_type"),
        }
        message = event.get("message") or ""
        if message:
            entry["message"] = message[:max_message_chars]
        for key in _STATE_FIELDS:
            value = details.get(key)
            if value in (None, "", []):
                continue
            if key == "prompt_text" and entry.get("message") == value:
                continue
            entry[key] = str(value)[:max_message_chars]
        compacted.append(entry)

    def size(entries: Sequence[Dict[str, Any]]) -> int:
        return sum(len(str(e)) for e in entries)

    truncated = False
    dropped = 0
    while len(compacted) > 2 and size(compacted) > max_state_chars:
        # Drop from the middle: keep the session's opening and its tail.
        middle = len(compacted) // 2
        compacted.pop(middle)
        dropped += 1
        truncated = True

    state: Dict[str, Any] = {
        "description": (
            "Chronological activity log for a single AI agent session. "
            "Answer each question about this session as a whole."
        ),
        "event_count": len(events),
        "events": compacted,
    }
    if truncated:
        state["note"] = (
            f"{dropped} event(s) from the middle of this session were omitted to "
            f"fit the context window; the opening and closing events are intact."
        )
    return state, truncated


# ----------------------------------------------------------------------
# The channel
# ----------------------------------------------------------------------

class JevChannel:
    """Evaluates evidence questions against a session using TypeSafe's Jev."""

    def __init__(
        self,
        enabled: bool = False,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        max_questions_per_call: int = DEFAULT_MAX_QUESTIONS_PER_CALL,
    ) -> None:
        self.enabled = enabled
        self.model = model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get(ENV_API_KEY)
        self.max_state_chars = max_state_chars
        self.max_questions_per_call = max_questions_per_call

    @classmethod
    def from_env(cls, enabled: Optional[bool] = None, **kwargs: Any) -> "JevChannel":
        """Build a channel, reading the enable flag from the environment.

        An explicit `enabled` argument (the --jev CLI flag) wins; otherwise
        SHADOWMANDATE_JEV=1 turns the channel on.
        """
        if enabled is None:
            enabled = os.environ.get(ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")
        return cls(enabled=enabled, **kwargs)

    # ------------------------------------------------------------------

    def evaluate(
        self,
        events: Sequence[Dict[str, Any]],
        questions: Iterable[JevQuestion],
    ) -> JevOutcome:
        """Ask Jev every question about this session. Never raises."""
        questions = list(questions)

        if not self.enabled:
            return JevOutcome(
                status="disabled",
                detail="Jev channel not enabled (use --jev or SHADOWMANDATE_JEV=1)",
            )
        if not questions:
            return JevOutcome(
                status="no_questions",
                detail="No evidence node opted in to the Jev channel",
                model=self.model,
            )

        try:
            client, noul_cls = self._client()
        except _ChannelUnavailable as exc:
            return JevOutcome(status="unavailable", detail=str(exc), model=self.model)

        state, truncated = build_state(events, max_state_chars=self.max_state_chars)
        state_chars = len(str(state))

        probabilities: Dict[Tuple[str, str], float] = {}
        try:
            for batch in self._batches(questions):
                payload = {q.key: noul_cls(instructions=q.instructions) for q in batch}
                response = client.system_one(state=state, questions=payload)
                answers = getattr(response, "answers", {}) or {}
                for q in batch:
                    answer = answers.get(q.key)
                    value = self._probability(answer)
                    if value is not None:
                        probabilities[(q.behavior_id, q.node)] = value
        except Exception as exc:  # noqa: BLE001 - a scan must never die here
            return JevOutcome(
                status="error",
                detail=f"{type(exc).__name__}: {exc}",
                probabilities=probabilities,
                questions_asked=len(questions),
                state_chars=state_chars,
                state_truncated=truncated,
                model=self.model,
            )

        return JevOutcome(
            status="ok",
            detail=f"Answered {len(probabilities)}/{len(questions)} questions",
            probabilities=probabilities,
            questions_asked=len(questions),
            state_chars=state_chars,
            state_truncated=truncated,
            model=self.model,
        )

    # ------------------------------------------------------------------

    def _batches(self, questions: List[JevQuestion]) -> Iterable[List[JevQuestion]]:
        size = max(1, self.max_questions_per_call)
        for i in range(0, len(questions), size):
            yield questions[i: i + size]

    def _client(self):
        try:
            from typesafe_sdk import Noul, TypeSafeClient  # type: ignore
        except ImportError as exc:
            raise _ChannelUnavailable(
                "typesafe-sdk is not installed (pip install typesafe-sdk)"
            ) from exc

        if not self.api_key:
            raise _ChannelUnavailable(
                f"no API key found in ${ENV_API_KEY}"
            )

        try:
            client = TypeSafeClient(api_key=self.api_key, model=self.model)
        except TypeError:
            # Older/newer SDK signatures: fall back to env-based construction.
            client = TypeSafeClient(model=self.model)
        return client, Noul

    @staticmethod
    def _probability(answer: Any) -> Optional[float]:
        """Pull the 0-1 probability out of a Jev Noul answer."""
        if answer is None:
            return None
        value = getattr(answer, "noul", None)
        if value is None and isinstance(answer, dict):
            value = answer.get("noul")
        if value is None and isinstance(answer, (int, float)):
            value = answer
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None


class _ChannelUnavailable(RuntimeError):
    """The channel cannot run at all (no SDK, no credentials)."""


def merge_evidence(
    pattern_evidence: Dict[str, int],
    jev_probabilities: Dict[str, float],
) -> Dict[str, float]:
    """Combine the two channels into one evidence vector.

    ``max`` is deliberate: the pattern channel is exact and its hits are
    trusted outright, so Jev can raise a node that patterns missed but can
    never talk one down. Enabling the channel is therefore monotonic - nothing
    that fires today stops firing.
    """
    merged: Dict[str, float] = {node: float(v) for node, v in pattern_evidence.items()}
    for node, probability in jev_probabilities.items():
        if node in merged:
            merged[node] = max(merged[node], float(probability))
    return merged
