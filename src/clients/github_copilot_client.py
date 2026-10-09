"""Isolated single-question Copilot SDK adapter for GPT and other Copilot models.

The optional SDK is imported lazily. Existing CLI authentication is used only
when no explicit token is supplied. Decision workspaces and sessions are isolated;
no device login, workspace instructions, or cross-question history is reused.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ._llm_choice import (
    SYSTEM_MESSAGE, ANSWER_ONLY_SYSTEM_MESSAGE, parse_answer_only,
    parse_choice, prompt_for, reject_unexpected_reasoning,
    token_usage, validate_settings,
)
from .base import (
    BaseDecisionClient, ClientCapabilities, ClientClosedError, ClientTimeoutError,
    ClientTransportError, DecisionClientError, DecisionRequest, DecisionResponse,
    InvalidResponseError, UnsupportedRequestError,
)

FUNCTION_CALL_SYSTEM_MESSAGE = (
    "Solve the supplied choice problem using only its state, question, and options. "
    "Submit your final answer by calling submit_decision exactly once with "
    'arguments {"choice": "<option id>"}. Use an option ID verbatim. '
    "Do not answer in text or call any other tool. Do not use external information."
)

_DECISION_EVENT_TYPES = frozenset({
    "assistant.message", "external_tool.requested", "external_tool.completed",
    "assistant.usage", "assistant.turn_start", "assistant.turn_end", "assistant.turn_retry",
    "session.error", "session.idle", "tool.execution_start", "tool.execution_complete",
    "session.model_change", "assistant.reasoning", "assistant.reasoning_delta",
})


def _decision_tool(request: DecisionRequest):
    from copilot.tools import Tool

    # Declaration-only: the SDK leaves the call pending, never sends a result,
    # and cannot trigger another model round before we abort the session.
    return Tool(
        name="submit_decision", description="Submit the single final option ID.",
        handler=None, skip_permission=True, defer="never",
        parameters={
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": [o.id for o in request.options]}},
            "required": ["choice"], "additionalProperties": False,
        },
    )


def _tool_choice(arguments, request: DecisionRequest) -> str:
    if isinstance(arguments, str):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate argument key")
                result[key] = value
            return result

        try:
            arguments = json.loads(arguments, object_pairs_hook=unique_object)
        except (ValueError, RecursionError):
            raise InvalidResponseError("Copilot returned invalid tool JSON arguments") from None
    if (not isinstance(arguments, dict) or set(arguments) != {"choice"}
            or not isinstance(arguments["choice"], str)
            or arguments["choice"] not in {o.id for o in request.options}):
        raise InvalidResponseError("Copilot tool choice must match exactly one option ID",
                                   diagnostic_code="invalid_choice")
    return arguments["choice"]


def _verify_disabled_reasoning(content, *, actual_effort, reasoning_tokens, reasoning):
    if actual_effort != "none" or type(reasoning_tokens) is not int or reasoning_tokens != 0:
        raise InvalidResponseError(
            "Copilot no-think requires reported effort none and exactly zero reasoning tokens",
            diagnostic_code="unexpected_reasoning")
    reject_unexpected_reasoning(content, reasoning=reasoning, reasoning_tokens=reasoning_tokens)


def _function_call_response(
    request: DecisionRequest, events: list[dict[str, Any]], *,
    expected_session_id: str | None = None, think: bool | None = None,
    reasoning_effort: str | None = None,
) -> DecisionResponse:
    def require(condition, message):
        if not condition:
            raise InvalidResponseError(message)

    require(isinstance(events, list), "Copilot events must be a JSON array")
    grouped = {}
    ids = set()
    for event in events:
        require(isinstance(event, dict) and isinstance(event.get("type"), str)
                and event["type"] in _DECISION_EVENT_TYPES
                and isinstance(event.get("data"), dict), "Invalid Copilot event envelope")
        event_id = event.get("id")
        require(isinstance(event_id, str) and bool(event_id) and event_id not in ids,
                "Missing or duplicate Copilot event ID")
        ids.add(event_id)
        grouped.setdefault(event["type"], []).append(event["data"])
    if grouped.get("session.error"):
        raise ClientTransportError("Copilot function-call session failed")
    require(not grouped.get("assistant.turn_retry"), "Copilot runtime retried")
    for kind in ("assistant.usage", "assistant.turn_start", "assistant.turn_end",
                 "tool.execution_start", "tool.execution_complete", "external_tool.completed", "session.idle",
                 "session.model_change"):
        require(len(grouped.get(kind, [])) <= 1, "Copilot returned multiple model turns or tool executions")
    require(len(grouped.get("assistant.message", [])) == 1
            and len(grouped.get("external_tool.requested", [])) == 1,
            "Copilot must return exactly one complete decision tool call")
    message = grouped["assistant.message"][0]
    call = grouped["external_tool.requested"][0]
    kinds = [event["type"] for event in events]
    message_index, call_index = kinds.index("assistant.message"), kinds.index("external_tool.requested")
    require(message_index < call_index, "Copilot tool request precedes assistant message")
    for kind in ("assistant.turn_start", "session.model_change", "tool.execution_start"):
        require(kind not in kinds or kinds.index(kind) < (
            call_index if kind == "tool.execution_start" else message_index),
            "Copilot model/tool start event is out of order")
    for kind in ("assistant.turn_end", "external_tool.completed", "tool.execution_complete", "session.idle"):
        require(kind not in kinds or kinds.index(kind) > call_index,
                "Copilot cleanup event precedes tool request")
    tools = message.get("toolRequests")
    require(isinstance(tools, list) and len(tools) == 1 and isinstance(tools[0], dict),
            "Copilot returned missing or duplicate tool calls")
    tool = tools[0]
    call_id = tool.get("toolCallId")
    require(isinstance(call_id, str) and bool(call_id)
            and tool.get("name") == call.get("toolName") == "submit_decision"
            and call.get("toolCallId") == call_id
            and isinstance(call.get("requestId"), str) and bool(call["requestId"])
            and isinstance(call.get("sessionId"), str) and bool(call["sessionId"])
            and (expected_session_id is None or call["sessionId"] == expected_session_id),
            "Copilot returned inconsistent or unexpected tool calls")
    choice = _tool_choice(tool.get("arguments"), request)
    require(_tool_choice(call.get("arguments"), request) == choice,
            "Copilot reported inconsistent tool arguments")
    for start in grouped.get("tool.execution_start", []):
        require(start.get("toolName") == "submit_decision" and start.get("toolCallId") == call_id
                and _tool_choice(start.get("arguments"), request) == choice,
                "Copilot reported unexpected tool execution")
    for end in grouped.get("tool.execution_complete", []):
        require(end.get("toolCallId") == call_id and end.get("success") is not True,
                "Copilot unexpectedly completed a tool result")
    for end in grouped.get("external_tool.completed", []):
        require(end.get("requestId") == call["requestId"], "Copilot external completion ID mismatch")
    turn_ids = {
        data["turnId"] for kind in ("assistant.turn_start", "assistant.turn_end", "tool.execution_start",
                                   "assistant.message")
        for data in grouped.get(kind, []) if isinstance(data.get("turnId"), str)
    }
    require(len(turn_ids) <= 1, "Copilot reported inconsistent model turns")
    usage = grouped.get("assistant.usage", [{}])[0]
    for field, expected in (("availableToolCount", 1), ("numToolCalls", 1),
                            ("toolCounts", {"submit_decision": 1})):
        require(field not in usage or (type(usage[field]) is type(expected) and usage[field] == expected),
                "Copilot reported unexpected tool counts")
    finish = usage.get("finishReason")
    require(finish in (None, "stop", "end_turn", "tool_calls", "function_call"),
            "Copilot completion did not finish normally")
    require(not usage.get("contentFilterTriggered", False), "Copilot completion was filtered")
    usage_model, message_model = usage.get("model"), message.get("model")
    require(not (usage_model and message_model and usage_model != message_model),
            "Copilot reported inconsistent response models")
    model = usage_model or message_model
    require(isinstance(model, str) and bool(model.strip()), "Copilot did not report the response model")
    api_ids = {data["apiCallId"] for data in (usage, message)
               if isinstance(data.get("apiCallId"), str)}
    require(len(api_ids) <= 1, "Copilot reported multiple model API calls")
    actual_effort = usage.get("reasoningEffort")
    require(reasoning_effort is None or actual_effort is None or reasoning_effort == actual_effort,
            "Copilot reported a different reasoning effort than requested")
    for change in grouped.get("session.model_change", []):
        require(change.get("newModel") == model
                and (change.get("reasoningEffort") is None or change["reasoningEffort"] == actual_effort),
                "Copilot model setting event contradicts response metadata")
    if think is False or reasoning_effort == "none":
        require(not grouped.get("assistant.reasoning") and not grouped.get("assistant.reasoning_delta"),
                "Copilot emitted reasoning events for a no-think request")
        _verify_disabled_reasoning(
            message.get("content"), reasoning=(message.get("reasoningText")
            or message.get("reasoningBlocks")), actual_effort=actual_effort,
            reasoning_tokens=usage.get("reasoningTokens"))
    return DecisionResponse(
        request_id=request.request_id, selected_option_id=choice, resolved_model=model,
        usage=token_usage(usage.get("inputTokens"), usage.get("outputTokens"), usage.get("reasoningTokens")),
        raw_output={
            "protocol": "choice-function-call-v1", "termination": "aborted_pending_external_tool",
            "model": model, "finish_reason": finish, "reasoning_effort": actual_effort,
            "events": deepcopy(events),
        },
    )


def replay_function_call(
    request: DecisionRequest, raw_output: dict[str, Any], *,
    audit_events: list[dict[str, Any]] | None = None,
    expected_session_id: str | None = None, think: bool | None = None,
    reasoning_effort: str | None = None,
) -> DecisionResponse:
    """Pure SDK-free replay; return a validated DecisionResponse or typed failure.

    ``audit_events`` is the ordered list of event objects after the caller verifies
    each audit row's request_sha256 against the actual request. Full raw events
    must match audit events in order; only cleanup receipts may be audit-only.
    Both streams are validated, including events after the response snapshot.
    Compare the returned ID, model and TokenUsage to the persisted response.
    With think=False or reasoning_effort="none", require reported effort "none",
    exactly zero reasoning tokens, and no reasoning text/events.
    This checks evidence consistency, not authenticity of externally supplied JSON.
    """
    if request.protocol_version != "choice-v1":
        raise UnsupportedRequestError("Only choice-v1 is supported")
    if (not isinstance(raw_output, dict) or set(raw_output) != {
            "protocol", "termination", "model", "finish_reason", "reasoning_effort", "events"}
            or raw_output["protocol"] != "choice-function-call-v1"
            or raw_output["termination"] != "aborted_pending_external_tool"):
        raise InvalidResponseError("Invalid Copilot function-call raw output")
    settings = {"expected_session_id": expected_session_id, "think": think,
                "reasoning_effort": reasoning_effort}
    response = _function_call_response(request, raw_output["events"], **settings)
    if response.raw_output != raw_output:
        raise InvalidResponseError("Copilot raw metadata does not match SDK events")
    if audit_events is not None:
        audited = _function_call_response(request, audit_events, **settings)
        index = 0
        cleanup_types = {"external_tool.completed", "assistant.turn_end", "session.idle",
                         "tool.execution_complete"}
        for event in audit_events:
            if index < len(raw_output["events"]) and event == raw_output["events"][index]:
                index += 1
            elif event["type"] not in cleanup_types:
                raise InvalidResponseError("Copilot raw events differ from request-bound audit")
        if (index != len(raw_output["events"])
                or audited.selected_option_id != response.selected_option_id
                or audited.resolved_model != response.resolved_model or audited.usage != response.usage
                or any(audited.raw_output[key] != raw_output[key] for key in (
                    "finish_reason", "reasoning_effort"))):
            raise InvalidResponseError("Copilot response differs from request-bound audit")
    return response


class _FunctionCallCapture:
    """Retain SDK events, not assistant text synthesized from a parsed choice."""

    def __init__(self):
        self.ready = asyncio.Event()
        self.events = []
        self.messages = []
        self.turns = 0

    def on_event(self, event):
        kind = getattr(event.type, "value", event.type)
        if kind in _DECISION_EVENT_TYPES:
            self.events.append(event)
        if kind == "assistant.message":
            self.messages.append(event)
            if not getattr(event.data, "tool_requests", None) or len(self.messages) > 1:
                self.ready.set()
        elif kind == "external_tool.requested":
            self.ready.set()
        elif kind == "assistant.turn_start":
            self.turns += 1
            if self.turns > 1:
                self.ready.set()
        elif kind == "session.error":
            self.ready.set()
        elif kind in ("session.idle", "assistant.turn_retry"):
            self.ready.set()

def _sdk_types():
    try:
        from copilot import CopilotClient
        from copilot.session import (
            ModelCapabilitiesOverride, ModelLimitsOverride, PermissionDecisionUserNotAvailable,
        )
    except ImportError:
        raise ImportError("Install the optional Copilot dependency: pip install -e '.[copilot]'") from None
    return CopilotClient, ModelCapabilitiesOverride, ModelLimitsOverride, PermissionDecisionUserNotAvailable


class GitHubCopilotClient(BaseDecisionClient):
    """One fresh isolated session per prediction, using the current asyncio loop.

    Authentication uses an explicit token, COPILOT_GITHUB_TOKEN, or an existing
    SDK/CLI login. Login is never started interactively by this adapter.
    ``think=None`` keeps the model default. ``think=True`` selects a supported
    reasoning effort. ``think=False`` uses explicit "none" when the model
    advertises it; "low" is never substituted. SDK 1.0.14's create-session
    literal excludes "none", so the public set_model API applies it to the
    fresh session, and an authoritative snapshot must confirm it before send.
    Explicit reasoning_effort="none" also requests this verified no-think path.
    Reported effort "none" and exactly zero reasoning tokens are mandatory.
    ``output_format`` selects JSON (default), the shared answer-only protocol,
    or opt-in ``function_call``. The latter exposes only a declaration-only
    submit_decision tool with a per-request choice enum. It aborts after the
    external-tool event without returning a tool result or awaiting session idle.
    No strict/tool_choice API is assumed and there is no text fallback.
    Reasoning, token limits and timeout defaults are unchanged.

    Function-call raw_output uses protocol=choice-function-call-v1 and genuine
    SDK ``events`` (including assistant.message and external_tool.requested).
    It intentionally has no synthetic ``content``: text-only raw collectors and
    verifiers must preserve/replay these events instead. finish_reason is the
    provider's value (or None); termination records our deliberate abort.

    Optional ``on_decision_event(request, event_dict)`` synchronously receives
    detached SDK event JSON before response validation, including failed calls
    and selected cleanup events. The caller owns request binding, redaction,
    durable writes and fsync. It must return None; async callbacks are rejected.
    Callback/serialization failures fail the decision with ClientTransportError,
    even during cleanup; the SDK otherwise logs and swallows handler exceptions.
    No callback is made for initialization or for unselected event types.

    One send per question, no adapter retries. The SDK may retry internally; an
    observed retry/multiple model calls is rejected rather than scored as one.
    Always close the client to release its runtime and temporary session files.
    """

    def __init__(
        self, *, model: str = "gpt-4o", github_token: str | None = None,
        think: bool | None = None, reasoning_effort: str | None = None,
        timeout_seconds: float = 180.0, max_tokens: int = 4096,
        output_format: str = "json",
        on_decision_event: Callable[[DecisionRequest, dict[str, Any]], None] | None = None,
    ) -> None:
        validate_settings(model, think, timeout_seconds, max_tokens)
        if output_format not in ("json", "answer_only", "function_call"):
            raise ValueError("output_format must be json, answer_only, or function_call")
        self.output_format = output_format
        if on_decision_event is not None and (
                not callable(on_decision_event)
                or inspect.iscoroutinefunction(on_decision_event)
                or inspect.iscoroutinefunction(getattr(on_decision_event, "__call__", None))):
            raise ValueError("on_decision_event must be a synchronous callable or None")
        self._on_decision_event = on_decision_event
        if reasoning_effort is not None and reasoning_effort not in ("none", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported SDK reasoning_effort")
        if think is False and reasoning_effort not in (None, "none"):
            raise ValueError("reasoning_effort conflicts with think=False")
        if think is True and reasoning_effort == "none":
            raise ValueError("reasoning_effort=none conflicts with think=True")
        token = os.environ.get("COPILOT_GITHUB_TOKEN") if github_token is None else github_token
        if token is not None and (not isinstance(token, str) or not token
                                  or not token.isascii() or any(c.isspace() for c in token)):
            raise ValueError("GitHub token must be nonempty ASCII without whitespace")
        self.model, self.think, self.reasoning_effort = model, think, reasoning_effort
        self.timeout_seconds, self.max_tokens = timeout_seconds, max_tokens
        self._token = token
        self._runtime = None
        self._started = False
        self._directory = None
        self._model_info = None
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> ClientCapabilities:
        # Specific mode support is checked against the account's model metadata.
        return ClientCapabilities(supports_thinking_control=True)

    async def _ensure_runtime(self) -> None:
        if self._model_info is not None:
            return
        client_type, capabilities_type, limits_type, deny_type = _sdk_types()
        if self._runtime is None:
            self._directory = TemporaryDirectory(prefix="graphdecisionbench-copilot-")
            self._limits = capabilities_type(limits=limits_type(max_output_tokens=self.max_tokens))
            self._deny_permission = lambda *_: deny_type()
            self._runtime = client_type(
                github_token=self._token, use_logged_in_user=self._token is None,
                mode="empty",
                base_directory=(os.environ.get("COPILOT_HOME") or str(Path.home() / ".copilot")
                                if self._token is None else self._directory.name),
                working_directory=self._directory.name, log_level="error",
            )
        if not self._started:
            await self._runtime.start()
            self._started = True
        auth = await self._runtime.get_auth_status()
        if not auth.isAuthenticated:
            raise ClientTransportError("Copilot authentication required; log in separately or set COPILOT_GITHUB_TOKEN")
        models = await self._runtime.list_models()
        self._model_info = next((m for m in models if m.id == self.model), None)
        if self._model_info is None:
            raise UnsupportedRequestError("Requested Copilot model is unavailable for this account")

    def _effort(self) -> str | None:
        info = self._model_info
        supported = getattr(info, "supported_reasoning_efforts", None) or []
        supports = getattr(getattr(info, "capabilities", None), "supports", None)
        reasoning = bool(supported or getattr(supports, "reasoning_effort", False)
                         or getattr(info, "default_reasoning_effort", None))
        effort = self.reasoning_effort
        if self.think is False and reasoning:
            if "none" not in supported:
                raise UnsupportedRequestError("Copilot model does not advertise disabled reasoning; low is not no-think")
            effort = "none"
        if self.think is True and effort is None:
            effort = getattr(info, "default_reasoning_effort", None) or "medium"
            if effort == "none":
                effort = "medium"
        if effort is not None and effort not in supported:
            raise UnsupportedRequestError("Requested reasoning effort is not advertised by this Copilot model")
        return effort

    async def initialize(self) -> None:
        """Fail once for unavailable models/auth/modes, before any scored attempt."""
        async with self._lock:
            if self._closed:
                raise ClientClosedError("Copilot client is closed")
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await self._ensure_runtime()
                    self._effort()
            except TimeoutError:
                raise ClientTimeoutError("Copilot initialization timed out") from None
            except (DecisionClientError, ImportError):
                raise
            except Exception:
                raise ClientTransportError("Copilot initialization failed") from None

    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        if self._closed:
            raise ClientClosedError("Copilot client is closed")
        self.validate_request(request)
        async with self._lock:
            if self._closed:
                raise ClientClosedError("Copilot client is closed")
            session = None
            completed = False
            aborted = False
            usage_events = []
            retried = False
            audit_failed = False
            cancelled = False
            capture = _FunctionCallCapture() if self.output_format == "function_call" else None

            def on_event(event):
                nonlocal retried, audit_failed
                kind = getattr(event.type, "value", event.type)
                if kind == "assistant.usage":
                    usage_events.append(event.data)
                elif kind == "assistant.turn_retry":
                    retried = True
                if capture is not None:
                    capture.on_event(event)
                    if len(usage_events) > 1:
                        capture.ready.set()
                if self._on_decision_event is not None and kind in _DECISION_EVENT_TYPES:
                    try:
                        returned = self._on_decision_event(request, deepcopy(event.to_dict()))
                        if returned is not None:
                            if inspect.iscoroutine(returned):
                                returned.close()
                            raise TypeError("on_decision_event must return None")
                    except Exception:
                        # SDK dispatch swallows exceptions, so retain a sticky
                        # failure and surface it outside the event handler.
                        audit_failed = True
                        if capture is not None:
                            capture.ready.set()

            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await self._ensure_runtime()
                    effort = self._effort()
                    session = await self._runtime.create_session(
                        model=self.model, reasoning_effort=None if effort == "none" else effort,
                        tools=[_decision_tool(request)] if capture is not None else [],
                        available_tools=["submit_decision"] if capture is not None else [],
                        tool_search={"enabled": False},
                        on_permission_request=self._deny_permission,
                        system_message={"mode": "replace", "content": (
                            FUNCTION_CALL_SYSTEM_MESSAGE if capture is not None else
                            ANSWER_ONLY_SYSTEM_MESSAGE if self.output_format == "answer_only" else SYSTEM_MESSAGE)},
                        working_directory=self._directory.name,
                        model_capabilities=self._limits, streaming=False,
                        memory={"enabled": False}, infinite_sessions={"enabled": False},
                        enable_config_discovery=False, skip_custom_instructions=True,
                        enable_on_demand_instruction_discovery=False,
                        enable_skills=False, skill_directories=[], plugin_directories=[],
                        instruction_directories=[], enable_file_hooks=False,
                        enable_host_git_operations=False, enable_session_store=False,
                        enable_file_change_tracking=False, enable_session_telemetry=False,
                        skip_embedding_retrieval=True, mcp_servers={}, enable_mcp_apps=False,
                        request_extensions=False, request_canvas_renderer=False,
                        on_auto_mode_switch_request=lambda *_: "no", on_event=on_event,
                    )
                    if effort == "none":
                        await session.set_model(
                            self.model, reasoning_effort="none", model_capabilities=self._limits)
                        snapshot = await session.rpc.model.get_current()
                        if snapshot.model_id != self.model or snapshot.reasoning_effort != "none":
                            raise ClientTransportError("Copilot did not confirm disabled reasoning before send")
                        if audit_failed:
                            raise ClientTransportError("Copilot decision event audit callback failed")
                    if capture is not None:
                        await session.send(prompt_for(request))
                        await capture.ready.wait()
                        # This is mandatory, not best-effort cleanup: never report
                        # success unless the pending external-tool turn is stopped.
                        await session.abort()
                        aborted = True
                        result = _function_call_response(
                            request, [e.to_dict() for e in capture.events],
                            expected_session_id=session.session_id, think=self.think,
                            reasoning_effort=effort)
                        completed = True
                        return result
                    event = await session.send_and_wait(prompt_for(request), timeout=self.timeout_seconds)
                    if retried or len(usage_events) > 1:
                        raise InvalidResponseError("Copilot runtime retried or made multiple model calls")
                    if event is None or getattr(event.type, "value", event.type) != "assistant.message":
                        raise InvalidResponseError("Copilot returned no final assistant message")
                    data = event.data
                    if getattr(data, "tool_requests", None):
                        raise InvalidResponseError("Copilot returned unexpected tool requests")
                    usage = usage_events[0] if usage_events else None
                    finish = getattr(usage, "finish_reason", None)
                    if finish not in (None, "stop", "end_turn"):
                        raise InvalidResponseError("Copilot completion did not finish normally")
                    if getattr(usage, "content_filter_triggered", False):
                        raise InvalidResponseError("Copilot completion was filtered")
                    usage_model, message_model = getattr(usage, "model", None), getattr(data, "model", None)
                    if usage_model and message_model and usage_model != message_model:
                        raise InvalidResponseError("Copilot reported inconsistent response models")
                    model = usage_model or message_model
                    if not isinstance(model, str) or not model.strip():
                        raise InvalidResponseError("Copilot did not report the response model")
                    actual_effort = getattr(usage, "reasoning_effort", None)
                    if effort is not None and actual_effort is not None and effort != actual_effort:
                        raise InvalidResponseError("Copilot reported a different reasoning effort than requested")
                    if effort == "none":
                        _verify_disabled_reasoning(
                            data.content, actual_effort=actual_effort,
                            reasoning=(getattr(data, "reasoning_text", None)
                                       or getattr(data, "reasoning_blocks", None)),
                            reasoning_tokens=getattr(usage, "reasoning_tokens", None))
                    elif self.think is False:
                        reject_unexpected_reasoning(
                            data.content,
                            reasoning=(getattr(data, "reasoning_text", None)
                                       or getattr(data, "reasoning_blocks", None)
                                       or actual_effort),
                            reasoning_tokens=getattr(usage, "reasoning_tokens", None),
                        )
                    parser = parse_answer_only if self.output_format == "answer_only" else parse_choice
                    selected = parser(data.content, request)
                    raw = {"model": model, "finish_reason": finish,
                           "reasoning_effort": actual_effort, "content": data.content}
                    result = DecisionResponse(
                        request_id=request.request_id, selected_option_id=selected, resolved_model=model,
                        usage=token_usage(getattr(usage, "input_tokens", None),
                                          getattr(usage, "output_tokens", None),
                                          getattr(usage, "reasoning_tokens", None)),
                        raw_output=raw,
                    )
                    completed = True
                    return result
            except asyncio.CancelledError:
                cancelled = True
                raise
            except TimeoutError:
                raise ClientTimeoutError("Copilot operation timed out") from None
            except (DecisionClientError, ImportError):
                raise
            except Exception:
                # SDK errors can contain provider bodies, prompts, or credentials.
                raise ClientTransportError("Copilot SDK operation failed") from None
            finally:
                if session is not None:
                    # A second cancellation must not abandon runtime cleanup.
                    cleanup = asyncio.create_task(self._cleanup_session(
                        session, abort=not completed and not aborted))
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            cancelled = True
                    cleanup.result()
                    if cancelled:
                        raise asyncio.CancelledError()
                if audit_failed and not cancelled:
                    raise ClientTransportError("Copilot decision event audit callback failed") from None
                if completed and capture is not None:
                    replay_function_call(
                        request, result.raw_output, audit_events=[e.to_dict() for e in capture.events],
                        expected_session_id=session.session_id, think=self.think, reasoning_effort=effort)

    async def _cleanup_session(self, session, *, abort: bool) -> None:
        # Best effort and bounded; preserve the original response/error. Residual
        # session files are also removed when the whole client is closed.
        if abort:
            with suppress(Exception):
                await asyncio.wait_for(session.abort(), timeout=5)
        with suppress(Exception):
            await asyncio.wait_for(session.disconnect(), timeout=5)
        with suppress(Exception):
            await asyncio.wait_for(self._runtime.delete_session(session.session_id), timeout=5)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._runtime is not None:
                    try:
                        await asyncio.wait_for(self._runtime.stop(), timeout=self.timeout_seconds)
                    except TimeoutError:
                        raise ClientTimeoutError("Copilot shutdown timed out") from None
                    except Exception:
                        raise ClientTransportError("Copilot shutdown failed") from None
            finally:
                self._runtime = None
                if self._directory is not None:
                    self._directory.cleanup()
                    self._directory = None