"""Generative client tests with mocked HTTP/SDK only; no login or model calls."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.clients._llm_choice import (
    ANSWER_ONLY_SYSTEM_MESSAGE, SYSTEM_MESSAGE, parse_answer_only, parse_choice, prompt_for,
)
from src.clients.base import (
    ClientClosedError, ClientTimeoutError, ClientTransportError, DecisionOption,
    DecisionRequest, InvalidResponseError, ProviderHTTPError, UnsupportedRequestError,
)
from src.clients.github_copilot_client import GitHubCopilotClient, replay_function_call
from src.clients.registry import available_clients, create_client
from src.clients.vllm_client import VLLMClient


def request(request_id="PRIVATE_SOURCE-42"):
    return DecisionRequest(request_id, {"nodes": [0, 1], "edges": [[0, 1]]},
                           "Are nodes 0 and 1 adjacent?",
                           (DecisionOption("yes", "An edge exists"), DecisionOption("no")))


def completion(content='{"choice":"yes"}', *, thinking=False):
    return {"model": "served-model", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content,
        "reasoning_content": 'PRIVATE_REASONING {"choice":"no"}' if thinking else None,
    }}], "usage": {"prompt_tokens": 20, "completion_tokens": 12,
                   "completion_tokens_details": {"reasoning_tokens": 8 if thinking else 0}}}


def model_info(reasoning=True):
    return NS(id="test-gpt", supported_reasoning_efforts=["low", "medium", "high"] if reasoning else [],
              default_reasoning_effort="medium" if reasoning else None,
              capabilities=NS(supports=NS(reasoning_effort=reasoning)))


ANSWER_ONLY_CASES = [
    ("yes", "yes"), (" NO\n", "no"), ("YeS", "yes"),
    ("<think>no is not right</think>yes", "yes"),
] + [(text, None) for text in (
    None, "", "yes.", '"yes"', '{"choice":"yes"}', "yes no", "The answer is yes",
    "```\nyes\n```", "yes\nno", "maybe", "ｙｅｓ", "<think>yes",
    "<think>yes</think>", "<think><think>x</think>yes",
)]


class ChoiceProtocolTests(unittest.TestCase):
    def test_parsers_accept_only_exact_answers_and_safe_wrappers(self):
        json_cases = [(text, "yes") for text in (
            ' {"choice":"yes"} ', '```json\n{"choice":"yes"}\n```',
            '<think>{"choice":"no"}</think>\n{"choice":"yes"}',
            '<think></think>\n```json\n{"choice":"yes"}\n```',
        )] + [(text, None) for text in (
            None, "", "yes", "YES", "The answer is yes.", '[]', '{"choice":"YES"}',
            '{"choice":"maybe"}', '{"choice":true}', '{"choice":["yes"]}',
            '{"choice":"yes","explanation":"x"}', '{"choice":"no","choice":"yes"}',
            '{"choice":"yes"} {"choice":"no"}', '<think>{"choice":"yes"}',
            '<think><think>x</think>{"choice":"yes"}',
            '<think>{"choice":"yes"}</think>', '{"choice":"yes"}\nMore text',
        )]
        for parser, cases in ((parse_answer_only, ANSWER_ONLY_CASES), (parse_choice, json_cases)):
            for text, expected in cases:
                with self.subTest(parser=parser.__name__, text=text):
                    if expected is None:
                        with self.assertRaises(InvalidResponseError):
                            parser(text, request())
                    else:
                        self.assertEqual(parser(text, request()), expected)
            custom = replace(request(), options=(DecisionOption("node_7"), DecisionOption("node_91")))
            for option in ("node_91", "NODE_91"):
                text = option if parser is parse_answer_only else json.dumps({"choice": option})
                with self.subTest(parser=parser.__name__, option=option):
                    if option == "node_91":
                        self.assertEqual(parser(text, custom), option)
                    else:
                        with self.assertRaises(InvalidResponseError):
                            parser(text, custom)

    def test_prompt_has_only_model_visible_fields(self):
        content = prompt_for(request())
        value = json.loads(content)
        self.assertEqual(set(value), {"state", "question", "options"})
        self.assertEqual(value["state"], request().state)
        self.assertEqual(value["options"], [{"id": "yes", "description": "An edge exists"},
                                             {"id": "no", "description": None}])
        self.assertNotIn("PRIVATE", content)
        self.assertNotIn("choice-v1", content)

class VLLMTests(unittest.IsolatedAsyncioTestCase):
    def response_client(self, body, **kwargs):
        return VLLMClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=json.dumps(body))), **kwargs)

    async def test_answer_only_wire_and_diagnostics(self):
        for text, code in ((" YES\n", None), ("<think></think>no", None),
                           ('{"choice":"yes"}', "invalid_choice"),
                           ("yes because ...", "invalid_choice"),
                           ("<think>reasoning</think>yes", "unexpected_reasoning")):
            def handler(wire):
                body = json.loads(wire.content)
                self.assertEqual(body["messages"][0]["content"], ANSWER_ONLY_SYSTEM_MESSAGE)
                self.assertEqual(body["messages"][1]["content"], prompt_for(request()))
                self.assertNotIn("response_format", body)
                self.assertNotIn("output_format", body)
                return httpx.Response(200, json=completion(text))

            async with VLLMClient(output_format="answer_only", think=False,
                                  transport=httpx.MockTransport(handler)) as client:
                if code is None:
                    result = await client.predict(request())
                    self.assertEqual(result.selected_option_id, "no" if text.endswith("no") else "yes")
                    self.assertEqual(result.usage.output_tokens, 12)
                else:
                    with self.assertRaises(InvalidResponseError) as caught:
                        await client.predict(request())
                    self.assertEqual(caught.exception.diagnostic_code, code)
                    self.assertEqual(caught.exception.finish_reason, "stop")
                    self.assertEqual(caught.exception.usage.output_tokens, 12)
        for think in (True, False):
            async with self.response_client(completion("yes", thinking=True),
                                            output_format="answer_only", think=think) as client:
                if think:
                    self.assertEqual((await client.predict(request())).selected_option_id, "yes")
                else:
                    with self.assertRaises(InvalidResponseError):
                        await client.predict(request())

    async def test_think_and_no_think_wire_and_usage(self):
        for think in (True, False, None):
            bodies = []

            def handler(wire):
                self.assertEqual(str(wire.url), "http://127.0.0.1:8000/v1/chat/completions")
                self.assertEqual(wire.headers["authorization"], "Bearer FAKE_LOCAL_KEY")
                body = json.loads(wire.content)
                bodies.append(body)
                self.assertEqual(body["model"], "local-model")
                self.assertEqual(body["max_tokens"], 1024)
                self.assertEqual(body["seed"], 7)
                self.assertEqual(body["temperature"], 0.1)
                self.assertEqual(body["top_p"], 0.9)
                self.assertFalse(body["stream"])
                self.assertEqual(len(body["messages"]), 2)
                self.assertEqual(body["messages"][0]["content"], SYSTEM_MESSAGE)
                self.assertEqual(json.loads(body["messages"][1]["content"]), json.loads(prompt_for(request())))
                if think is None:
                    self.assertNotIn("chat_template_kwargs", body)
                else:
                    self.assertIs(body["chat_template_kwargs"]["enable_thinking"], think)
                self.assertNotIn("extra_body", body)
                self.assertNotIn("tools", body)
                self.assertNotIn("PRIVATE_SOURCE", json.dumps(body))
                return httpx.Response(200, json=completion(thinking=think is not False))

            with self.subTest(think=think):
                async with create_client("vllm", model="local-model", think=think, max_tokens=1024,
                                         seed=7, temperature=0.1, top_p=0.9, api_key="FAKE_LOCAL_KEY",
                                         transport=httpx.MockTransport(handler)) as client:
                    result = await client.predict(request())
                    await client.predict(request("different-id"))
                    self.assertTrue(client.capabilities.supports_thinking_control)
                    self.assertFalse(client.capabilities.supports_probabilities)
                self.assertEqual(len(bodies), 2)
                self.assertEqual(bodies[0], bodies[1])
                self.assertEqual(result.request_id, request().request_id)
                self.assertEqual(result.selected_option_id, "yes")
                self.assertEqual(result.resolved_model, "served-model")
                self.assertEqual((result.usage.input_tokens, result.usage.output_tokens,
                                  result.usage.reasoning_tokens), (20, 12, 0 if think is False else 8))
                self.assertIsNone(result.probabilities)
                self.assertIsNone(result.probability_kind)
                self.assertIsNone(result.confidence)

    async def test_optional_auth_and_unknown_usage(self):
        for key in (None, "FAKE_ENV_KEY"):
            def handler(wire):
                if key:
                    self.assertEqual(wire.headers["authorization"], f"Bearer {key}")
                else:
                    self.assertNotIn("authorization", wire.headers)
                body = completion()
                body.pop("usage")
                return httpx.Response(200, json=body)

            with patch.dict(os.environ, {"VLLM_API_KEY": key} if key else {}, clear=True):
                async with VLLMClient(transport=httpx.MockTransport(handler)) as client:
                    result = await client.predict(request())
                    self.assertIsNone(result.usage.input_tokens)
                    self.assertIsNone(result.usage.reasoning_tokens)

    async def test_no_think_rejects_observed_reasoning_but_allows_empty_block(self):
        for body in (completion(thinking=True),
                     completion('<think>PRIVATE_REASONING</think>{"choice":"yes"}'),
                     completion() | {"usage": {"completion_tokens_details": {"reasoning_tokens": 1}}}):
            async with self.response_client(body, think=False) as client:
                with self.assertRaisesRegex(InvalidResponseError, "no-think"):
                    await client.predict(request())
        async with self.response_client(
                completion('<think></think>{"choice":"yes"}'), think=False) as client:
            self.assertEqual((await client.predict(request())).selected_option_id, "yes")

    async def test_http_errors_no_retry_no_redirect_no_body_leak(self):
        for status in (302, 400, 401, 429, 503):
            calls = []

            def handler(wire):
                calls.append(wire)
                return httpx.Response(status, text="PRIVATE_SERVER_BODY",
                                      headers={"retry-after": "2", "location": "http://elsewhere/"})

            async with VLLMClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ProviderHTTPError) as caught:
                    await client.predict(request())
            self.assertEqual(len(calls), 1)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(caught.exception.retry_after, "2")
            self.assertNotIn("PRIVATE", str(caught.exception))

    async def test_transport_timeout_and_cancellation(self):
        for error, expected in ((httpx.ReadTimeout, ClientTimeoutError),
                                (httpx.ConnectError, ClientTransportError),
                                (asyncio.CancelledError, asyncio.CancelledError)):
            def handler(wire):
                raise error("PRIVATE_TRANSPORT")

            async with VLLMClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(expected) as caught:
                    await client.predict(request())
                if expected is not asyncio.CancelledError:
                    self.assertNotIn("PRIVATE", str(caught.exception))

        async def hanging(wire):
            await asyncio.Future()

        async with VLLMClient(timeout_seconds=0.01, transport=httpx.MockTransport(hanging)) as client:
            with self.assertRaises(ClientTimeoutError):
                await client.predict(request())

    async def test_bad_responses_are_typed_failures(self):
        bodies = [None, [], {}, completion() | {"model": ""}, completion() | {"choices": []},
                  completion() | {"usage": []}, completion() | {"usage": {"prompt_tokens": -1}},
                  completion() | {"usage": {"completion_tokens_details": []}},
                  completion() | {"usage": {"completion_tokens": True}},
                  completion() | {"usage": {"completion_tokens_details": {"reasoning_tokens": "8"}}}]
        for finish in ("length", "tool_calls", "content_filter", None):
            body = completion()
            body["choices"][0]["finish_reason"] = finish
            bodies.append(body)
        for change in ({"content": None}, {"tool_calls": [{}]}, {"refusal": "refused"},
                       {"role": "user"}, {"content": '<think>{"choice":"yes"}'}):
            body = completion()
            body["choices"][0]["message"].update(change)
            bodies.append(body)
        for body in bodies:
            with self.subTest(body=body):
                async with self.response_client(body) as client:
                    with self.assertRaises(InvalidResponseError):
                        await client.predict(request())
        async with VLLMClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text="PRIVATE_NOT_JSON")
        )) as client:
            with self.assertRaises(InvalidResponseError) as caught:
                await client.predict(request())
            self.assertNotIn("PRIVATE", str(caught.exception))

    async def test_closed_and_protocol_checks_precede_http(self):
        def fail(_):
            raise AssertionError("No HTTP expected")

        client = VLLMClient(transport=httpx.MockTransport(fail))
        with self.assertRaises(UnsupportedRequestError):
            await client.predict(replace(request(), protocol_version="future"))
        await client.aclose()
        await client.aclose()
        with self.assertRaises(ClientClosedError):
            await client.predict(request())

    def test_invalid_settings(self):
        for kwargs in ({"output_format": "unknown"}, {"output_format": []}, {"output_format": None},
                       {"think": "false"}, {"max_tokens": True}, {"max_tokens": 0},
                       {"model": ""}, {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
                       {"temperature": float("inf")}, {"temperature": -1}, {"top_p": 0},
                       {"top_p": 2}, {"seed": True}, {"seed": -1}, {"api_key": "bad\nkey"},
                       {"base_url": "http://user:pass@localhost:8000/v1"},
                       {"base_url": "http://localhost:8000/v1?token=secret"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VLLMClient(**kwargs)


def mock_event_dict(event, event_id):
    def wire(value):
        if isinstance(value, NS):
            return {"".join(part if index == 0 else part.title() for index, part in enumerate(key.split("_"))):
                    wire(item) for key, item in vars(value).items()}
        if isinstance(value, list):
            return [wire(item) for item in value]
        return value

    return {"type": event.type, "id": event_id, "data": wire(event.data)}


class FakeSession:
    def __init__(self, runtime, options):
        self.runtime, self.options = runtime, options
        self.session_id = f"session-{len(runtime.sessions)}"
        self.abort = AsyncMock()
        self.disconnect = AsyncMock()
        self.set_model = AsyncMock()
        self.rpc = NS(model=NS(get_current=AsyncMock(
            return_value=NS(model_id=options["model"], reasoning_effort="none"))))

    async def send_and_wait(self, prompt, *, timeout):
        self.runtime.prompts.append(prompt)
        for event in self.runtime.events:
            self.options["on_event"](event)
        if self.runtime.error:
            raise self.runtime.error
        return self.runtime.answer

    async def send(self, prompt):
        self.runtime.prompts.append(prompt)
        for index, event in enumerate(self.runtime.events):
            if not hasattr(event, "to_dict"):
                event.to_dict = lambda event=event, index=index: mock_event_dict(event, f"event-{index}")
            self.options["on_event"](event)
        if self.runtime.error:
            raise self.runtime.error


class FakeRuntime:
    def __init__(self):
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.get_auth_status = AsyncMock(return_value=NS(isAuthenticated=True))
        self.list_models = AsyncMock(return_value=[model_info()])
        self.delete_session = AsyncMock()
        self.create_session = AsyncMock(side_effect=self.new_session)
        self.sessions, self.prompts = [], []
        self.error = None
        self.answer = NS(type="assistant.message", data=NS(content='{"choice":"yes"}',
                                                         model="resolved-gpt", tool_requests=[]))
        self.events = [NS(type="assistant.usage", data=NS(model="resolved-gpt", input_tokens=100,
                          output_tokens=20, reasoning_tokens=14, finish_reason="stop",
                          reasoning_effort="medium"))]

    async def new_session(self, **options):
        session = FakeSession(self, options)
        self.sessions.append(session)
        return session


class CopilotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.factory_options = []

        def factory(**kwargs):
            self.factory_options.append(kwargs)
            return self.runtime

        self.patch = patch("src.clients.github_copilot_client._sdk_types",
                           return_value=(factory, NS, NS, lambda: NS(kind="denied")))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def client(self, **kwargs):
        return GitHubCopilotClient(**({"model": "test-gpt", "github_token": "FAKE_GITHUB_KEY"} | kwargs))

    async def predict(self, **kwargs):
        async with self.client(**kwargs) as client:
            return await client.predict(request())

    def assert_session_closed(self, aborted=False):
        session = self.runtime.sessions[-1]
        if aborted:
            session.abort.assert_awaited_once()
        else:
            session.abort.assert_not_called()
        session.disconnect.assert_awaited_once()
        self.runtime.delete_session.assert_any_await(session.session_id)

    def function_events(self, arguments=None):
        arguments = {"choice": "yes"} if arguments is None else arguments
        tool = NS(name="submit_decision", tool_call_id="call-1", arguments=arguments)
        message = NS(type="assistant.message", data=NS(
            content="", model="resolved-gpt", tool_requests=[tool]))
        call = NS(type="external_tool.requested", data=NS(
            tool_name="submit_decision", tool_call_id="call-1",
            request_id="external-1", session_id=f"session-{len(self.runtime.sessions)}",
            arguments=arguments))
        usage = NS(type="assistant.usage", data=NS(
            model="resolved-gpt", input_tokens=100, output_tokens=20, reasoning_tokens=14,
            finish_reason="tool_calls", reasoning_effort=None))
        events = [usage, message, call]
        for i, event in enumerate(events):
            event.to_dict = lambda event=event, i=i: mock_event_dict(event, str(i))
        return events

    async def predict_function(self, **kwargs):
        with patch("src.clients.github_copilot_client._decision_tool") as tool:
            tool.return_value = NS(name="submit_decision", handler=None)
            return await self.predict(output_format="function_call", **kwargs)

    def no_think_events(self):
        info = model_info()
        info.supported_reasoning_efforts.insert(0, "none")
        self.runtime.list_models.return_value = [info]
        events = self.function_events()
        events[0].data.model = events[1].data.model = "test-gpt"
        events[0].data.reasoning_effort = "none"
        events[0].data.reasoning_tokens = 0
        return events

    async def test_no_think_uses_public_setter_and_snapshot_before_each_send(self):
        original_send = FakeSession.send

        async def checked_send(session, prompt):
            session.set_model.assert_awaited_once_with(
                "test-gpt", reasoning_effort="none",
                model_capabilities=session.options["model_capabilities"])
            session.rpc.model.get_current.assert_awaited_once()
            return await original_send(session, prompt)

        for settings in ({"think": False}, {"reasoning_effort": "none"},
                         {"think": False, "reasoning_effort": "none"}):
            with self.subTest(settings=settings):
                self.runtime.events = self.no_think_events()
                with patch.object(FakeSession, "send", new=checked_send):
                    response = await self.predict_function(**settings)
                opts = self.runtime.sessions[-1].options
                self.assertIsNone(opts["reasoning_effort"])
                self.assertEqual(opts["model_capabilities"].limits.max_output_tokens, 4096)
                self.assertEqual(response.usage.reasoning_tokens, 0)
                self.assertEqual(response.raw_output["reasoning_effort"], "none")
                self.assertEqual(replay_function_call(
                    request(), response.raw_output, audit_events=response.raw_output["events"],
                    think=False, reasoning_effort="none"), response)
                self.assert_session_closed(aborted=True)

    async def test_no_think_missing_advertisement_rejects_before_session_or_send(self):
        for settings in ({"think": False}, {"reasoning_effort": "none"},
                         {"think": False, "reasoning_effort": "none"}):
            self.runtime.list_models.return_value = [NS(
                **{**vars(model_info()), "id": "gpt-6-astra"})]
            with self.assertRaises(UnsupportedRequestError):
                await self.predict_function(model="gpt-6-astra", **settings)
        self.runtime.create_session.assert_not_called()
        self.assertEqual(self.runtime.prompts, [])

    async def test_no_think_snapshot_mismatch_or_setter_failure_never_sends(self):
        original = self.runtime.new_session
        for case in ("wrong_model", "wrong_effort", "missing_effort", "setter_failure", "snapshot_failure"):
            self.runtime.events = self.no_think_events()

            async def configured_session(**options):
                session = await original(**options)
                if case == "setter_failure":
                    session.set_model.side_effect = RuntimeError("PRIVATE")
                elif case == "snapshot_failure":
                    session.rpc.model.get_current.side_effect = RuntimeError("PRIVATE")
                else:
                    session.rpc.model.get_current.return_value = NS(
                        model_id="other" if case == "wrong_model" else "test-gpt",
                        reasoning_effort="medium" if case == "wrong_effort" else None)
                return session

            self.runtime.create_session.side_effect = configured_session
            with self.subTest(case=case), self.assertRaises(ClientTransportError) as caught:
                await self.predict_function(think=False)
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertEqual(self.runtime.prompts, [])
            self.assert_session_closed(aborted=True)

    async def test_no_think_requires_reported_none_and_integer_zero_live_and_replay(self):
        self.runtime.events = self.no_think_events()
        original = (await self.predict_function(think=False)).raw_output
        for field, value in (("reasoning_effort", None), ("reasoning_effort", "low"),
                             ("reasoning_tokens", None), ("reasoning_tokens", 1),
                             ("reasoning_tokens", False), ("reasoning_tokens", 0.0)):
            with self.subTest(field=field, value=value):
                self.runtime.events = self.no_think_events()
                setattr(self.runtime.events[0].data, field, value)
                with self.assertRaises(InvalidResponseError):
                    await self.predict_function(think=False)
                raw = deepcopy(original)
                wire_field = "reasoningEffort" if field == "reasoning_effort" else "reasoningTokens"
                raw["events"][0]["data"][wire_field] = value
                if field == "reasoning_effort":
                    raw["reasoning_effort"] = value
                with self.assertRaises(InvalidResponseError):
                    replay_function_call(request(), raw, think=False)
        self.runtime.events = self.no_think_events()[1:]
        with self.assertRaises(InvalidResponseError):
            await self.predict_function(think=False)

    async def test_no_think_rejects_reasoning_text_and_events_even_with_zero_usage(self):
        for field in ("reasoning_text", "reasoning_blocks", "content", "event"):
            self.runtime.events = self.no_think_events()
            if field == "event":
                self.runtime.events.insert(1, NS(type="assistant.reasoning", data=NS(content="reasoning")))
            else:
                setattr(self.runtime.events[1].data, field, (
                    "<think>reasoning</think>" if field == "content" else "reasoning"))
            with self.subTest(field=field), self.assertRaises(InvalidResponseError):
                await self.predict_function(think=False)

    async def test_no_think_model_change_is_audited_and_replayed(self):
        self.runtime.events = self.no_think_events()
        original = self.runtime.new_session

        async def with_setting_event(**options):
            session = await original(**options)
            event = NS(type="session.model_change", data=NS(
                new_model="test-gpt", reasoning_effort="none"))
            event.to_dict = lambda: mock_event_dict(event, "model-change")
            session.set_model.side_effect = lambda *_args, **_kwargs: options["on_event"](event)
            return session

        self.runtime.create_session.side_effect = with_setting_event
        records = []

        def audit(req, event):
            records.append(event)

        response = await self.predict_function(think=False, on_decision_event=audit)
        self.assertEqual(records[0]["type"], "session.model_change")
        self.assertEqual(replay_function_call(
            request(), response.raw_output, audit_events=records, think=False), response)
        records[0]["data"]["reasoningEffort"] = "low"
        with self.assertRaises(InvalidResponseError):
            replay_function_call(request(), response.raw_output, audit_events=records, think=False)

    async def test_no_think_setting_audit_failure_aborts_before_send(self):
        self.runtime.events = self.no_think_events()
        original = self.runtime.new_session

        async def with_setting_event(**options):
            session = await original(**options)
            event = NS(type="session.model_change", data=NS())
            event.to_dict = lambda: mock_event_dict(event, "setting")
            session.set_model.side_effect = lambda *_args, **_kwargs: options["on_event"](event)
            return session

        def failed_audit(*_):
            raise OSError("PRIVATE")

        self.runtime.create_session.side_effect = with_setting_event
        with self.assertRaisesRegex(ClientTransportError, "audit callback failed"):
            await self.predict_function(think=False, on_decision_event=failed_audit)
        self.assertEqual(self.runtime.prompts, [])
        self.assert_session_closed(aborted=True)

    async def test_no_think_text_modes_and_unspecified_defaults_are_preserved(self):
        for output_format, content in (("json", '{"choice":"yes"}'), ("answer_only", "yes")):
            self.runtime.events = self.no_think_events()[:1]
            self.runtime.events[0].data.finish_reason = "stop"
            self.runtime.answer.data.model = "test-gpt"
            self.runtime.answer.data.content = content
            response = await self.predict(think=False, output_format=output_format)
            self.assertEqual(response.selected_option_id, "yes")
            self.assertEqual(response.usage.reasoning_tokens, 0)
            self.runtime.sessions[-1].set_model.assert_awaited_once()
        self.runtime.events = self.no_think_events()
        await self.predict_function()
        self.runtime.sessions[-1].set_model.assert_not_called()
        self.runtime.sessions[-1].rpc.model.get_current.assert_not_called()

    async def test_decision_audit_is_request_bound_before_invalid_argument_validation(self):
        records = []
        self.runtime.events = self.function_events({"choice": "invalid"})
        external = self.runtime.events[-1]

        def audit(req, event):
            records.append((req, event))
            # A consumer's mutation must not repair the actual invalid call.
            if event["type"] == "external_tool.requested":
                event["data"]["arguments"]["choice"] = "yes"

        from src.clients.github_copilot_client import _tool_choice

        def validate(arguments, req):
            self.assertEqual(records[-1][1]["type"], "external_tool.requested")
            self.assertEqual(records[-1][0], req)
            return _tool_choice(arguments, req)

        with patch("src.clients.github_copilot_client._tool_choice", side_effect=validate):
            with self.assertRaises(InvalidResponseError):
                await self.predict_function(on_decision_event=audit)
        self.assertEqual(external.data.arguments, {"choice": "invalid"})
        self.assertEqual(len(records), 3)
        self.assert_session_closed(aborted=True)

    async def test_decision_audit_success_and_fresh_request_bindings(self):
        records = []

        def audit(req, event):
            records.append((req.request_id, event))

        with patch("src.clients.github_copilot_client._decision_tool", return_value=NS()):
            async with self.client(output_format="function_call", on_decision_event=audit) as client:
                for request_id in ("one", "two"):
                    self.runtime.events = self.function_events()
                    result = await client.predict(request(request_id))
                    self.assertEqual(records[-3:], [
                        (request_id, event.to_dict()) for event in self.runtime.events])
                    self.assertEqual(result.raw_output["events"],
                                     [event for _, event in records[-3:]])
        self.assertEqual([request_id for request_id, _ in records], ["one"] * 3 + ["two"] * 3)

    async def test_decision_audit_persistence_or_serialization_failure_is_not_swallowed(self):
        def broken_audit(req, event):
            raise OSError("PRIVATE_STORAGE_ERROR")

        def broken_serialization():
            raise ValueError("PRIVATE")

        for serialization_failure in (False, True):
            self.runtime.events = self.function_events()
            if serialization_failure:
                self.runtime.events[-1].to_dict = broken_serialization
            with self.assertRaisesRegex(ClientTransportError, "audit callback failed") as caught:
                await self.predict_function(
                    on_decision_event=(lambda *_: None) if serialization_failure else broken_audit)
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assert_session_closed(aborted=True)

    async def test_decision_audit_failure_during_cleanup_prevents_success(self):
        self.runtime.events = self.function_events()
        original = self.runtime.new_session

        async def with_cleanup_event(**options):
            session = await original(**options)
            idle = NS(type="session.idle", data=NS(), to_dict=lambda: {
                "type": "session.idle", "id": "idle-1", "data": {}})
            session.disconnect.side_effect = lambda: options["on_event"](idle)
            return session

        def audit(req, event):
            if event["type"] == "session.idle":
                raise OSError("PRIVATE_STORAGE_ERROR")

        self.runtime.create_session.side_effect = with_cleanup_event
        with self.assertRaisesRegex(ClientTransportError, "audit callback failed"):
            await self.predict_function(on_decision_event=audit)
        self.assert_session_closed(aborted=True)

    async def test_decision_audit_rejects_async_and_non_none_callbacks(self):
        async def async_audit(*_):
            return None

        class AsyncCallable:
            async def __call__(self, *_):
                return None

        for callback in (False, "invalid", async_audit, AsyncCallable()):
            with self.assertRaises(ValueError):
                self.client(on_decision_event=callback)
        for callback in (lambda *_: True, lambda *_: async_audit()):
            self.runtime.events = self.function_events()
            with self.assertRaisesRegex(ClientTransportError, "audit callback failed"):
                await self.predict_function(on_decision_event=callback)
            self.assert_session_closed(aborted=True)

    async def test_adapter_can_load_under_frozen_package_alias_with_original_base_types(self):
        import src.clients.github_copilot_client as original

        spec = importlib.util.spec_from_file_location("src.clients.gpt54_tool_frozen", original.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIs(module.DecisionResponse, original.DecisionResponse)
        self.assertIs(module.InvalidResponseError, InvalidResponseError)
        self.runtime.events = self.function_events()
        records = []

        def audit(req, event):
            records.append(event)

        with patch.object(module, "_sdk_types", return_value=(
                lambda **_: self.runtime, NS, NS, lambda: NS(kind="denied"))), patch.object(
                module, "_decision_tool", return_value=NS()):
            async with module.GitHubCopilotClient(
                    model="test-gpt", output_format="function_call", on_decision_event=audit) as client:
                response = await client.predict(request())
        self.assertIsInstance(response, original.DecisionResponse)
        self.assertEqual(records, response.raw_output["events"])

    async def test_offline_replay_matches_response_and_full_audit_without_sdk(self):
        self.runtime.events = self.function_events()
        response = await self.predict_function()
        raw = response.raw_output
        audit = deepcopy(raw["events"])
        audit.extend([
            {"type": "external_tool.completed", "id": "completed", "data": {"requestId": "external-1"}},
            {"type": "assistant.turn_end", "id": "end", "data": {"turnId": "0"}},
            {"type": "session.idle", "id": "idle", "data": {"aborted": True}},
        ])
        with patch("src.clients.github_copilot_client._sdk_types", side_effect=AssertionError("No SDK")):
            replayed = replay_function_call(request(), raw, audit_events=audit, expected_session_id="session-0")
        self.assertEqual(replayed, response)
        replayed.raw_output["events"][0]["data"]["inputTokens"] = 999
        self.assertEqual(raw["events"][0]["data"]["inputTokens"], 100)
        self.assertEqual(audit[0]["data"]["inputTokens"], 100)
        with self.assertRaises(InvalidResponseError):
            replay_function_call(request(), raw, audit_events=audit, expected_session_id="wrong")

    async def test_offline_replay_rejects_raw_and_audit_tampering(self):
        self.runtime.events = self.function_events()
        response = await self.predict_function()
        original = response.raw_output
        for case in (
            "raw_model", "raw_finish", "raw_reasoning", "raw_protocol", "synthetic_content",
            "choice_outside_options", "choice_changed_to_other_valid_option", "usage_negative",
            "audit_usage_changed", "duplicate_call", "raw_event_changed", "audit_event_missing",
            "audit_retry", "audit_second_turn", "audit_second_usage", "audit_second_execution",
            "audit_wrong_tool", "audit_wrong_completion", "audit_successful_execution",
            "raw_api_call_mismatch", "raw_missing_id", "raw_unknown_event", "raw_reordered",
        ):
            with self.subTest(case=case):
                raw, audit = deepcopy(original), deepcopy(original["events"])
                if case.startswith("raw_") and case[4:] in ("model", "finish", "reasoning", "protocol"):
                    raw[{"raw_model": "model", "raw_finish": "finish_reason",
                         "raw_reasoning": "reasoning_effort", "raw_protocol": "protocol"}[case]] = "tampered"
                elif case == "synthetic_content":
                    raw["content"] = '{"choice":"yes"}'
                elif case.startswith("choice_"):
                    value = "not-an-option" if case == "choice_outside_options" else "no"
                    raw["events"][1]["data"]["toolRequests"][0]["arguments"] = {"choice": value}
                    raw["events"][2]["data"]["arguments"] = {"choice": value}
                elif case == "usage_negative":
                    raw["events"][0]["data"]["inputTokens"] = -1
                elif case == "audit_usage_changed":
                    audit[0]["data"]["inputTokens"] += 1
                elif case == "duplicate_call":
                    raw["events"][1]["data"]["toolRequests"] *= 2
                elif case == "raw_event_changed":
                    raw["events"][1]["id"] = "different"
                elif case == "audit_event_missing":
                    audit.pop()
                elif case == "audit_retry":
                    audit.append({"type": "assistant.turn_retry", "id": "retry", "data": {}})
                elif case == "audit_second_turn":
                    audit[0:0] = [
                        {"type": "assistant.turn_start", "id": str(index), "data": {"turnId": str(index)}}
                        for index in (10, 11)]
                elif case == "audit_second_usage":
                    audit.append({**deepcopy(audit[0]), "id": "usage-2"})
                elif case in ("audit_second_execution", "audit_wrong_tool"):
                    for index in range(2 if case == "audit_second_execution" else 1):
                        audit.insert(2, {"type": "tool.execution_start", "id": f"start-{index}",
                                        "data": {"toolCallId": "call-1", "arguments": {"choice": "yes"},
                                                 "toolName": "bash" if case == "audit_wrong_tool"
                                                 else "submit_decision"}})
                elif case == "audit_wrong_completion":
                    audit.append({"type": "external_tool.completed", "id": "completed",
                                  "data": {"requestId": "wrong"}})
                elif case == "audit_successful_execution":
                    audit.append({"type": "tool.execution_complete", "id": "executed",
                                  "data": {"toolCallId": "call-1", "success": True}})
                elif case == "raw_api_call_mismatch":
                    raw["events"][0]["data"]["apiCallId"] = "first"
                    raw["events"][1]["data"]["apiCallId"] = "second"
                elif case == "raw_missing_id":
                    del raw["events"][1]["id"]
                elif case == "raw_unknown_event":
                    raw["events"][1]["type"] = "unknown"
                elif case == "raw_reordered":
                    raw["events"][1:] = reversed(raw["events"][1:])
                with self.assertRaises(InvalidResponseError):
                    replay_function_call(request(), raw, audit_events=audit)

    async def test_late_observable_retry_during_cleanup_prevents_live_success(self):
        self.runtime.events = self.function_events()
        original = self.runtime.new_session

        async def with_late_retry(**options):
            session = await original(**options)
            retry = NS(type="assistant.turn_retry", data=NS(), to_dict=lambda: {
                "type": "assistant.turn_retry", "id": "retry", "data": {}})
            session.disconnect.side_effect = lambda: options["on_event"](retry)
            return session

        self.runtime.create_session.side_effect = with_late_retry
        with self.assertRaisesRegex(InvalidResponseError, "retried"):
            await self.predict_function()
        self.assert_session_closed(aborted=True)

    async def test_function_call_isolated_single_send_raw_events_and_defaults(self):
        for _ in range(2):
            self.runtime.events = self.function_events()
            with patch.object(FakeSession, "send_and_wait", side_effect=AssertionError("No idle wait")):
                result = await self.predict_function()
            opts = self.runtime.sessions[-1].options
            self.assertEqual(opts["available_tools"], ["submit_decision"])
            self.assertEqual(len(opts["tools"]), 1)
            self.assertIsNone(opts["tools"][0].handler)
            self.assertEqual(opts["tool_search"], {"enabled": False})
            self.assertFalse(opts["enable_config_discovery"])
            self.assertFalse(opts["enable_skills"])
            self.assertFalse(opts["enable_file_hooks"])
            self.assertEqual(opts["mcp_servers"], {})
            self.assertEqual(opts["on_permission_request"]().kind, "denied")
            self.assertEqual(opts["model_capabilities"].limits.max_output_tokens, 4096)
            self.assertIsNone(opts["reasoning_effort"])
            self.assertNotIn("tool_choice", opts)
            self.assertNotIn("strict", opts)
            self.assertEqual(result.selected_option_id, "yes")
            self.assertEqual(result.resolved_model, "resolved-gpt")
            self.assertEqual(result.usage.input_tokens, 100)
            self.assertEqual(result.raw_output["finish_reason"], "tool_calls")
            self.assertEqual(result.raw_output["protocol"], "choice-function-call-v1")
            self.assertNotIn("content", result.raw_output)
            self.assertEqual(result.raw_output["events"], [e.to_dict() for e in self.runtime.events])
            json.dumps(result.raw_output, allow_nan=False)
            self.assert_session_closed(aborted=True)
        self.assertEqual(len(self.runtime.sessions), 2)
        self.assertEqual(self.runtime.prompts, [prompt_for(request())] * 2)

    async def test_function_call_rejects_invalid_arguments_without_text_fallback(self):
        for arguments in (
            {}, [], {"choice": "YES"}, {"choice": 1}, {"choice": "missing"},
            {"choice": "yes", "extra": True}, '{"choice":"yes","choice":"no"}',
            '```json\n{"choice":"yes"}\n```', "yes", '{"choice":',
        ):
            with self.subTest(arguments=arguments):
                self.runtime.events = self.function_events(arguments)
                self.runtime.events[1].data.content = '{"choice":"yes"}'
                with self.assertRaises(InvalidResponseError):
                    await self.predict_function()
                self.assert_session_closed(aborted=True)

    async def test_function_call_rejects_missing_duplicate_wrong_and_retried_calls(self):
        for case in ("missing", "duplicate_message_call", "duplicate_external", "wrong_tool",
                     "wrong_id", "wrong_session", "wrong_arguments", "retry", "multi_usage",
                     "multi_turn", "multi_message", "error", "idle_without_call"):
            with self.subTest(case=case):
                events = self.function_events()
                usage, message, call = events
                if case == "missing":
                    message.data.tool_requests = []
                    message.data.content = '{"choice":"yes"}'
                    events.remove(call)
                elif case == "duplicate_message_call":
                    message.data.tool_requests *= 2
                elif case == "duplicate_external":
                    events.append(call)
                elif case == "wrong_tool":
                    call.data.tool_name = "bash"
                elif case == "wrong_id":
                    call.data.tool_call_id = "other"
                elif case == "wrong_session":
                    call.data.session_id = "other"
                elif case == "wrong_arguments":
                    call.data.arguments = {"choice": "no"}
                elif case == "retry":
                    events.append(NS(type="assistant.turn_retry", data=NS()))
                elif case == "multi_usage":
                    events.append(usage)
                elif case == "multi_turn":
                    events.extend([NS(type="assistant.turn_start", data=NS())] * 2)
                elif case == "multi_message":
                    events.append(message)
                elif case == "error":
                    events.append(NS(type="session.error", data=NS(message="PRIVATE")))
                elif case == "idle_without_call":
                    events = [usage, NS(type="session.idle", data=NS())]
                self.runtime.events = events
                with self.assertRaises(ClientTransportError if case == "error" else InvalidResponseError):
                    await self.predict_function()
                self.assert_session_closed(aborted=True)

    async def test_function_call_metadata_checks_and_missing_usage(self):
        for field, value in (("finish_reason", "length"), ("content_filter_triggered", True),
                             ("model", "other"), ("input_tokens", -1),
                             ("reasoning_effort", "high")):
            with self.subTest(field=field):
                self.runtime.events = self.function_events()
                with patch.object(self.runtime.events[0].data, field, value, create=True):
                    with self.assertRaises(InvalidResponseError):
                        await self.predict_function(reasoning_effort="low")
        self.runtime.events = self.function_events('{"choice":"yes"}')[1:]
        result = await self.predict_function()
        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.raw_output["finish_reason"])

    async def test_function_call_abort_failure_timeout_and_cancellation(self):
        for error, expected in ((TimeoutError(), ClientTimeoutError),
                                (RuntimeError("PRIVATE"), ClientTransportError),
                                (asyncio.CancelledError(), asyncio.CancelledError)):
            self.runtime.events = self.function_events()
            with patch.object(FakeSession, "send", side_effect=error):
                with self.assertRaises(expected) as caught:
                    await self.predict_function()
                self.assertNotIn("PRIVATE", str(caught.exception))
                self.assert_session_closed(aborted=True)
        self.runtime.events = []
        with self.assertRaises(ClientTimeoutError):
            await self.predict_function(timeout_seconds=0.01)
        self.assert_session_closed(aborted=True)
        self.runtime.events = self.function_events()
        original = self.runtime.new_session

        async def broken_abort(**options):
            session = await original(**options)
            session.abort.side_effect = RuntimeError("PRIVATE")
            return session

        self.runtime.create_session.side_effect = broken_abort
        with self.assertRaises(ClientTransportError):
            await self.predict_function()
        self.runtime.sessions[-1].disconnect.assert_awaited_once()

    async def test_function_call_installed_sdk_api_schema_and_event_serialization(self):
        if importlib.util.find_spec("copilot") is None:
            self.skipTest("Optional SDK not installed")
        from copilot import CopilotClient as SDKClient
        from copilot.generated.session_events import SessionEvent
        from src.clients.github_copilot_client import _decision_tool
        req = replace(request(), options=(DecisionOption("node-7"), DecisionOption("node-12")))
        tool = _decision_tool(req)
        self.assertEqual(tool.parameters, {
            "type": "object", "properties": {"choice": {"type": "string", "enum": ["node-7", "node-12"]}},
            "required": ["choice"], "additionalProperties": False})
        self.assertIsNone(tool.handler)
        self.assertTrue(tool.skip_permission)
        self.assertEqual(tool.defer, "never")
        self.runtime.events = self.function_events()
        self.runtime.events[1] = SessionEvent.from_dict({
            "id": "00000000-0000-0000-0000-000000000001", "timestamp": "2026-10-04T08:00:00Z",
            "type": "assistant.message", "data": {"content": "", "model": "resolved-gpt",
                "messageId": "message-1", "toolRequests": [
                    {"name": "submit_decision", "toolCallId": "call-1", "arguments": {"choice": "yes"}}]}})
        self.runtime.events[2] = SessionEvent.from_dict({
            "id": "00000000-0000-0000-0000-000000000002", "timestamp": "2026-10-04T08:00:00Z",
            "type": "external_tool.requested", "data": {
                "requestId": "external-1", "sessionId": "session-0", "toolCallId": "call-1",
                "toolName": "submit_decision", "arguments": {"choice": "yes"}}})
        result = await self.predict(output_format="function_call")
        inspect.signature(SDKClient.create_session).bind(None, **self.runtime.sessions[-1].options)
        self.assertEqual(result.raw_output["events"][-1], self.runtime.events[-1].to_dict())
        json.dumps(result.raw_output)

    async def test_output_format_wire_matches_vllm_for_both_gpt_models(self):
        for output_format in (None, "json", "answer_only"):
            settings = {} if output_format is None else {"output_format": output_format}
            content = " YES\n" if output_format == "answer_only" else '{"choice":"yes"}'
            messages = []

            def handler(wire):
                messages.extend(json.loads(wire.content)["messages"])
                return httpx.Response(200, json=completion(content))

            async with VLLMClient(**settings, transport=httpx.MockTransport(handler)) as client:
                await client.predict(request())
            for model in ("gpt-5.4", "gpt-6-astra"):
                with self.subTest(output_format=output_format, model=model):
                    self.runtime.list_models.return_value = [NS(**{**vars(model_info()), "id": model})]
                    self.runtime.answer.data.content = content
                    result = await self.predict(model=model, **settings)
                    opts = self.runtime.sessions[-1].options
                    self.assertEqual(opts["system_message"], {"mode": "replace", "content": messages[0]["content"]})
                    self.assertEqual(self.runtime.prompts[-1], messages[1]["content"])
                    self.assertEqual(opts["model"], model)
                    self.assertEqual(opts["model_capabilities"].limits.max_output_tokens, 4096)
                    self.assertNotIn("output_format", opts)
                    self.assertNotIn("response_format", opts)
                    self.assertEqual(result.selected_option_id, "yes")
                    self.assertEqual(result.raw_output["content"], content)
                    self.assert_session_closed()

    async def test_answer_only_parser_acceptance_rejection_and_cleanup(self):
        for text, expected in ANSWER_ONLY_CASES:
            with self.subTest(text=text):
                self.runtime.answer.data.content = text
                before = len(self.runtime.prompts)
                if expected is None:
                    with self.assertRaises(InvalidResponseError) as shared:
                        parse_answer_only(text, request())
                    with self.assertRaises(InvalidResponseError) as caught:
                        await self.predict(output_format="answer_only")
                    self.assertEqual(caught.exception.diagnostic_code, shared.exception.diagnostic_code)
                else:
                    self.assertEqual((await self.predict(output_format="answer_only")).selected_option_id, expected)
                self.assertEqual(len(self.runtime.prompts), before + 1)
                self.assert_session_closed(aborted=expected is None)

    async def test_answer_only_preserves_reasoning_controls_and_contradiction_checks(self):
        self.runtime.answer.data.content = "yes"
        with self.assertRaises(UnsupportedRequestError):
            await self.predict(output_format="answer_only", think=False)
        self.runtime.create_session.assert_not_called()
        self.runtime.events[0].data.reasoning_effort = "high"
        result = await self.predict(output_format="answer_only", think=True, reasoning_effort="high")
        self.assertEqual(result.usage.reasoning_tokens, 14)
        self.assertEqual(self.runtime.sessions[-1].options["reasoning_effort"], "high")
        with self.assertRaisesRegex(InvalidResponseError, "different reasoning"):
            await self.predict(output_format="answer_only", think=True, reasoning_effort="low")

        self.runtime.list_models.return_value = [model_info(False)]
        usage = self.runtime.events[0].data
        usage.reasoning_effort, usage.reasoning_tokens = None, 0
        for field, value in (("content", "<think>reasoning</think>yes"),
                             ("reasoning_text", "reasoning"), ("reasoning_blocks", ["reasoning"]),
                             ("reasoning_tokens", 1), ("reasoning_effort", "low")):
            target = usage if field in ("reasoning_tokens", "reasoning_effort") else self.runtime.answer.data
            with self.subTest(field=field), patch.object(target, field, value, create=True):
                with self.assertRaises(InvalidResponseError) as caught:
                    await self.predict(output_format="answer_only", think=False)
                self.assertEqual(caught.exception.diagnostic_code, "unexpected_reasoning")
        self.runtime.answer.data.content = "<think></think>no"
        self.assertEqual((await self.predict(output_format="answer_only", think=False)).selected_option_id, "no")
        self.assertIsNone(self.runtime.sessions[-1].options["reasoning_effort"])

    async def test_lazy_isolated_tool_free_sessions_and_normalized_result(self):
        self.runtime.events[0].data.reasoning_effort = "high"
        client = self.client(think=True, reasoning_effort="high", max_tokens=2048)
        self.assertFalse(self.factory_options)
        async with client:
            first = await client.predict(request())
            await client.predict(request("different-local-id"))
            self.assertEqual(first.resolved_model, "resolved-gpt")
            self.assertEqual(first.request_id, request().request_id)
            self.assertEqual(first.selected_option_id, "yes")
            self.assertEqual(first.usage.reasoning_tokens, 14)
            self.assertIsNone(first.probabilities)
            self.assertIsNone(first.confidence)
            self.assertEqual(len(self.runtime.sessions), 2)
            self.assertEqual(self.runtime.prompts[0], self.runtime.prompts[1])
            self.assertNotIn("PRIVATE", self.runtime.prompts[0])
            self.assertEqual(self.factory_options[0]["mode"], "empty")
            self.assertFalse(self.factory_options[0]["use_logged_in_user"])
            directory = Path(self.factory_options[0]["base_directory"])
            self.assertTrue(directory.is_dir())
            for session in self.runtime.sessions:
                opts = session.options
                self.assertEqual(opts["available_tools"], [])
                self.assertEqual(opts["tools"], [])
                self.assertEqual(opts["reasoning_effort"], "high")
                self.assertEqual(opts["model_capabilities"].limits.max_output_tokens, 2048)
                self.assertEqual(opts["system_message"]["mode"], "replace")
                self.assertEqual(opts["system_message"]["content"], SYSTEM_MESSAGE)
                self.assertEqual(opts["memory"], {"enabled": False})
                self.assertEqual(opts["infinite_sessions"], {"enabled": False})
                self.assertFalse(opts["enable_config_discovery"])
                self.assertFalse(opts["enable_skills"])
                self.assertFalse(opts["enable_file_hooks"])
                self.assertEqual(opts["working_directory"], str(directory))
                self.assertEqual(opts["on_permission_request"](None, None).kind, "denied")
                self.assertEqual(opts["on_auto_mode_switch_request"](), "no")
                session.disconnect.assert_awaited_once()
                session.abort.assert_not_called()
                self.runtime.delete_session.assert_any_await(session.session_id)
        self.assertFalse(directory.exists())
        self.runtime.start.assert_awaited_once()
        self.runtime.stop.assert_awaited_once()
        await client.aclose()
        self.runtime.stop.assert_awaited_once()

    async def test_optional_dependency_api_signatures_without_runtime(self):
        if importlib.util.find_spec("copilot") is None:
            self.skipTest("Optional SDK not installed")
        from copilot import CopilotClient as SDKClient
        from copilot.session import ModelCapabilitiesOverride, ModelLimitsOverride
        await self.predict()
        inspect.signature(SDKClient).bind(**self.factory_options[0])
        opts = self.runtime.sessions[0].options.copy()
        opts["model_capabilities"] = ModelCapabilitiesOverride(limits=ModelLimitsOverride(max_output_tokens=4096))
        inspect.signature(SDKClient.create_session).bind(None, **opts)

    async def test_env_token_and_existing_login_without_device_flow(self):
        for token in (None, "FAKE_ENV_TOKEN"):
            with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": token} if token else {}, clear=True):
                async with create_client("github_copilot", model="test-gpt") as client:
                    await client.predict(request())
                    options = self.factory_options[-1]
                    workspace = Path(options["working_directory"])
                    self.assertTrue(workspace.is_dir())
                    self.assertEqual(options["base_directory"],
                                     str(workspace) if token else str(Path.home() / ".copilot"))
                    self.assertEqual(self.runtime.sessions[-1].options["working_directory"],
                                     str(workspace))
            self.assertEqual(self.factory_options[-1]["github_token"], token)
            self.assertEqual(self.factory_options[-1]["use_logged_in_user"], token is None)
            self.assertFalse(workspace.exists())

    async def test_existing_login_honors_copilot_home_without_reusing_workspace(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"COPILOT_HOME": home}, clear=True):
                async with create_client("github_copilot", model="test-gpt") as client:
                    await client.predict(request())
                    options = self.factory_options[-1]
                    self.assertEqual(options["base_directory"], home)
                    workspace = Path(options["working_directory"])
                    self.assertNotEqual(workspace, Path(home))
                    self.assertTrue(workspace.is_dir())
                self.assertTrue(Path(home).is_dir())
                self.assertFalse(workspace.exists())

    async def test_supported_thinking_controls_and_default(self):
        for think, explicit, expected in ((True, None, "medium"), (None, None, None),
                                          (None, "low", "low")):
            self.runtime.events[0].data.reasoning_effort = expected
            await self.predict(think=think, reasoning_effort=explicit)
            self.assertEqual(self.runtime.sessions[-1].options["reasoning_effort"], expected)
        self.runtime.list_models.return_value = [model_info(False)]
        self.runtime.events[0].data.reasoning_effort = None
        self.runtime.events[0].data.reasoning_tokens = 0
        await self.predict(think=False)
        self.assertIsNone(self.runtime.sessions[-1].options["reasoning_effort"])

    async def test_unsupported_model_thinking_or_auth_never_sends_question(self):
        for info, settings, authenticated in (
            (model_info(), {"think": False}, True),
            (model_info(False), {"think": True}, True),
            (model_info(), {"reasoning_effort": "max"}, True),
            (model_info(), {"model": "not-available"}, True),
            (model_info(), {}, False),
        ):
            with self.subTest(settings=settings, authenticated=authenticated):
                self.runtime.list_models.return_value = [info]
                self.runtime.get_auth_status.return_value = NS(
                    isAuthenticated=authenticated, statusMessage="PRIVATE_AUTH")
                with self.assertRaises(
                        UnsupportedRequestError if authenticated else ClientTransportError) as caught:
                    await self.predict(**settings)
                self.assertNotIn("PRIVATE", str(caught.exception))
        self.runtime.create_session.assert_not_called()
        self.assertEqual(self.runtime.prompts, [])

    async def test_timeout_cancellation_and_sdk_failure_abort_and_clean_up(self):
        for error, expected in ((TimeoutError("PRIVATE"), ClientTimeoutError),
                                (RuntimeError("PRIVATE_SDK"), ClientTransportError),
                                (asyncio.CancelledError(), asyncio.CancelledError)):
            self.runtime.error = error
            with self.assertRaises(expected) as caught:
                await self.predict()
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assert_session_closed(aborted=True)
        self.assertEqual(len(self.runtime.prompts), 3)

    async def test_wall_clock_timeout_during_send(self):
        async def hanging(*args, **kwargs):
            await asyncio.Future()
        with patch.object(FakeSession, "send_and_wait", new=hanging):
            with self.assertRaises(ClientTimeoutError):
                await self.predict(timeout_seconds=0.01)
        self.runtime.sessions[-1].abort.assert_awaited_once()

    async def test_external_and_repeated_cancellation_still_cleans_session(self):
        entered, cleaning, finish_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def hanging(*args, **kwargs):
            entered.set()
            await asyncio.Future()

        async def slow_abort():
            cleaning.set()
            await finish_cleanup.wait()

        with patch.object(FakeSession, "send_and_wait", new=hanging):
            async with self.client() as client:
                task = asyncio.create_task(client.predict(request()))
                await asyncio.wait_for(entered.wait(), timeout=1)
                session = self.runtime.sessions[-1]
                session.abort.side_effect = slow_abort
                task.cancel()
                await asyncio.wait_for(cleaning.wait(), timeout=1)
                task.cancel()
                finish_cleanup.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_session_closed(aborted=True)

    async def test_reported_model_effort_or_no_think_contradictions_are_rejected(self):
        usage = self.runtime.events[0].data
        usage.model = "different-model"
        with self.assertRaisesRegex(InvalidResponseError, "inconsistent"):
            await self.predict()
        usage.model = "resolved-gpt"
        with self.assertRaisesRegex(InvalidResponseError, "different reasoning"):
            await self.predict(think=True, reasoning_effort="high")
        self.runtime.list_models.return_value = [model_info(False)]
        with self.assertRaisesRegex(InvalidResponseError, "no-think"):
            await self.predict(think=False)

    async def test_initialization_failure_is_sanitized_and_resources_closed(self):
        self.runtime.start.side_effect = RuntimeError("PRIVATE_START_FAILURE")
        async with self.client() as client:
            with self.assertRaises(ClientTransportError) as caught:
                await client.predict(request())
            self.assertNotIn("PRIVATE", str(caught.exception))
            directory = Path(self.factory_options[0]["base_directory"])
        self.assertFalse(directory.exists())
        self.runtime.stop.assert_awaited_once()

    async def test_invalid_outputs_metadata_and_sdk_retries_fail_without_repair(self):
        usage = self.runtime.events[0]
        cases = [(self.runtime.answer.data, "content", content) for content in
                 ("", "yes", '{"choice":"not-an-option"}', '<think>{"choice":"yes"}')]
        cases += [(self.runtime, "events", events) for events in
                  ([usage, usage], [usage, NS(type="assistant.turn_retry", data=NS())])]
        cases += [(usage.data, key, value) for key, value in (
            ("finish_reason", "length"), ("input_tokens", True),
            ("reasoning_tokens", -1), ("content_filter_triggered", True))]
        cases += [(self.runtime.answer.data, "tool_requests", [{}]), (self.runtime, "answer", None)]
        for target, field, value in cases:
            with self.subTest(field=field, value=value), patch.object(target, field, value, create=True):
                before = len(self.runtime.prompts)
                with self.assertRaises(InvalidResponseError):
                    await self.predict()
                self.assertEqual(len(self.runtime.prompts), before + 1)
                self.assert_session_closed(aborted=True)

    async def test_missing_usage_is_unknown_not_invented(self):
        self.runtime.events = []
        result = await self.predict()
        self.assertEqual(result.resolved_model, "resolved-gpt")
        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.usage.reasoning_tokens)

    async def test_close_unused_and_protocol_validation_are_side_effect_free(self):
        client = self.client()
        with self.assertRaises(UnsupportedRequestError):
            await client.predict(replace(request(), protocol_version="future"))
        await client.aclose()
        with self.assertRaises(ClientClosedError):
            await client.predict(request())
        self.assertFalse(self.factory_options)
        self.runtime.start.assert_not_called()

    def test_registry_listing_and_constructor_do_not_load_sdk(self):
        with patch("src.clients.github_copilot_client._sdk_types", side_effect=AssertionError("No import")):
            self.assertIn("github_copilot", available_clients())
            self.assertIsInstance(create_client("github_copilot"), GitHubCopilotClient)

    def test_invalid_settings(self):
        for kwargs in ({"output_format": "unknown"}, {"output_format": []}, {"output_format": None},
                       {"output_format": False}, {"think": "no"}, {"reasoning_effort": "minimal"},
                       {"think": True, "reasoning_effort": "none"},
                       {"think": False, "reasoning_effort": "low"},
                       {"max_tokens": -1}, {"model": ""}, {"timeout_seconds": 0},
                       {"github_token": ""}, {"github_token": "bad\nkey"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.client(**kwargs)


if __name__ == "__main__":
    unittest.main()