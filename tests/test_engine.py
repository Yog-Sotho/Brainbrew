"""
tests/test_engine.py

The async OpenAI-compatible client against the in-memory fake server: real
openai SDK + real HTTP request/response handling, no network.
"""
from __future__ import annotations

import asyncio
import json

import httpx2
import pytest
from openai import AuthenticationError
from pydantic import BaseModel

from engine import ChatClient, EndpointSettings, StructuredOutputError
from engine.client import parse_json_reply
from pipeline.prompts import QuestionSet, question_messages
from tests.fake_openai import FakeOpenAI


def _client(fake: FakeOpenAI, **kw) -> ChatClient:
    settings = EndpointSettings(model="m", base_url="http://fake/v1", api_key="k", max_retries=2, **kw)
    return ChatClient(settings, http_client=httpx2.AsyncClient(transport=fake.transport))


def _run(coro):
    return asyncio.run(coro)


QUESTIONS = question_messages("Plants convert sunlight into chemical energy.", 3, ("factual", "conceptual"))


class TestStructuredOutput:

    def test_json_schema_used_when_supported(self):
        fake = FakeOpenAI()
        qs = _run(_client(fake).chat_json(QUESTIONS, QuestionSet))
        assert len(qs.questions) == 3
        fmt = fake.calls[0]["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"]["additionalProperties"] is False

    def test_falls_back_to_json_object_then_remembers(self):
        fake = FakeOpenAI(reject_json_schema=True)
        client = _client(fake)

        async def twice():
            await client.chat_json(QUESTIONS, QuestionSet)
            await client.chat_json(QUESTIONS, QuestionSet)

        _run(twice())
        kinds = [(c.get("response_format") or {}).get("type") for c in fake.calls]
        assert kinds == ["json_schema", "json_object", "json_object"]
        assert "JSON schema" in fake.calls[1]["messages"][0]["content"]

    def test_queued_calls_use_the_learned_format(self):
        # Calls waiting on the concurrency limit must not retry an unsupported format.
        fake = FakeOpenAI(reject_json_schema=True)
        client = _client(fake, concurrency=1)

        async def many():
            await asyncio.gather(*(client.chat_json(QUESTIONS, QuestionSet) for _ in range(6)))

        _run(many())
        kinds = [(c.get("response_format") or {}).get("type") for c in fake.calls]
        assert kinds.count("json_schema") == 1 and kinds.count("json_object") == 6

    def test_500_naming_response_format_also_degrades(self):
        # llama-cpp-python's server rejects json_schema with a 500 validation error.
        fake = FakeOpenAI(reject_json_schema=True, reject_status=500)
        qs = _run(_client(fake).chat_json(QUESTIONS, QuestionSet))
        assert qs.questions
        assert (fake.calls[-1].get("response_format") or {}).get("type") == "json_object"

    def test_other_server_errors_do_not_degrade(self):
        fake = FakeOpenAI(server_error_first=True)
        client = _client(fake)
        _run(client.chat_json(QUESTIONS, QuestionSet))  # the SDK retries the 500
        assert client._caps.response_format == "json_schema"

    def test_falls_back_to_prompt_only(self):
        fake = FakeOpenAI(reject_json_schema=True, reject_json_object=True)
        _run(_client(fake).chat_json(QUESTIONS, QuestionSet))
        assert "response_format" not in fake.calls[-1]

    def test_invalid_json_is_retried(self):
        fake = FakeOpenAI(garbage_json_first=True)
        qs = _run(_client(fake).chat_json(QUESTIONS, QuestionSet))
        assert qs.questions and len(fake.calls) == 2

    def test_gives_up_after_attempts(self):
        class Never(BaseModel):
            impossible_field: int

        with pytest.raises(StructuredOutputError):
            _run(_client(FakeOpenAI()).chat_json(QUESTIONS, Never, attempts=2))


class TestTransport:

    def test_rate_limit_is_retried(self):
        fake = FakeOpenAI(rate_limit_first=True)
        assert _run(_client(fake).chat_json(QUESTIONS, QuestionSet)).questions

    def test_auth_error_is_raised(self):
        with pytest.raises(AuthenticationError):
            _run(_client(FakeOpenAI(auth_error=True)).chat(QUESTIONS))

    def test_usage_is_counted(self):
        client = _client(FakeOpenAI())
        _run(client.chat(QUESTIONS))
        assert client.usage.as_dict() == {"requests": 1, "prompt_tokens": 100, "completion_tokens": 50}

    def test_sampling_settings_sent(self):
        fake = FakeOpenAI()
        _run(_client(fake, temperature=0.3, max_tokens=321).chat(QUESTIONS))
        assert fake.calls[0]["temperature"] == 0.3 and fake.calls[0]["max_tokens"] == 321

    def test_concurrency_is_bounded(self):
        in_flight, peak = 0, 0

        async def slow(request: httpx2.Request) -> httpx2.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return httpx2.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}],
            })

        settings = EndpointSettings(model="m", base_url="http://fake/v1", api_key="k", concurrency=3)
        client = ChatClient(settings, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(slow)))

        async def many():
            await asyncio.gather(*(client.chat(QUESTIONS) for _ in range(12)))

        _run(many())
        assert peak == 3


class TestEmbeddings:

    def test_vectors_in_input_order_across_batches(self):
        from engine.client import EMBED_BATCH
        from tests.fake_openai import fake_embedding

        fake = FakeOpenAI()
        client = _client(fake)
        texts = [f"text number {i} about rivers" for i in range(EMBED_BATCH + 5)]
        vectors = _run(client.embed(texts))
        assert vectors == [fake_embedding(t) for t in texts]
        assert len(fake.calls) == 2 and all(c["encoding_format"] == "float" for c in fake.calls)
        assert client.usage.requests == 2 and client.usage.prompt_tokens > 0


class TestParseReply:

    class M(BaseModel):
        a: int

    @pytest.mark.parametrize("text", [
        '{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go: {"a": 1} hope it helps',
    ])
    def test_tolerant_parsing(self, text):
        assert parse_json_reply(text, self.M).a == 1

    @pytest.mark.parametrize("text", ["no json here", '{"b": 2}', "{broken"])
    def test_rejects_invalid(self, text):
        with pytest.raises(StructuredOutputError):
            parse_json_reply(text, self.M)


def test_settings_repr_hides_key():
    s = EndpointSettings(model="m", api_key="sk-secret-value")
    assert "sk-secret-value" not in repr(s)
    assert "sk-secret-value" not in json.dumps({"s": repr(s)})


class TestDefaultBaseUrl:
    """base_url=None means OPENAI_BASE_URL, or OpenAI; an empty variable counts as unset."""

    @pytest.mark.parametrize(("env", "expected"), [
        (None, "https://api.openai.com/v1/"),
        ("", "https://api.openai.com/v1/"),            # what an undefined CI variable expands to
        ("   ", "https://api.openai.com/v1/"),
        ("http://vllm:8000/v1", "http://vllm:8000/v1/"),
    ])
    def test_resolution(self, monkeypatch, env, expected):
        if env is None:
            monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("OPENAI_BASE_URL", env)
        client = ChatClient(EndpointSettings(model="m"))
        assert str(client._client.base_url) == expected

    def test_explicit_url_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8000/v1")
        client = ChatClient(EndpointSettings(model="m", base_url="http://other:9000/v1"))
        assert str(client._client.base_url) == "http://other:9000/v1/"
