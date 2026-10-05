"""
Async client for any OpenAI-compatible chat-completions API.

One code path for OpenAI, `vllm serve`, Ollama, llama.cpp and hosted providers:

* bounded concurrency (an asyncio semaphore per client);
* transport retries with exponential backoff and jitter — 408/409/429/5xx and
  connection errors, honouring Retry-After — via the openai SDK's built-in
  retry loop;
* content retries via tenacity: when a structured reply is not valid JSON or
  does not match the schema, ask again;
* structured outputs: `response_format={"type": "json_schema", ...}` when the
  server supports it, falling back to JSON mode and then to a schema-in-prompt
  request for servers that reject it;
* token usage accounting, so runs can report what they cost.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx2
from openai import AsyncOpenAI, BadRequestError, UnprocessableEntityError
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    """The model did not return valid JSON for the requested schema."""


@dataclass(frozen=True)
class EndpointSettings:
    """Where and how to call the model. Never logged with the key."""

    model: str
    base_url: str | None = None  # None: the SDK default (OPENAI_BASE_URL or api.openai.com)
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_s: float = 120.0
    max_retries: int = 4
    concurrency: int = 8

    def __repr__(self) -> str:
        return (f"EndpointSettings(model={self.model!r}, base_url={self.base_url!r}, "
                f"api_key={'***' if self.api_key else None}, concurrency={self.concurrency})")


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {"requests": self.requests, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


@dataclass
class _Capabilities:
    """What the server accepted for structured output (learned on first use)."""

    response_format: str = "json_schema"  # -> "json_object" -> "prompt"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    try:
        from openai.lib._pydantic import to_strict_json_schema

        return to_strict_json_schema(model)
    except Exception:  # pragma: no cover - SDK internals moved
        schema = model.model_json_schema()
        schema["additionalProperties"] = False
        return schema


def parse_json_reply[T: BaseModel](text: str, model: type[T]) -> T:
    """Parse a model reply as *model*, tolerating code fences and leading prose."""
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return model.model_validate_json(cleaned)
    except ValidationError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise StructuredOutputError(f"No JSON object in reply: {text[:200]!r}")
    try:
        return model.model_validate(json.loads(cleaned[start:end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(f"Reply does not match {model.__name__}: {exc}") from exc


class ChatClient:
    """Async chat client bound to one endpoint + model."""

    def __init__(self, settings: EndpointSettings, http_client: httpx2.AsyncClient | None = None) -> None:
        self.settings = settings
        self.usage = Usage()
        self._caps = _Capabilities()
        self._semaphore = asyncio.Semaphore(max(1, settings.concurrency))
        self._client = AsyncOpenAI(
            api_key=settings.api_key or "not-needed",
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
            http_client=http_client,
        )

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── plain text ────────────────────────────────────────────────────────
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """One chat completion; returns the assistant text ('' if empty)."""
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        async with self._semaphore:
            completion = await self._client.chat.completions.create(**kwargs)
        self.usage.add(completion.usage)
        if not completion.choices:
            return ""
        return (completion.choices[0].message.content or "").strip()

    # ── structured ────────────────────────────────────────────────────────
    async def chat_json[T: BaseModel](
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        attempts: int = 3,
    ) -> T:
        """A chat completion parsed into *schema*, with content retries."""
        retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            retry=retry_if_exception_type(StructuredOutputError),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                text = await self._structured_call(messages, schema, temperature, max_tokens)
                return parse_json_reply(text, schema)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _structured_call(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        mode = self._caps.response_format
        while True:
            try:
                if mode == "json_schema":
                    fmt: dict[str, Any] | None = {
                        "type": "json_schema",
                        "json_schema": {"name": schema.__name__, "schema": _strict_schema(schema),
                                        "strict": True},
                    }
                    msgs = messages
                elif mode == "json_object":
                    fmt = {"type": "json_object"}
                    msgs = _with_schema_hint(messages, schema)
                else:
                    fmt = None
                    msgs = _with_schema_hint(messages, schema)
                return await self.chat(msgs, temperature=temperature, max_tokens=max_tokens,
                                       response_format=fmt)
            except (BadRequestError, UnprocessableEntityError):
                # The server rejected this response_format: degrade once, remember it.
                next_mode = {"json_schema": "json_object", "json_object": "prompt"}.get(mode)
                if next_mode is None:
                    raise
                async with self._caps.lock:
                    if self._caps.response_format == mode:
                        self._caps.response_format = next_mode
                mode = self._caps.response_format


def _with_schema_hint(messages: list[dict[str, str]], schema: type[BaseModel]) -> list[dict[str, str]]:
    hint = (
        "Reply with a single JSON object and nothing else. It must match this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    return [{"role": "system", "content": hint}, *messages]
