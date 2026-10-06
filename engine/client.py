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
* token usage accounting, so runs can report what they cost;
* no requests to cloud metadata addresses, even via DNS or redirects
  (engine/netguard.py).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx2
from openai import (
    APIStatusError,
    AsyncOpenAI,
    BadRequestError,
    DefaultAsyncHttpxClient,
    InternalServerError,
    UnprocessableEntityError,
)
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from engine.netguard import guard_client

OPENAI_URL = "https://api.openai.com/v1"


def default_base_url() -> str:
    """OPENAI_BASE_URL, or OpenAI. An empty or blank variable counts as unset: the SDK
    would otherwise use "" as the base URL and fail every request."""
    return os.getenv("OPENAI_BASE_URL", "").strip() or OPENAI_URL


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    """The model did not return valid JSON for the requested schema."""


@dataclass(frozen=True)
class EndpointSettings:
    """Where and how to call the model. Never logged with the key."""

    model: str
    base_url: str | None = None  # None: OPENAI_BASE_URL if set and non-empty, else api.openai.com
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_s: float = 120.0
    max_retries: int = 4
    concurrency: int = 8
    seed: int | None = None  # sent with every chat request when set (reproducible sampling)

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


EMBED_BATCH = 64


class ChatClient:
    """Async chat client bound to one endpoint + model."""

    def __init__(self, settings: EndpointSettings, http_client: httpx2.AsyncClient | None = None) -> None:
        self.settings = settings
        self.usage = Usage()
        self._caps = _Capabilities()
        self._semaphore = asyncio.Semaphore(max(1, settings.concurrency))
        self._client = AsyncOpenAI(
            api_key=settings.api_key or "not-needed",
            base_url=settings.base_url or default_base_url(),
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
            # Refuse metadata addresses after DNS resolution and on redirects (SSRF).
            http_client=guard_client(http_client or DefaultAsyncHttpxClient()),
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
        async with self._semaphore:
            return await self._complete(messages, temperature, max_tokens, response_format)

    async def _complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> str:
        """One request; the caller holds the concurrency semaphore."""
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.settings.seed is not None:
            kwargs["seed"] = self.settings.seed
        completion = await self._client.chat.completions.create(**kwargs)
        self.usage.add(completion.usage)
        if not completion.choices:
            return ""
        return (completion.choices[0].message.content or "").strip()

    # ── embeddings ────────────────────────────────────────────────────────
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors for *texts*, in input order (batched, concurrent)."""

        async def batch(chunk: list[str]) -> list[list[float]]:
            async with self._semaphore:
                resp = await self._client.embeddings.create(
                    model=self.settings.model, input=chunk, encoding_format="float",
                )
            self.usage.add(resp.usage)
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

        chunks = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        results = await asyncio.gather(*(batch(c) for c in chunks))
        return [vec for result in results for vec in result]

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
        # Read the capability only once a slot is free: calls queued behind the
        # limit then use what an earlier call learned instead of retrying it.
        async with self._semaphore:
            while True:
                mode = self._caps.response_format
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
                try:
                    return await self._complete(msgs, temperature, max_tokens, fmt)
                except (BadRequestError, UnprocessableEntityError, InternalServerError) as exc:
                    # The server rejected this response_format: degrade once, remember it.
                    next_mode = {"json_schema": "json_object", "json_object": "prompt"}.get(mode)
                    if next_mode is None or not _rejects_response_format(exc):
                        raise
                    async with self._caps.lock:
                        if self._caps.response_format == mode:
                            self._caps.response_format = next_mode


def _rejects_response_format(exc: APIStatusError) -> bool:
    """A 400/422 on a structured request, or a 5xx that names `response_format`.

    Most servers answer an unsupported response_format with 400 or 422;
    llama-cpp-python's server answers 500 with a validation message instead.
    Any other 5xx is a real server failure and is not a capability signal.
    """
    if isinstance(exc, (BadRequestError, UnprocessableEntityError)):
        return True
    return "response_format" in str(exc)


def _with_schema_hint(messages: list[dict[str, str]], schema: type[BaseModel]) -> list[dict[str, str]]:
    hint = (
        "Reply with a single JSON object and nothing else. It must match this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    return [{"role": "system", "content": hint}, *messages]
