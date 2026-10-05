"""
An in-memory, deterministic OpenAI-compatible chat-completions server for tests.

Plug it into the engine with `httpx2.AsyncClient(transport=fake.transport)`.
It recognises Brainbrew's prompts (pipeline/prompts.py) and answers each kind
plausibly, and it can be told to misbehave the way real servers and models do.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx2

from pipeline.prompts import (
    ANSWER_SYSTEM,
    EVOLVE_SYSTEM,
    JUDGE_SYSTEM,
    QUESTION_SYSTEM,
    QUESTION_TYPES,
)

_PASSAGE_RE = re.compile(r"(?:Source passage|Document excerpt):\n<<<\n(?P<p>.*?)\n>>>", re.DOTALL)
_K_RE = re.compile(r"Write (?P<k>\d+) (?:self-contained )?questions")
_QUESTION_RE = re.compile(r"Question: (?P<q>.*?)(?:\n\n|$)", re.DOTALL)
_WORD_RE = re.compile(r"[A-Za-z]{6,}")


def _tag(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def fake_answer(question: str) -> str:
    """The answer the fake gives to *question* (long enough to pass filters)."""
    return (f"Answer to [{_tag(question)}]: the key idea is explained step by step, "
            "with the mechanism, an example, and the main consequence spelled out.")


EMBEDDING_DIM = 64


def fake_embedding(text: str) -> list[float]:
    """Hashed bag of words: texts sharing most words get a high cosine similarity."""
    vec = [0.0] * EMBEDDING_DIM
    for word in re.findall(r"[a-z]{3,}", text.lower()):
        vec[int(_tag(word), 16) % EMBEDDING_DIM] += 1.0
    return vec


@dataclass
class FakeOpenAI:
    """Configurable fake. All counters are per instance."""

    reject_json_schema: bool = False      # 400 on response_format=json_schema
    reject_json_object: bool = False      # 400 on response_format=json_object too
    reject_status: int = 400              # llama-cpp-python's server rejects with 500
    server_error_first: bool = False      # first request gets a plain 500
    garbage_json_first: bool = False      # first structured reply is not JSON
    refuse_every: int = 0                 # every Nth answer is a refusal
    low_score_every: int = 0              # every Nth judge call scores 2
    rate_limit_first: bool = False        # first request gets a 429
    auth_error: bool = False              # every request gets a 401
    questions_per_passage: int | None = None  # cap distinct questions per passage; then repeats
    leaky_first: bool = False             # first round per passage: questions that cite "the passage"
    hold: threading.Event | None = None   # requests wait until this is set (a run that stays busy)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)
    _asked: dict[str, int] = field(default_factory=dict)

    @property
    def transport(self) -> httpx2.MockTransport:
        if self.hold is None:
            return httpx2.MockTransport(self._handle)
        hold = self.hold

        async def held(request: httpx2.Request) -> httpx2.Response:
            while not hold.is_set():
                await asyncio.sleep(0.02)
            return self._handle(request)

        return httpx2.MockTransport(held)

    def count(self, kind: str) -> int:
        return self._counters.get(kind, 0)

    def _bump(self, kind: str) -> int:
        self._counters[kind] = self._counters.get(kind, 0) + 1
        return self._counters[kind]

    # ── HTTP ──────────────────────────────────────────────────────────────
    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if self.auth_error:
            return httpx2.Response(401, json={"error": {"message": "bad key", "type": "invalid_api_key"}})
        if self.rate_limit_first and self._bump("http") == 1:
            return httpx2.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        if self.server_error_first and self._bump("server_error") == 1:
            return httpx2.Response(500, json={"error": {"message": "CUDA error: out of memory"}})
        body = json.loads(request.content)
        self.calls.append(body)
        if request.url.path.endswith("/embeddings"):
            return self._embeddings(body)
        fmt = (body.get("response_format") or {}).get("type")
        if fmt == "json_schema" and self.reject_json_schema:
            return httpx2.Response(self.reject_status,
                                   json={"error": {"message": "response_format json_schema not supported"}})
        if fmt == "json_object" and self.reject_json_object:
            return httpx2.Response(self.reject_status,
                                   json={"error": {"message": "response_format json_object not supported"}})
        content = self._reply(body["messages"])
        return httpx2.Response(200, json={
            "id": f"chatcmpl-{_tag(content)}", "object": "chat.completion", "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    def _embeddings(self, body: dict[str, Any]) -> httpx2.Response:
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = [{"object": "embedding", "index": i, "embedding": fake_embedding(t)}
                for i, t in enumerate(texts)]
        tokens = sum(len(t.split()) for t in texts)
        return httpx2.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                          "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

    # ── content ───────────────────────────────────────────────────────────
    def _reply(self, messages: list[dict[str, str]]) -> str:
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        user = messages[-1]["content"]
        structured = QUESTION_SYSTEM in system or EVOLVE_SYSTEM in system or JUDGE_SYSTEM in system
        if structured and self.garbage_json_first and self._bump("garbage") == 1:
            return "Sure! Here is what you asked for, but not as JSON."
        if QUESTION_SYSTEM in system:
            return self._questions(user)
        if EVOLVE_SYSTEM in system:
            q = _QUESTION_RE.search(user).group("q").strip()
            return json.dumps({"question": f"Compare and explain in depth: {q}", "answerable_from_source": True})
        if ANSWER_SYSTEM in system:
            n = self._bump("answer")
            if self.refuse_every and n % self.refuse_every == 0:
                return "I'm sorry, but as an AI I cannot help with that."
            return fake_answer(_QUESTION_RE.search(user).group("q").strip())
        if JUDGE_SYSTEM in system:
            n = self._bump("judge")
            low = self.low_score_every and n % self.low_score_every == 0
            s = 2 if low else 5
            return json.dumps({"faithfulness": s, "helpfulness": s, "correctness": s, "reason": "ok"})
        return "Unrecognised prompt."

    def _questions(self, user: str) -> str:
        passage = _PASSAGE_RE.search(user).group("p")
        k = int(_K_RE.search(user).group("k"))
        key = _tag(passage)
        if self.leaky_first and self._bump(f"leaky:{key}") == 1:
            return json.dumps({"questions": [
                {"type": "factual", "question": f"In source passage A, what is point {j}?"} for j in range(k)
            ]})
        start = self._asked.get(key, 0)
        stop = start + k
        if self.questions_per_passage is not None and stop > self.questions_per_passage:
            # Like real models: once out of material, repeat earlier questions.
            start, stop = 0, min(k, self.questions_per_passage)
        words = _WORD_RE.findall(passage) or ["topic"]
        questions = []
        for j in range(start, stop):
            word = words[j % len(words)]
            questions.append({
                "type": QUESTION_TYPES[j % (len(QUESTION_TYPES) - 1)],
                "question": f"What role does '{word}' play in section {key}, point {j}?",
            })
        self._asked[key] = max(start, stop)
        return json.dumps({"questions": questions})
