"""
tests/test_synth.py

Grounded generation end to end against the fake OpenAI-compatible server:
rounds, target size, filters, judge, dedup, evolve and failure handling.
"""
from __future__ import annotations

import asyncio

import httpx2
import pytest
from openai import AuthenticationError

from engine import ChatClient, EndpointSettings
from pipeline.prompts import ANSWER_SYSTEM, JUDGE_SYSTEM, QUESTION_SYSTEM, JudgeScores
from pipeline.synth import (
    MAX_QUESTIONS_PER_CHUNK,
    SynthSettings,
    judge_passes,
    questions_per_chunk,
    synthesize,
)
from tests.fake_openai import FakeOpenAI, fake_answer, fake_embedding


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))

CHUNKS = [
    f"Section {i}. Photosynthesis converts sunlight into chemical energy; chlorophyll{i} absorbs light "
    f"and the Calvin{i} cycle fixes carbon dioxide into sugars."
    for i in range(5)
]


def _client(fake: FakeOpenAI, model: str = "teacher") -> ChatClient:
    settings = EndpointSettings(model=model, base_url="http://fake/v1", api_key="k", concurrency=4, max_retries=2)
    return ChatClient(settings, http_client=httpx2.AsyncClient(transport=fake.transport))


def _synth(fake: FakeOpenAI, judge: bool = True, chunks: list[str] = CHUNKS, embed: bool = False, **settings):
    teacher = _client(fake)
    judge_client = _client(fake, "judge") if judge else None
    embedder = _client(fake, "embedder") if embed else None
    progress: list[float] = []
    recs, stats = asyncio.run(synthesize(
        chunks, [teacher], SynthSettings(**{"target": 20, **settings}), judge=judge_client, progress=progress.append,
        embedder=embedder,
    ))
    return recs, stats, progress


def _kinds(fake: FakeOpenAI) -> list[str]:
    out = []
    for call in fake.calls:
        system = call["messages"][0]["content"]
        out.append("question" if QUESTION_SYSTEM in system else "answer" if ANSWER_SYSTEM in system
                   else "judge" if JUDGE_SYSTEM in system else "other")
    return out


class TestTarget:

    def test_reaches_target_exactly(self):
        recs, stats, progress = _synth(FakeOpenAI())
        assert len(recs) == 20 and stats.accepted == 20
        assert progress[-1] == pytest.approx(1.0)

    def test_extra_rounds_make_up_for_rejections(self):
        recs, stats, _ = _synth(FakeOpenAI(refuse_every=3, low_score_every=4))
        assert len(recs) == 20
        assert stats.rounds >= 2
        assert stats.answers_filtered["refusal"] > 0 and stats.judge_rejected > 0

    def test_stops_when_source_is_exhausted(self):
        recs, stats, _ = _synth(FakeOpenAI(questions_per_passage=2), target=50)
        assert len(recs) == 10  # 5 chunks × 2 distinct questions
        assert stats.exhausted_chunks == 5

    def test_malformed_round_does_not_retire_a_chunk(self):
        # A round of questions that cite "the passage" is dropped, but the chunk gets another round.
        recs, stats, _ = _synth(FakeOpenAI(leaky_first=True))
        assert len(recs) == 20
        assert stats.questions_dropped > 0 and stats.exhausted_chunks == 0

    def test_repeats_retire_a_chunk(self):
        _, stats, _ = _synth(FakeOpenAI(questions_per_passage=2), target=50)
        assert stats.questions_repeated > 0 and stats.exhausted_chunks == 5

    def test_rejected_pairs_are_kept_with_a_reason(self):
        _, stats, _ = _synth(FakeOpenAI(refuse_every=3, low_score_every=4))
        reasons = {r.meta["rejected"] for r in stats.rejected}
        assert reasons == {"filter: refusal", "judge"}
        judged = [r for r in stats.rejected if r.meta["rejected"] == "judge"]
        assert all(r.meta["judge"]["faithfulness"] == 2 for r in judged)
        assert "rejected" not in stats.as_dict()

    def test_question_prompt_has_no_labels_to_quote(self):
        fake = FakeOpenAI()
        _synth(fake)
        prompts = [c["messages"][-1]["content"] for c, k in zip(fake.calls, _kinds(fake), strict=True)
                   if k == "question"]
        assert prompts and not any("passage A" in p or "passage B" in p for p in prompts)

    def test_avoid_list_sent_in_later_rounds(self):
        fake = FakeOpenAI(questions_per_passage=2)
        _synth(fake, target=50)
        later = [c for c in fake.calls if "Already asked" in c["messages"][-1]["content"]]
        assert later, "second-round question prompts must list the questions already asked"


class TestPairs:

    def test_answers_are_grounded_and_meta_recorded(self):
        recs, _, _ = _synth(FakeOpenAI())
        for rec in recs:
            assert rec.output == fake_answer(rec.instruction)
            assert rec.meta["model"] == "teacher"
            assert rec.meta["judge"]["faithfulness"] == 5
            assert 0 <= rec.meta["chunk"] < len(CHUNKS)

    def test_answer_prompt_contains_the_source(self):
        fake = FakeOpenAI()
        _synth(fake)
        answer_calls = [c for c, k in zip(fake.calls, _kinds(fake), strict=True) if k == "answer"]
        assert all("Source passage" in c["messages"][-1]["content"] for c in answer_calls)

    def test_types_are_varied(self):
        recs, _, _ = _synth(FakeOpenAI())
        assert len({r.meta["type"] for r in recs}) >= 3

    def test_no_judge_in_fast_mode(self):
        fake = FakeOpenAI()
        recs, stats, _ = _synth(fake, judge=False)
        assert len(recs) == 20 and stats.judged == 0
        assert "judge" not in _kinds(fake)
        assert all(r.meta["judge"] is None for r in recs)

    def test_evolve_rewrites_questions(self):
        recs, stats, _ = _synth(FakeOpenAI(), evolve=True)
        assert stats.evolved > 0
        assert all(r.meta["evolved"] and r.instruction.startswith("Compare and explain") for r in recs)

    def test_semantic_dedup_drops_paraphrases(self):
        fake = FakeOpenAI()
        recs, stats, _ = _synth(fake, embed=True, semantic_threshold=0.95)
        assert stats.semantic_duplicates > 0
        assert any(c.get("input") for c in fake.calls), "embeddings endpoint must be used"
        vectors = [fake_embedding(f"{r.instruction}\n{r.output}") for r in recs]
        assert all(_cos(a, b) < 0.95 for i, a in enumerate(vectors) for b in vectors[i + 1:])

    def test_semantic_dedup_off_without_embedder(self):
        fake = FakeOpenAI()
        _, stats, _ = _synth(fake)
        assert stats.semantic_duplicates == 0 and not any("input" in c for c in fake.calls)

    def test_results_are_in_stable_order(self):
        a, _, _ = _synth(FakeOpenAI())
        b, _, _ = _synth(FakeOpenAI())
        assert [r.instruction for r in a] == [r.instruction for r in b]


class TestFailures:

    def test_fatal_error_stops_the_run(self):
        with pytest.raises(AuthenticationError):
            _synth(FakeOpenAI(auth_error=True))

    def test_transient_errors_are_survived(self):
        recs, _, _ = _synth(FakeOpenAI(rate_limit_first=True, garbage_json_first=True, reject_json_schema=True))
        assert len(recs) == 20

    def test_needs_chunks_and_teachers(self):
        with pytest.raises(ValueError, match="No source chunks"):
            asyncio.run(synthesize([], [_client(FakeOpenAI())], SynthSettings(target=1)))
        with pytest.raises(ValueError, match="teacher"):
            asyncio.run(synthesize(["text"], [], SynthSettings(target=1)))


class TestHelpers:

    @pytest.mark.parametrize("remaining,active,acceptance,expected", [
        (20, 5, 0.7, 6), (1, 5, 0.7, 1), (1000, 2, 0.5, MAX_QUESTIONS_PER_CHUNK), (0, 5, 0.7, 0), (10, 0, 0.7, 0),
    ])
    def test_questions_per_chunk(self, remaining, active, acceptance, expected):
        assert questions_per_chunk(remaining, active, acceptance) == expected

    @pytest.mark.parametrize("scores,threshold,ok", [
        ((5, 5, 5), 4, True), ((4, 5, 4), 4, True), ((3, 5, 5), 4, False), ((5, 9, 5), 4, False), ((2, 2, 2), 2, True),
    ])
    def test_judge_passes(self, scores, threshold, ok):
        f, h, c = scores
        assert judge_passes(JudgeScores(faithfulness=f, helpfulness=h, correctness=c, reason=""), threshold) is ok
