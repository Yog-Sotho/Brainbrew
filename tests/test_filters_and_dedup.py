"""
tests/test_filters_and_dedup.py

Question/answer filters (Phase 2.4) — including the real failure cases seen in
the Phase 1 gate run — and MinHash-LSH near-duplicate removal (Phase 2.5).
"""
from __future__ import annotations

import random
import time

import pytest

from pipeline.dedup import Deduplicator, SemanticDeduplicator, deduplicate, normalise, signature
from pipeline.filters import answer_problem, clean_question
from pipeline.records import Record


class TestCleanQuestion:

    @pytest.mark.parametrize("raw,clean", [
        ("What is photosynthesis?", "What is photosynthesis?"),
        ("Sure! Here's a more complex version of the prompt:\n\n### Prompt Rewritten\n\nHow do plants store energy?",
         "How do plants store energy?"),
        ("1. How do plants store energy?", "How do plants store energy?"),
        ("Question: How do plants store energy?", "How do plants store energy?"),
        ('"How do plants store energy?"', "How do plants store energy?"),
        # Talking about a text is fine when the text is the subject, not the source.
        ("How should the text of a long quotation be punctuated?",
         "How should the text of a long quotation be punctuated?"),
    ])
    def test_cleans(self, raw, clean):
        assert clean_question(raw) == clean

    @pytest.mark.parametrize("raw", [
        "Certainly! I will rewrite the given prompt. Please provide the original prompt so I can proceed.",
        "**Given Prompt:** Explain the following concept from the document clearly and completely:",
        "Why?",
        "",
        # Seen with a 3B teacher in the Phase 2 gate run: not self-contained.
        "What is the main difference between identifying an audience and identifying the day of a place in "
        "the comparison made in the text following passage B?",
        "According to the passage, why do plants need light?",
        "Based on the text, when is a comma required?",
    ])
    def test_drops(self, raw):
        assert clean_question(raw) is None


class TestAnswerProblem:

    @pytest.mark.parametrize("answer,problem", [
        ("", "empty"),
        ("UNANSWERABLE", "unanswerable"),
        ("Too short.", "too short"),
        ("I'm sorry, but as an AI language model I cannot answer questions about this topic at all.", "refusal"),
        ("Understood! If you have any specific questions about this topic or need further clarification, feel free to ask.",
         "non-answer"),
        ("According to the passage, plants use chlorophyll to absorb light and turn it into chemical energy.",
         "mentions the source"),
        ("The text states that plants use chlorophyll to absorb light and turn it into chemical energy.",
         "mentions the source"),
    ])
    def test_problems(self, answer, problem):
        assert answer_problem(answer) == problem

    def test_good_answer_passes(self):
        good = ("Plants capture light with chlorophyll, split water to release oxygen, and use the "
                "energy to fix carbon dioxide into sugars in the Calvin cycle.")
        assert answer_problem(good) is None

    def test_long_answer_starting_with_sure_passes(self):
        # A deflection phrase only disqualifies short replies; a full answer may start politely.
        long = "Sure. " + "Photosynthesis turns light energy into chemical energy stored in glucose. " * 8
        assert answer_problem(long) is None


def _rec(q: str, a: str) -> Record:
    return Record(instruction=q, output=a)


class TestMinHash:

    def test_exact_duplicates(self):
        assert len(deduplicate([_rec("Q one?", "Answer one is here.")] * 3)) == 1

    def test_punctuation_and_case_variants_are_duplicates(self):
        recs = [_rec("What is machine learning?", "Machine learning is a subset of AI."),
                _rec("what is MACHINE learning ?", "Machine learning is a subset of AI!")]
        assert len(deduplicate(recs)) == 1

    def test_near_duplicates_removed(self):
        base = "Photosynthesis converts light energy into chemical energy stored in glucose molecules in plants. " * 3
        recs = [_rec("Explain photosynthesis.", base), _rec("Explain photosynthesis.", base + " Also in algae.")]
        assert len(deduplicate(recs, threshold=0.8)) == 1

    def test_different_records_kept_in_order(self):
        recs = [_rec(f"Question about topic {w}?", f"An answer that is specifically about {w} and nothing else.")
                for w in ("rivers", "volcanoes", "glaciers", "deserts")]
        assert deduplicate(recs) == recs

    def test_incremental_api(self):
        d = Deduplicator(0.85)
        assert d.add_if_new("the quick brown fox jumps over the lazy dog")
        assert d.is_duplicate("the quick brown fox jumps over the lazy dog")
        assert not d.add_if_new("the quick brown fox jumps over the lazy dog")
        assert d.add_if_new("an entirely different sentence about astronomy and stars")

    def test_signature_is_deterministic(self):
        assert (signature(normalise("Hello world, again")) == signature(normalise("hello world again"))).all()

    def test_scales_roughly_linearly(self):
        random.seed(1)
        words = ["light", "energy", "plant", "cell", "water", "carbon", "sugar", "leaf", "sun", "oxygen", "root", "stem"]

        def make(n):
            return [_rec(f"Q{i} " + " ".join(random.choices(words, k=10)), " ".join(random.choices(words, k=40)))
                    for i in range(n)]

        small, large = make(500), make(2000)
        def timed(records):
            start = time.perf_counter()
            deduplicate(records)
            return time.perf_counter() - start

        t_small, t_large = timed(small), timed(large)
        assert t_large < t_small * 8  # quadratic would be ~16x


class TestSemantic:

    def test_cosine_threshold(self):
        d = SemanticDeduplicator(0.9)
        assert d.add_if_new([1.0, 0.0, 0.0])
        assert not d.add_if_new([0.99, 0.05, 0.0])   # cosine ~0.999
        assert d.add_if_new([0.0, 1.0, 0.0])
        assert not d.add_if_new([2.0, 0.0, 0.0])     # scale does not matter
        assert len(d) == 2

    def test_zero_vector_kept(self):
        d = SemanticDeduplicator()
        assert d.add_if_new([0.0, 0.0]) and d.add_if_new([0.0, 0.0])

    def test_grows_past_initial_capacity(self):
        d = SemanticDeduplicator(0.99)
        basis = [[1.0 if j == i else 0.0 for j in range(200)] for i in range(200)]
        assert all(d.add_if_new(v) for v in basis)
        assert len(d) == 200 and not d.add_if_new(basis[150])
