"""
tests/test_decontam.py

Benchmark decontamination (Phase 2.6): 13-gram overlap against eval sets the
user selected. Eval sets are injected, so no network is needed.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from pipeline import decontam
from pipeline.decontam import EVAL_SETS, EvalIndex, decontaminate, load_eval_texts
from pipeline.records import Record

GSM = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
       "clips in May. How many clips did Natalia sell altogether in April and May?")
SHORT = "Which gas do plants absorb from the air during photosynthesis?"  # 10 words


def _index(*texts: str) -> EvalIndex:
    index = EvalIndex()
    for t in texts:
        index.add(t)
    return index


class TestEvalIndex:

    def test_long_overlap_detected_despite_case_and_punctuation(self):
        index = _index(GSM)
        assert index.overlaps("Problem: natalia SOLD clips to 48 of her friends in april, and then she sold half")

    def test_short_overlap_is_not_enough(self):
        assert not _index(GSM).overlaps("Natalia sold clips to 48 of her friends.")

    def test_short_items_match_whole(self):
        index = _index(SHORT)
        assert index.overlaps(f"Quiz: {SHORT} Answer: carbon dioxide.")
        assert not index.overlaps("Which gas do plants absorb?")

    def test_very_short_items_ignored(self):
        assert len(_index("What is 2 + 2?")) == 0

    def test_unrelated_text(self):
        assert not _index(GSM, SHORT).overlaps(
            "The French Revolution began in 1789 and ended the absolute monarchy of Louis XVI."
        )


def _rec(q: str, a: str = "A sufficiently long answer about the topic at hand.") -> Record:
    return Record(instruction=q, output=a)


class TestDecontaminate:

    def test_removes_and_counts_per_set(self):
        sets = {"gsm8k": [GSM], "arc": [SHORT]}
        recs = [_rec(GSM), _rec("What is photosynthesis?", SHORT), _rec("Explain tectonic plates.")]
        kept, removed = decontaminate(recs, ["gsm8k", "arc"], loader=sets.__getitem__)
        assert kept == [recs[2]]
        assert removed == {"gsm8k": 1, "arc": 1}

    def test_no_sets_keeps_everything(self):
        recs = [_rec(GSM)]
        assert decontaminate(recs, [], loader=lambda k: []) == (recs, {})


class TestLoading:

    def test_fields_are_flattened(self):
        rows = [
            {"question": "Q one is here", "choices": {"text": ["alpha", "beta"], "label": ["A", "B"]}},
            {"question": "", "choices": {"text": []}},
        ]
        with patch("datasets.load_dataset", return_value=rows) as load:
            texts = load_eval_texts("arc")
        load.assert_called_once_with("allenai/ai2_arc", "ARC-Challenge", split="test")
        assert texts == ["Q one is here alpha beta"]

    def test_download_errors_are_explained(self):
        with patch("datasets.load_dataset", side_effect=ConnectionError("offline")), \
             pytest.raises(RuntimeError, match=r"Could not download the GSM8K benchmark.*offline"):
            load_eval_texts("gsm8k")

    def test_every_set_is_described(self):
        for spec in EVAL_SETS.values():
            assert spec.repo.count("/") == 1 and spec.fields and spec.split

    def test_build_index_uses_loader(self):
        index = decontam.build_index("gsm8k", loader=lambda k: [GSM])
        assert index.overlaps(GSM)
