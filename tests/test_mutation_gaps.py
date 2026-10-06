"""
tests/test_mutation_gaps.py

Behaviour that mutation testing (mutmut, Phase 4.2) showed no test pinned down
in the sanitizer, PII detection and dedup. Each test kills a group of
surviving mutants.
"""
from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from pipeline import pii
from pipeline.dedup import SemanticDeduplicator
from pipeline.pii import iban_valid, luhn_valid, redact_pii
from pipeline.records import Record
from pipeline.sanitizer import (
    SanitizerConfig,
    _extract_text_for_quality,
    _sanitize_record_internal,
    check_quality,
    clean_text,
    get_record_hash,
    sanitize_dataset,
    sanitize_records,
)

LONG = "Plants turn sunlight, water and carbon dioxide into sugar and oxygen in their leaves every day."


def _rec(instruction: str = "What does photosynthesis produce?", output: str = LONG, **kw) -> Record:
    return Record(instruction=instruction, output=output, **kw)


class TestRedactPii:

    def test_text_between_several_urls_is_kept(self):
        text = "First https://a.example.com/x then mail bob@example.com and https://b.example.org/y end."
        out, found = redact_pii(text)
        assert out == "First https://a.example.com then mail [PII_EMAIL] and https://b.example.org end."
        assert found

    def test_a_shortened_url_alone_counts_as_found(self):
        out, found = redact_pii("see https://example.com/private/path")
        assert out == "see https://example.com" and found

    def test_masking_applies_inside_and_around_urls(self):
        out, _ = redact_pii("mail ann@example.com at https://x.example.com/?u=bob@example.com",
                            mask=True, url_policy="keep")
        assert out == "mail a***n@example.com at https://x.example.com/?u=b***b@example.com"

    def test_redact_policy_reports_found(self):
        assert redact_pii("go to https://example.com", url_policy="redact") == ("go to [PII_URL]", True)

    @pytest.mark.parametrize("url", ["http://[::1", "http:///only/a/path"])
    def test_unparseable_or_hostless_urls_are_removed(self, url):
        out, found = redact_pii(f"x {url} y")
        assert out == "x [PII_URL] y" and found


class TestMasks:

    def test_one_character_email_local_part(self):
        assert redact_pii("a@example.com", mask=True)[0] == "***@example.com"
        assert redact_pii("ab@example.com", mask=True)[0] == "a***b@example.com"

    def test_ssn_mask(self):
        assert redact_pii("SSN 123-45-6789", mask=True)[0] == "SSN ***-**-6789"

    def test_short_digit_runs_are_starred(self):
        assert pii._last("12") == "****" and pii._last("1234") == "1234" and pii._last("12345") == "2345"


class TestValidators:

    @pytest.mark.parametrize(("number", "ok"), [
        ("4222222222222", True),          # 13-digit Visa test number: the minimum length
        ("4111111111111111111", False),   # 19 digits, fails the checksum
        ("1" * 12, False),                # too short even if it passed
        ("4111111111111111x", False),     # not all digits
        ("79927398713" + "00000000", True),  # 19 digits, valid
        ("799273987130000000000", False),    # 21 digits
    ])
    def test_luhn_lengths(self, number, ok):
        assert luhn_valid(number) is ok

    @pytest.mark.parametrize(("iban", "ok"), [
        ("NO9386011117947", True),                # 15 characters, the shortest real IBAN
        ("NO938601111794", False),                # 14
        ("12820000123456789012345", False),       # no country code
        ("GBXX WEST 1234 5698 7654 32", False),   # no check digits
        ("GB82-WEST-1234-5698-7654-32", False),   # not alphanumeric
        ("gb82 west 1234 5698 7654 32", True),    # case does not matter
        ("GB82" + "1" * 31, False),               # 35 characters
    ])
    def test_iban_rules(self, iban, ok):
        assert iban_valid(iban) is ok


class TestCleanText:

    def test_old_mac_line_endings(self):
        assert clean_text("one\rtwo\r\nthree") == "one\ntwo\nthree"

    def test_runs_of_spaces_collapse_but_indentation_stays(self):
        assert clean_text("a  b\t c\n    indented") == "a b c\n    indented"

    def test_html_kept_when_asked(self):
        assert clean_text("<b>bold</b>", remove_html=False) == "<b>bold</b>"
        assert clean_text("<b>bold</b>") == "bold"


class TestQuality:

    def test_length_limits_are_inclusive(self):
        cfg = SanitizerConfig(min_chars=10, max_chars=60, min_words=2, min_unique_ratio=0.0, min_ascii_ratio=0.0)
        text = "word " * 12  # 60 characters, trimmed by nobody
        assert check_quality(text[:60], cfg) is None
        assert check_quality(text[:60] + "x", cfg) is not None

    def test_empty_text(self):
        assert check_quality("", SanitizerConfig()) == "empty text"

    def test_quality_text_is_capped(self):
        record = {"instruction": "a" * 50, "output": "b" * 50, "input": "c" * 50}
        text = _extract_text_for_quality(record, max_chars=60)
        assert len(text) <= 61 and text.startswith("a" * 50)
        assert _extract_text_for_quality({"instruction": "q", "output": "a"}) == "q a"


class TestSanitizeRecords:

    def test_counts_and_options_reach_the_records(self):
        recs = [
            _rec(output=LONG + " Contact ann@example.com <b>today</b>."),
            _rec(instruction="Where can I read more about this topic in depth?",
                 output=LONG + " See https://docs.example.com/guide/page for details."),
            _rec(instruction="Why do leaves look green to our eyes?", output=LONG + " Chlorophyll reflects green."),
        ]
        kept, stats = sanitize_records(recs, SanitizerConfig(min_chars=10, pii_mask=True, url_policy="redact",
                                                              clean_html=False))
        assert stats.total == stats.kept == 3 and stats.pii_redacted == 2
        assert "a***n@example.com <b>today</b>" in kept[0].output
        assert "[PII_URL]" in kept[1].output
        assert kept[2].output == recs[2].output  # clean record untouched

    def test_missing_fields_are_counted_per_record(self):
        good = {"instruction": "What is it?", "output": LONG}
        bad = {"instruction": "What is it?", "output": "  "}
        assert _sanitize_record_internal(bad, SanitizerConfig(min_chars=10))[1] == "missing required field 'output'"
        assert _sanitize_record_internal(["not", "a", "dict"], SanitizerConfig()) == (None, "not a dict", False)  # type: ignore[arg-type]
        assert _sanitize_record_internal(good, SanitizerConfig(min_chars=10))[1] is None

    def test_counts_with_several_rejects_and_duplicates(self):
        dup = _rec()
        recs = [
            _rec(instruction="Q?", output="too short"),       # quality reject first: processing must go on
            dup, dup.model_copy(), dup.model_copy(),          # two duplicates of the first copy
            _rec(instruction="Q2?", output="tiny"),           # second quality reject
            _rec(instruction="<b> </b>", output=LONG),       # cleaning empties the instruction
            _rec(instruction="<i>\t</i>", output=LONG),      # and again
            _rec(instruction="Why do leaves look green to our eyes?", output=LONG + " Chlorophyll."),
        ]
        kept, stats = sanitize_records(recs, SanitizerConfig(min_chars=40))
        assert [r.instruction for r in kept] == [dup.instruction, "Why do leaves look green to our eyes?"]
        assert (stats.total, stats.kept, stats.deduplicated) == (8, 2, 2)
        assert stats.filtered_quality + stats.filtered_require == 4
        assert stats.filtered_quality >= 2 and stats.filtered_require == 4 - stats.filtered_quality

    def test_default_config(self):
        kept, stats = sanitize_records([_rec(output=LONG + " Write to ann@example.com.")])
        assert stats.pii_redacted == 1 and "[PII_EMAIL]" in kept[0].output

    def test_nested_values_up_to_max_depth(self):
        cfg = SanitizerConfig(max_depth=2, min_chars=1)
        shallow = {"a": {"b": "mail ann@example.com"}}
        deep = {"a": {"b": {"c": "mail ann@example.com"}}}
        from pipeline.sanitizer import _sanitize_value

        assert _sanitize_value(shallow, cfg=cfg) == ({"a": {"b": "mail [PII_EMAIL]"}}, True)
        assert _sanitize_value(deep, cfg=cfg) == (deep, False)  # deeper than max_depth: left alone
        assert _sanitize_value(["x", "ann@example.com"], cfg=cfg) == (["x", "[PII_EMAIL]"], True)
        assert _sanitize_value(7, cfg=cfg) == (7, False)


class TestSanitizeDataset:

    def test_file_entry_point_with_default_config(self, tmp_path: Path):
        src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
        rows = [
            {"instruction": "What does photosynthesis produce?", "output": LONG + " ann@example.com"},
            {"instruction": "What does photosynthesis produce?", "output": LONG + " ann@example.com"},
            {"instruction": "x", "output": ""},
            "not json",
        ]
        src.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        stats = sanitize_dataset(src, dst)
        out = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
        assert len(out) == 1 and "[PII_EMAIL]" in out[0]["output"]
        assert stats.kept == 1 and stats.deduplicated == 1 and stats.pii_redacted >= 1


class TestRecordHash:

    def test_case_whitespace_and_key_order_do_not_matter(self):
        a = {"instruction": "What  is   it?", "output": "Plants", "tags": ["A b", "c"]}
        b = {"tags": ["a  B", "C"], "output": "plants", "instruction": "what is it?"}
        assert get_record_hash(a) == get_record_hash(b)

    def test_content_does_matter(self):
        assert get_record_hash({"output": "plants"}) != get_record_hash({"output": "animals"})
        assert get_record_hash({"tags": ["a", "b"]}) != get_record_hash({"tags": ["a", "c"]})

    def test_without_normalisation_case_matters(self):
        assert get_record_hash({"o": "A"}, normalize=False) != get_record_hash({"o": "a"}, normalize=False)

    def test_non_ascii_is_hashed_as_text(self):
        assert get_record_hash({"o": "café"}) == get_record_hash({"o": "CAFÉ"})


class TestSemanticDedup:

    def test_stored_vectors_are_unit_length(self):
        d = SemanticDeduplicator(0.9)
        assert d.add_if_new([3.0, 0.0, 0.0])
        # Cosine 0.8 with the first vector: different enough, whatever the first vector's length.
        assert d.add_if_new([0.8, 0.6, 0.0])

    def test_integer_vectors(self):
        d = SemanticDeduplicator(0.99)
        assert d.add_if_new([1, 0]) and not d.add_if_new([5, 0]) and d.add_if_new([0, 7])

    def test_threshold_is_inclusive(self):
        d = SemanticDeduplicator(0.8)
        assert d.add_if_new([1.0, 0.0])
        assert not d.add_if_new([0.8, 0.6])  # cosine exactly 0.8


@dataclass
class _Result:
    entity_type: str
    start: int
    end: int


class TestPresidioDetails:

    def test_overlapping_spans_keep_the_earlier_longer_one(self, monkeypatch):
        captured = {}

        class Engine:
            def analyze(self, text, language, entities, score_threshold):
                captured["threshold"] = score_threshold
                return [_Result("PERSON", 0, 12), _Result("PERSON", 4, 12), _Result("LOCATION", 13, 19)]

        monkeypatch.setattr(pii, "_presidio_engine", lambda: Engine())
        out, n = pii._redact_presidio("Ada Lovelace London")
        assert out == "[PII_PERSON] [PII_LOCATION]" and n == 2
        assert captured["threshold"] == pii.PRESIDIO_MIN_SCORE

    def test_adjacent_spans_are_both_redacted(self, monkeypatch):
        class Engine:
            def analyze(self, **_):
                return [_Result("PERSON", 0, 3), _Result("PERSON", 3, 6)]

        monkeypatch.setattr(pii, "_presidio_engine", lambda: Engine())
        assert pii._redact_presidio("AdaBob") == ("[PII_PERSON][PII_PERSON]", 2)

    def test_largest_installed_spacy_model_wins(self, monkeypatch):
        installed = {"en_core_web_md", "en_core_web_sm"}
        monkeypatch.setattr(pii.importlib.util, "find_spec",
                            lambda name: types.SimpleNamespace() if name in installed else None)
        assert pii._spacy_model() == "en_core_web_md"
        installed.clear()
        assert pii._spacy_model() is None

    def test_presidio_available_reflects_the_import(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "presidio_analyzer", types.ModuleType("presidio_analyzer"))
        assert pii.presidio_available()
