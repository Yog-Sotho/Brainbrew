"""
tests/test_pii.py

PII v2 (Phase 2.7): validated detectors (Luhn, IBAN mod-97, IP context), the
URL policy and the optional Presidio backend.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from pipeline import pii
from pipeline.pii import iban_valid, luhn_valid, redact_pii


def _redact(text: str, **kw) -> str:
    return redact_pii(text, **kw)[0]


class TestValidators:

    @pytest.mark.parametrize("number,ok", [
        ("4111111111111111", True),   # Visa test number
        ("5500005555555559", True),   # Mastercard test number
        ("378282246310005", True),    # Amex test number
        ("4111111111111112", False),
        ("1234567890123456", False),
        ("123", False),
    ])
    def test_luhn(self, number, ok):
        assert luhn_valid(number) is ok

    @pytest.mark.parametrize("iban,ok", [
        ("GB82 WEST 1234 5698 7654 32", True),
        ("DE89370400440532013000", True),
        ("GB82 WEST 1234 5698 7654 33", False),
        ("XX00", False),
    ])
    def test_iban(self, iban, ok):
        assert iban_valid(iban) is ok


class TestNumbers:

    def test_valid_card_redacted(self):
        assert _redact("Card: 4111 1111 1111 1111.") == "Card: [PII_CARD]."
        assert _redact("Card: 4111-1111-1111-1111", mask=True) == "Card: ****-****-****-1111"

    def test_luhn_invalid_number_kept(self):
        # Order numbers, ISBN-like ids and long counts are not card numbers.
        text = "Order 1234 5678 9012 3456 shipped."
        assert _redact(text) == text

    def test_iban(self):
        assert _redact("Pay to GB82 WEST 1234 5698 7654 32 today") == "Pay to [PII_IBAN] today"
        assert _redact("IBAN DE89370400440532013000", mask=True) == "IBAN DE**-****-3000"

    def test_ssn_and_phone(self):
        assert _redact("SSN 123-45-6789") == "SSN [PII_SSN]"
        assert _redact("Call +44 20 7946 0958 now") == "Call [PII_PHONE] now"


class TestIpAddresses:

    @pytest.mark.parametrize("text", [
        "Connect to 203.0.113.7 over SSH.",
        "The office range is 10.20.30.40.",
    ])
    def test_addresses_redacted(self, text):
        out, found = redact_pii(text)
        assert found and "[PII_IP]" in out

    @pytest.mark.parametrize("text", [
        "Upgrade to version 10.2.1.4 first.",
        "Requires v1.2.3.4 or later.",
        "Release 4.3.2.1 fixed the bug.",
        "The OID 1.3.6.1.4.1 identifies the vendor.",
        "The server listens on 127.0.0.1 and 0.0.0.0.",
        "Not an address: 999.1.1.1.",
    ])
    def test_non_addresses_kept(self, text):
        assert _redact(text) == text

    def test_ipv6(self):
        assert _redact("Host 2001:db8:85a3::8a2e:370:7334 is up") == "Host [PII_IP] is up"
        for text in ("Slice with a[1::2].", "Loopback ::1 only.", "Meet at 12:30:45."):
            assert _redact(text) == text

    def test_masking(self):
        assert _redact("Connecting to 192.168.1.1.", mask=True) == "Connecting to 192.168.***.***."


class TestUrlPolicy:

    URL = "See https://docs.example.com/guide/install?ref=nav#step-2."

    def test_domain_is_the_default(self):
        out, found = redact_pii(self.URL)
        assert out == "See https://docs.example.com." and found

    def test_bare_domain_is_not_pii(self):
        text = "Visit https://example.com for details."
        assert redact_pii(text) == (text, False)

    def test_redact(self):
        assert _redact(self.URL, url_policy="redact") == "See [PII_URL]."

    def test_keep(self):
        assert redact_pii(self.URL, url_policy="keep") == (self.URL, False)

    @pytest.mark.parametrize("policy,expected", [
        ("domain", "Clone https://git.example.com"),
        ("keep", "Clone https://git.example.com/repo.git"),
    ])
    def test_credentials_always_removed(self, policy, expected):
        out, found = redact_pii("Clone https://alice:hunter2@git.example.com/repo.git", url_policy=policy)
        assert out == expected and found

    def test_keep_removes_secret_query_values_and_emails(self):
        out = _redact("https://api.example.com/v1?api_key=abc123&user=bob@example.com&page=2", url_policy="keep")
        assert out == "https://api.example.com/v1?api_key=[REDACTED]&user=[PII_EMAIL]&page=2"

    def test_ip_host(self):
        assert _redact("Open http://203.0.113.9:8080/admin") == "Open http://[PII_IP]:8080"
        assert _redact("Open http://127.0.0.1:8501/") == "Open http://127.0.0.1:8501"

    def test_trailing_punctuation_and_parentheses(self):
        text = "(see https://en.wikipedia.org/wiki/Python_(programming_language))."
        assert _redact(text, url_policy="keep") == text
        assert _redact(text) == "(see https://en.wikipedia.org)."

    def test_www_without_scheme(self):
        assert _redact("Go to www.example.org/a/b, then") == "Go to www.example.org, then"

    def test_url_digits_are_not_phones(self):
        text = "https://example.com/item/1234567890"
        assert _redact(text, url_policy="keep") == text


class TestEmails:

    def test_email(self):
        assert _redact("Mail test@example.com.") == "Mail [PII_EMAIL]."
        assert _redact("Mail test@example.com.", mask=True) == "Mail t***t@example.com."

    def test_plain_text_untouched(self):
        text = "Photosynthesis converts light into chemical energy."
        assert redact_pii(text) == (text, False)


@dataclass
class _Result:
    entity_type: str
    start: int
    end: int
    score: float = 0.9


class _FakeAnalyzer:
    def __init__(self, nlp_engine, supported_languages):
        assert nlp_engine == ("spacy", "en_core_web_sm") and supported_languages == ["en"]

    def analyze(self, text, language, entities, score_threshold):
        assert language == "en" and "PERSON" in entities
        out = []
        for name in ("Ada Lovelace", "Lovelace"):
            i = text.find(name)
            if i >= 0:
                out.append(_Result("PERSON", i, i + len(name)))
        return out


class _FakeProvider:
    def __init__(self, nlp_configuration):
        self.config = nlp_configuration

    def create_engine(self):
        (model,) = self.config["models"]
        return self.config["nlp_engine_name"], model["model_name"]


class TestPresidio:

    @pytest.fixture()
    def fake_presidio(self, monkeypatch):
        module = types.ModuleType("presidio_analyzer")
        module.AnalyzerEngine = _FakeAnalyzer  # type: ignore[attr-defined]
        nlp = types.ModuleType("presidio_analyzer.nlp_engine")
        nlp.NlpEngineProvider = _FakeProvider  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
        monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp)
        monkeypatch.setattr(pii, "_spacy_model", lambda: "en_core_web_sm")
        pii._presidio_engine.cache_clear()
        yield
        pii._presidio_engine.cache_clear()

    def test_entities_added_on_top_of_regex(self, fake_presidio):
        out, found = redact_pii("Ada Lovelace wrote to ada@example.com.", presidio=True)
        assert out == "[PII_PERSON] wrote to [PII_EMAIL]." and found
        assert pii.presidio_available()

    def test_off_by_default(self, fake_presidio):
        assert _redact("Ada Lovelace") == "Ada Lovelace"

    def test_missing_backend_is_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
        pii._presidio_engine.cache_clear()
        assert not pii.presidio_available()
        with pytest.raises(RuntimeError, match=r"brainbrew\[pii\]"):
            redact_pii("Ada Lovelace", presidio=True)
        pii._presidio_engine.cache_clear()

    def test_missing_spacy_model_is_a_clear_error(self, fake_presidio, monkeypatch):
        monkeypatch.setattr(pii, "_spacy_model", lambda: None)
        with pytest.raises(RuntimeError, match="spacy download"):
            redact_pii("Ada Lovelace", presidio=True)
