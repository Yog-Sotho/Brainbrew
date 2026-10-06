"""
PII detection and redaction.

Regex detectors are validated to keep false positives out of the training
data:

* card numbers must pass the Luhn check and IBANs the mod-97 check;
* IPv4/IPv6 addresses are skipped in version-number or OID contexts and when
  they are loopback or unspecified addresses;
* URLs follow a policy: ``domain`` keeps ``scheme://host`` (the default),
  ``redact`` replaces the whole URL, ``keep`` leaves it. Credentials in a URL
  are always removed, and so are secret-looking query values.

An optional Presidio backend adds NER-based entities (names, passport and
licence numbers, ...) on top of the regex detectors.
"""
from __future__ import annotations

import functools
import importlib.util
import ipaddress
import re
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlsplit

UrlPolicy = Literal["domain", "redact", "keep"]

# Cheap pre-check: every detector needs one of these.
_CANDIDATE_RE = re.compile(r'[@0-9+]|http|www\.', re.IGNORECASE)


# ============================================================================
# Validators
# ============================================================================
def luhn_valid(digits: str) -> bool:
    """True if *digits* (13-19 digits) passes the Luhn checksum."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """True if *iban* (spaces allowed) passes the ISO 13616 mod-97 check."""
    s = iban.replace(' ', '').upper()
    if not 15 <= len(s) <= 34 or not s[:2].isalpha() or not s[2:4].isdigit() or not s.isalnum():
        return False
    rearranged = s[4:] + s[:4]
    return int(''.join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


# ============================================================================
# Detectors
# ============================================================================
def _last(digits: str, n: int = 4) -> str:
    return digits[-n:] if len(digits) >= n else '*' * n


def _digits(s: str) -> str:
    return re.sub(r'\D', '', s)


def _email(m: re.Match[str], mask: bool) -> str:
    if not mask:
        return '[PII_EMAIL]'
    local, domain = m.group(0).split('@', 1)
    masked = '***' if len(local) <= 1 else local[0] + '***' + local[-1]
    return f"{masked}@{domain}"


def _card(m: re.Match[str], mask: bool) -> str | None:
    digits = _digits(m.group(0))
    if not luhn_valid(digits):
        return None
    return f"****-****-****-{_last(digits)}" if mask else '[PII_CARD]'


def _iban(m: re.Match[str], mask: bool) -> str | None:
    raw = m.group(0)
    if not iban_valid(raw):
        return None
    compact = raw.replace(' ', '')
    return f"{compact[:2]}**-****-{_last(compact)}" if mask else '[PII_IBAN]'


def _ssn(m: re.Match[str], mask: bool) -> str:
    return f"***-**-{_last(_digits(m.group(0)))}" if mask else '[PII_SSN]'


def _phone(m: re.Match[str], mask: bool) -> str:
    return f"***-***-{_last(_digits(m.group(0)))}" if mask else '[PII_PHONE]'


# "v1.2", "version 10.0.0.1", "release 4.3.2.1": software versions, not addresses.
_VERSION_CONTEXT_RE = re.compile(r'(?:\bv|\bversion|\bver\.?|\brelease|\bbuild|\bfirmware|\brev\.?)\s*$', re.IGNORECASE)


def _ip_is_private_data(text: str, m: re.Match[str]) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(m.group(0))
    except ValueError:
        return None
    if ip.is_loopback or ip.is_unspecified or ip.is_link_local:
        return None
    if _VERSION_CONTEXT_RE.search(text[max(0, m.start() - 12):m.start()]):
        return None
    return ip


def _ipv4(text: str) -> Callable[[re.Match[str], bool], str | None]:
    def replace(m: re.Match[str], mask: bool) -> str | None:
        ip = _ip_is_private_data(text, m)
        if ip is None:
            return None
        if mask:
            a, b, *_ = m.group(0).split('.')
            return f"{a}.{b}.***.***"
        return '[PII_IP]'
    return replace


def _ipv6(text: str) -> Callable[[re.Match[str], bool], str | None]:
    def replace(m: re.Match[str], mask: bool) -> str | None:
        # Needs 3+ colons, so code such as `a[1::2]` or `fe80::1` is not an address here.
        if m.group(0).count(':') < 3 or _ip_is_private_data(text, m) is None:
            return None
        return '[PII_IP]'
    return replace


_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_CARD_RE = re.compile(r'(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])')
_IBAN_RE = re.compile(r'\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b')
_SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_PHONE_RES = (
    re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    re.compile(r'\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}\b'),
)
_IPV4_RE = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d)')
_IPV6_RE = re.compile(r'(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])')
_URL_RE = re.compile(r'(?:https?://|www\.)[^\s<>"\'`]+', re.IGNORECASE)

_Replacer = Callable[[re.Match[str], bool], str | None]


def _apply(pattern: re.Pattern[str], replace: _Replacer, text: str, mask: bool) -> tuple[str, int]:
    count = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal count
        out = replace(m, mask)
        if out is None:
            return m.group(0)
        count += 1
        return out

    return pattern.sub(sub, text), count


def _redact_plain(text: str, mask: bool) -> tuple[str, int]:
    """Every detector except URLs, in an order where longer patterns win."""
    found = 0
    detectors: list[tuple[re.Pattern[str], _Replacer]] = [
        (_EMAIL_RE, _email),
        (_IBAN_RE, _iban),
        (_CARD_RE, _card),
        (_SSN_RE, _ssn),
        *((p, _phone) for p in _PHONE_RES),
    ]
    for pattern, replace in detectors:
        text, n = _apply(pattern, replace, text, mask)
        found += n
    # IP detectors look at the surrounding text, so they bind to the current string.
    for pattern, factory in ((_IPV4_RE, _ipv4), (_IPV6_RE, _ipv6)):
        text, n = _apply(pattern, factory(text), text, mask)
        found += n
    return text, found


# ============================================================================
# URLs
# ============================================================================
_URL_TRAILING = '.,;:!?\'"]}>'
_SECRET_PARAM_RE = re.compile(
    r'([?&][^=&#]*(?:token|key|secret|pass(?:word)?|pwd|auth|sig(?:nature)?|session)[^=&#]*=)[^&#]*',
    re.IGNORECASE,
)


def _trim_url(url: str) -> tuple[str, str]:
    """Split trailing sentence punctuation (and unbalanced ')') off a URL."""
    end = len(url)
    while end:
        ch = url[end - 1]
        if ch in _URL_TRAILING or (ch == ')' and url[:end].count('(') < url[:end].count(')')):
            end -= 1
        else:
            break
    return url[:end], url[end:]


def _handle_url(url: str, policy: UrlPolicy, mask: bool) -> tuple[str, bool]:
    """Apply the URL policy. Returns (replacement, changed)."""
    core, tail = _trim_url(url)
    if policy == 'redact':
        return '[PII_URL]' + tail, True
    has_scheme = core.lower().startswith(('http://', 'https://'))
    try:
        parts = urlsplit(core if has_scheme else 'http://' + core)
        host = parts.hostname or ''
        port = parts.port
    except ValueError:
        return '[PII_URL]' + tail, True
    if not host:
        return '[PII_URL]' + tail, True
    host_found = False
    try:
        ip = ipaddress.ip_address(host)
        if not (ip.is_loopback or ip.is_unspecified or ip.is_link_local):
            host, host_found = '[PII_IP]', True
    except ValueError:
        pass
    if ':' in host:  # IPv6 literal: urlsplit drops the brackets
        host = f"[{host}]"
    netloc = host + (f":{port}" if port else '')
    prefix = f"{parts.scheme}://" if has_scheme else ''

    if policy == 'domain':
        out = prefix + netloc
        return out + tail, out != core or host_found

    # keep: everything except credentials, secret-looking query values and emails
    rest = core.split(parts.netloc, 1)[1] if parts.netloc else ''
    rest = _SECRET_PARAM_RE.sub(r'\1[REDACTED]', rest)
    # Paths are full of ids that look like phone numbers; only emails count here.
    rest, n = _apply(_EMAIL_RE, _email, rest, mask)
    out = prefix + netloc + rest
    return out + tail, out != core or host_found or n > 0


# ============================================================================
# Optional Presidio backend
# ============================================================================
PRESIDIO_ENTITIES = (
    "PERSON", "US_PASSPORT", "US_DRIVER_LICENSE", "US_BANK_NUMBER",
    "UK_NHS", "MEDICAL_LICENSE", "CRYPTO", "IBAN_CODE",
)
PRESIDIO_MIN_SCORE = 0.6


def presidio_available() -> bool:
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        return False
    return True


# Largest installed English model wins.
SPACY_MODELS = ("en_core_web_lg", "en_core_web_md", "en_core_web_sm")


def _spacy_model() -> str | None:
    return next((m for m in SPACY_MODELS if importlib.util.find_spec(m) is not None), None)


@functools.cache
def _presidio_engine() -> Any:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise RuntimeError(
            "The Presidio PII backend is not installed. Install it with "
            "`pip install 'brainbrew[pii]'` and `python -m spacy download en_core_web_lg`."
        ) from exc
    model = _spacy_model()
    if model is None:
        raise RuntimeError(
            "The Presidio PII backend needs a spaCy English model: "
            "`python -m spacy download en_core_web_lg` (or en_core_web_sm)."
        )
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}],
    }).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])


def _redact_presidio(text: str) -> tuple[str, int]:
    results = _presidio_engine().analyze(
        text=text, language="en", entities=list(PRESIDIO_ENTITIES), score_threshold=PRESIDIO_MIN_SCORE,
    )
    spans: list[tuple[int, int, str]] = []
    last_end = -1
    for r in sorted(results, key=lambda r: (r.start, -r.end)):
        if r.start >= last_end:  # skip overlaps; the earlier, longer span wins
            spans.append((r.start, r.end, r.entity_type))
            last_end = r.end
    for start, end, entity in reversed(spans):
        text = f"{text[:start]}[PII_{entity}]{text[end:]}"
    return text, len(spans)


# ============================================================================
# Entry point
# ============================================================================
def redact_pii(
    text: str,
    mask: bool = False,
    url_policy: UrlPolicy = "domain",
    presidio: bool = False,
) -> tuple[str, bool]:
    """Redact (or partially mask) PII in *text*.

    Returns (cleaned_text, pii_was_found).
    """
    found = 0
    if _CANDIDATE_RE.search(text):
        pieces: list[str] = []
        pos = 0
        for m in _URL_RE.finditer(text):
            plain, n = _redact_plain(text[pos:m.start()], mask)
            url, changed = _handle_url(m.group(0), url_policy, mask)
            pieces += [plain, url]
            found += n + changed
            pos = m.end()
        plain, n = _redact_plain(text[pos:], mask)
        pieces.append(plain)
        found += n
        text = ''.join(pieces)
    if presidio:
        text, n = _redact_presidio(text)
        found += n
    return text, found > 0
