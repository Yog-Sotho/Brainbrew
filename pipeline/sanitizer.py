"""
Brainbrew sanitizer — post-export dataset cleaning for LLM training.

Core sanitization functions extracted from the standalone LLM Dataset Sanitizer
PRO v2.8 script. Provides PII redaction, HTML/control-char cleaning, quality
filtering, and per-record sanitization — all as pure functions with no CLI
scaffolding or external I/O dependencies.

Usage from the orchestrator:
    from pipeline.sanitizer import sanitize_dataset, SanitizerConfig
    cfg = SanitizerConfig(remove_pii=True, clean_html=True, ...)
    stats = sanitize_dataset(input_path, output_path, cfg)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class SanitizerConfig:
    """Settings for the sanitization pass.

    Sensible defaults are tuned for Brainbrew's synthetic Alpaca-style data.
    Non-technical users just flip the toggle; power users can adjust thresholds.
    """
    remove_pii: bool = True
    pii_mask: bool = False
    clean_html: bool = True
    deduplicate: bool = True
    min_chars: int = 50
    max_chars: int = 20000
    min_words: int = 8
    min_unique_ratio: float = 0.25
    min_ascii_ratio: float = 0.85
    require_fields: List[str] = field(default_factory=lambda: ["instruction", "output"])
    max_depth: int = 100


@dataclass
class SanitizeStats:
    """Counters for the sanitization report."""
    total: int = 0
    kept: int = 0
    filtered_quality: int = 0
    filtered_require: int = 0
    deduplicated: int = 0
    pii_redacted: int = 0


# ============================================================================
# HTML stripping
# ============================================================================
_HTML_TAG_RE = re.compile(r'<[^>]+>')


def strip_html(text: str) -> str:
    """Remove HTML/XML tags, replacing them with a single space."""
    return _HTML_TAG_RE.sub(' ', text)


# ============================================================================
# PII patterns + masking (from Sanitizer v2.8 §1.7)
# ============================================================================
_PII_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            re.IGNORECASE,
        ),
        '[PII_EMAIL]', 'email',
    ),
    (re.compile(r'https?://\S+', re.IGNORECASE), '[PII_URL]', 'url'),
    (re.compile(r'www\.\S+', re.IGNORECASE), '[PII_URL]', 'url'),
    (
        re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        '[PII_PHONE]', 'phone',
    ),
    (
        re.compile(
            r'\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}\b'
        ),
        '[PII_PHONE]', 'phone',
    ),
    (
        re.compile(r'\b(?:\d{4}[ \-]?){3}\d{4}\b'),
        '[PII_CARD]', 'card',
    ),
    (
        re.compile(
            r'\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)\.){3}'
            r'(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)\b'
        ),
        '[PII_IP]', 'ip',
    ),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[PII_SSN]', 'ssn'),
]


def _mask_last_digits(digits: str, n: int = 4) -> str:
    """Return last *n* digits, or '****' if fewer than *n* present."""
    return digits[-n:] if len(digits) >= n else '*' * n


def _mask_email(m: re.Match[str]) -> str:
    """Mask email: keep first+last char of local part."""
    full = m.group(0)
    try:
        local, domain = full.split('@', 1)
        if len(local) <= 1:
            masked_local = '***'
        else:
            masked_local = local[0] + '***' + local[-1]
        return f"{masked_local}@{domain}"
    except ValueError:
        return '[PII_EMAIL]'


def _mask_phone(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"***-***-{_mask_last_digits(digits)}"


def _mask_card(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"****-****-****-{_mask_last_digits(digits)}"


def _mask_ip(m: re.Match[str]) -> str:
    parts = m.group(0).split('.')
    return f"{parts[0]}.{parts[1]}.***.***" if len(parts) == 4 else '[PII_IP]'


def _mask_ssn(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"***-**-{_mask_last_digits(digits)}"


_MASK_FN: Dict[str, Callable[[re.Match[str]], str]] = {
    'email': _mask_email,
    'phone': _mask_phone,
    'card':  _mask_card,
    'ip':    _mask_ip,
    'ssn':   _mask_ssn,
}


# ============================================================================
# Core cleaning functions
# ============================================================================
def redact_pii(text: str, mask: bool = False) -> Tuple[str, bool]:
    """Redact or mask PII in text.

    Args:
        text: Input string.
        mask: If True, use partial masking instead of full token replacement.

    Returns:
        Tuple of (cleaned_text, pii_was_found).
    """
    pii_found = False
    for pattern, token, kind in _PII_PATTERNS:
        if pattern.search(text):
            pii_found = True
            if mask and kind in _MASK_FN:
                text = pattern.sub(_MASK_FN[kind], text)
            else:
                text = pattern.sub(token, text)
    return text, pii_found


def clean_text(text: str, remove_html: bool = True) -> str:
    """Normalize unicode, strip HTML, remove control chars, collapse whitespace."""
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFKC', text)
    if remove_html:
        text = strip_html(text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _sanitize_value(
    v: Any,
    *,
    cfg: SanitizerConfig,
    _depth: int = 0,
) -> Tuple[Any, bool]:
    """Recursively sanitize a value (string, dict, or list).

    Returns (sanitized_value, pii_was_found).
    """
    if _depth > cfg.max_depth:
        return v, False

    if isinstance(v, str):
        cleaned = clean_text(v, cfg.clean_html)
        pii_found = False
        if cfg.remove_pii:
            cleaned, pii_found = redact_pii(cleaned, mask=cfg.pii_mask)
        return cleaned, pii_found

    if isinstance(v, dict):
        pii_found = False
        result = {}
        for k, val in v.items():
            sanitized_val, found = _sanitize_value(
                val, cfg=cfg, _depth=_depth + 1,
            )
            result[k] = sanitized_val
            pii_found = pii_found or found
        return result, pii_found

    if isinstance(v, list):
        pii_found = False
        result_list = []
        for item in v:
            sanitized_item, found = _sanitize_value(
                item, cfg=cfg, _depth=_depth + 1,
            )
            result_list.append(sanitized_item)
            pii_found = pii_found or found
        return result_list, pii_found

    return v, False


# ============================================================================
# Quality gates (from Sanitizer v2.8 quality checks)
# ============================================================================
_WORD_RE = re.compile(r'\b\w+\b')


def _extract_text_for_quality(record: Dict[str, Any], max_chars: int = 8192) -> str:
    """Extract text from record fields for quality scoring.

    Concatenates all string values up to max_chars budget.
    """
    parts: List[str] = []
    budget = max_chars
    for v in record.values():
        if budget <= 0:
            break
        if isinstance(v, str):
            chunk = v[:budget]
            parts.append(chunk)
            budget -= len(chunk)
    return ' '.join(parts)


def check_quality(text: str, cfg: SanitizerConfig) -> Optional[str]:
    """Check text against quality gates.

    Returns rejection reason string, or None if text passes all gates.
    """
    if not text:
        return 'empty text'
    if len(text) < cfg.min_chars:
        return f'too short ({len(text)} < {cfg.min_chars} chars)'
    if len(text) > cfg.max_chars:
        return f'too long ({len(text)} > {cfg.max_chars} chars)'
    words = _WORD_RE.findall(text)
    if len(words) < cfg.min_words:
        return f'too few words ({len(words)} < {cfg.min_words})'
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    if unique_ratio < cfg.min_unique_ratio:
        return f'low unique-word ratio ({unique_ratio:.3f} < {cfg.min_unique_ratio})'
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < cfg.min_ascii_ratio:
        return f'low ASCII ratio ({ascii_ratio:.3f} < {cfg.min_ascii_ratio})'
    return None


# ============================================================================
# Dedup hashing
# ============================================================================
def get_record_hash(record: Dict[str, Any], normalize: bool = True) -> str:
    """SHA-256 hash of a record for exact deduplication.

    Args:
        record: Dict to hash.
        normalize: If True, lowercase and collapse whitespace before hashing.
    """
    target: Any = record
    if normalize:
        def _norm(v: Any) -> Any:
            if isinstance(v, str):
                return re.sub(r'\s+', ' ', v.lower().strip())
            if isinstance(v, dict):
                return {k: _norm(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_norm(item) for item in v]
            return v
        target = _norm(target)
    serialized = json.dumps(
        target, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


# ============================================================================
# Per-record sanitizer
# ============================================================================
def sanitize_record(
    record: Dict[str, Any],
    cfg: SanitizerConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Sanitize a single record: clean, redact PII, check quality.

    Args:
        record: Raw dict from the exported JSONL.
        cfg: Sanitizer configuration.

    Returns:
        (sanitized_record, None) on success, or (None, rejection_reason) on failure.
    """
    if not isinstance(record, dict):
        return None, 'not a dict'

    # Require-fields gate (pre-sanitization — check structure)
    for rf in cfg.require_fields:
        v = record.get(rf)
        is_empty = (
            v is None
            or (isinstance(v, str) and not v.strip())
            or (not isinstance(v, (bool, int, float)) and not v)
        )
        if is_empty:
            return None, f"missing required field '{rf}'"

    # Sanitize all fields
    sanitized, pii_found = _sanitize_value(record, cfg=cfg)

    # Quality gate (post-sanitization — check content quality)
    quality_text = _extract_text_for_quality(sanitized)
    rejection = check_quality(quality_text, cfg)
    if rejection is not None:
        return None, rejection

    return sanitized, None


# ============================================================================
# Dataset-level entry point
# ============================================================================
def sanitize_dataset(
    input_path: Path,
    output_path: Path,
    cfg: Optional[SanitizerConfig] = None,
) -> SanitizeStats:
    """Sanitize an entire JSONL dataset file.

    Reads input_path line by line, sanitizes each record, deduplicates,
    and writes passing records to output_path. Returns statistics.

    Args:
        input_path: Path to the exported JSONL file.
        output_path: Path for the sanitized output JSONL file.
        cfg: Sanitizer configuration (uses defaults if None).

    Returns:
        SanitizeStats with counts of total, kept, filtered, deduplicated records.
    """
    if cfg is None:
        cfg = SanitizerConfig()

    stats = SanitizeStats()
    seen_hashes: Set[str] = set()

    with open(input_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            stats.total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.filtered_quality += 1
                continue

            sanitized, rejection = sanitize_record(record, cfg)

            if sanitized is None:
                if rejection and 'missing required field' in rejection:
                    stats.filtered_require += 1
                else:
                    stats.filtered_quality += 1
                continue

            # Deduplication
            if cfg.deduplicate:
                h = get_record_hash(sanitized)
                if h in seen_hashes:
                    stats.deduplicated += 1
                    continue
                seen_hashes.add(h)

            # Track PII redaction
            _, pii_found = _sanitize_value(record, cfg=SanitizerConfig(
                remove_pii=False, clean_html=False, max_depth=0,
            ))
            # Check original vs sanitized for PII changes
            if cfg.remove_pii:
                original_text = ' '.join(
                    str(v) for v in record.values() if isinstance(v, str)
                )
                for pattern, _, _ in _PII_PATTERNS:
                    if pattern.search(original_text):
                        stats.pii_redacted += 1
                        break

            fout.write(json.dumps(sanitized, ensure_ascii=False) + '\n')
            stats.kept += 1

    logger.info(
        "Sanitization complete: %d total, %d kept, %d quality-filtered, "
        "%d require-filtered, %d deduplicated, %d PII-redacted",
        stats.total, stats.kept, stats.filtered_quality,
        stats.filtered_require, stats.deduplicated, stats.pii_redacted,
    )

    return stats
