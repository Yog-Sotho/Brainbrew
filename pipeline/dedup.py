"""
Near-duplicate detection with MinHash + LSH (Phase 2.5).

Each record (instruction + output, normalised) is turned into character
5-gram shingles, summarised by a 128-value MinHash signature and bucketed by
LSH banding. Only records that share a bucket are compared, so the cost grows
roughly linearly with the dataset instead of quadratically.

`Deduplicator` is incremental: generation rounds add accepted records one at a
time and ask whether a candidate duplicates anything kept so far.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import xxhash

from pipeline.records import Record

NUM_PERM = 128
SHINGLE = 5
_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_NORM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

_rng = np.random.default_rng(1234567)  # fixed: signatures are comparable across runs
_A = _rng.integers(1, (1 << 31) - 1, size=NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, (1 << 31) - 1, size=NUM_PERM, dtype=np.uint64)


def normalise(text: str) -> str:
    return _WS_RE.sub(" ", _NORM_RE.sub(" ", text.lower())).strip()


def record_text(rec: Record) -> str:
    return normalise(f"{rec.instruction} {rec.input} {rec.output}")


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    if len(text) <= k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def signature(text: str) -> np.ndarray:
    """MinHash signature (uint64[NUM_PERM]) of the text's shingles."""
    hashes = np.fromiter(
        (xxhash.xxh32_intdigest(s.encode("utf-8")) for s in shingles(text)), dtype=np.uint64,
    )
    # (a*h + b) mod p, folded to 32 bits; a, h < 2^32 so a*h fits in uint64.
    permuted = ((np.outer(hashes, _A) + _B) % _MERSENNE) & _MAX_HASH
    return np.asarray(permuted.min(axis=0), dtype=np.uint64)


def _bands_for(threshold: float) -> tuple[int, int]:
    """(bands, rows) with bands*rows == NUM_PERM whose LSH threshold is closest."""
    options = [(b, NUM_PERM // b) for b in (8, 16, 32, 64) if NUM_PERM % b == 0]
    return min(options, key=lambda br: abs((1 / br[0]) ** (1 / br[1]) - threshold))


class Deduplicator:
    """Incremental exact + near-duplicate filter."""

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self.bands, self.rows = _bands_for(threshold)
        self._exact: set[str] = set()
        self._buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(self.bands)]
        self._sigs: list[np.ndarray] = []

    def is_duplicate(self, text: str) -> bool:
        """True if *text* matches something added before. Does not add it."""
        return self._check(text)[0]

    def add_if_new(self, text: str) -> bool:
        """Add *text* unless it duplicates something kept; True if it was added."""
        dup, digest, sig = self._check(text)
        if dup:
            return False
        self._exact.add(digest)
        idx = len(self._sigs)
        self._sigs.append(sig)
        for band, key in enumerate(self._band_keys(sig)):
            self._buckets[band][key].append(idx)
        return True

    def _check(self, text: str) -> tuple[bool, str, np.ndarray]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._exact:
            return True, digest, np.empty(0, dtype=np.uint64)
        sig = signature(text)
        seen: set[int] = set()
        for band, key in enumerate(self._band_keys(sig)):
            for idx in self._buckets[band].get(key, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                if float(np.mean(self._sigs[idx] == sig)) >= self.threshold:
                    return True, digest, sig
        return False, digest, sig

    def _band_keys(self, sig: np.ndarray) -> Iterable[bytes]:
        for band in range(self.bands):
            yield sig[band * self.rows:(band + 1) * self.rows].tobytes()


class SemanticDeduplicator:
    """Greedy cosine-similarity dedup over embedding vectors.

    Exact (no index): each new vector is compared with every kept one, which is
    fast enough for the dataset sizes Brainbrew produces (a few thousand rows).
    """

    def __init__(self, threshold: float = 0.92) -> None:
        self.threshold = threshold
        self._kept: np.ndarray | None = None  # capacity-doubling buffer of unit vectors
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add_if_new(self, vector: Sequence[float]) -> bool:
        v = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return True  # nothing to compare; keep it
        v = v / norm
        if self._kept is None:
            self._kept = np.empty((64, v.shape[0]), dtype=np.float32)
        elif self._n and float((self._kept[:self._n] @ v).max()) >= self.threshold:
            return False
        if self._n == len(self._kept):
            self._kept = np.concatenate([self._kept, np.empty_like(self._kept)])
        self._kept[self._n] = v
        self._n += 1
        return True


def deduplicate(
    records: list[Record],
    threshold: float = 0.85,
    key: Callable[[Record], str] = record_text,
) -> list[Record]:
    """Keep the first of each group of exact or near-duplicate records, in order."""
    dedup = Deduplicator(threshold)
    return [rec for rec in records if dedup.add_if_new(key(rec))]
