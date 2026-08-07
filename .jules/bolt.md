# Bolt's Journal

Critical performance-related learnings specific to this codebase's architecture.

## 2026-03-30 - [ASCII Ratio Check Optimization and Deduplication Bucketing Pitfalls]
**Learning:**
1. The quality-filtering gate's ASCII ratio check was previously calculated character-by-character using `sum(1 for c in text if ord(c) < 128)`. This creates a heavy pure-Python generator loop. We found that replacing this with a C-optimized encoding length check `len(text.encode('ascii', errors='ignore'))` performs the exact same count but is over 60x faster, resulting in a ~6.4% end-to-end speedup for dataset sanitization.
2. We experimented with bucketing near-duplicate deduplication based on instruction shingle lengths to avoid $O(N^2)$ pairs. Surprisingly, this did NOT yield a speedup on realistic synthetic datasets because the dataset generation prompts are built from uniform templates with a fixed chunk size (e.g. 800-1000 chars), meaning shingle lengths vary very little and fall into the same range. The added overhead of range calculation and dictionary lookups made it slightly slower.

**Action:** Prefer C-level string/bytes operations over pure Python loops for char-by-char checks. Always analyze the distribution of indexing keys (like shingle lengths) before committing to a bucketing strategy, as uniform key distributions render bucketing ineffective.

## 2026-03-29 - [Jaccard Similarity Set Union and Near-Duplicate Pruning]
**Learning:** In the near-duplicate deduplication pipeline, checking Jaccard similarity across large record batches (up to 20,000 records) is a severe CPU bottleneck due to its $O(N^2)$ nature. We discovered two major optimization opportunities:
1. Calculating Jaccard similarity using `len(a | b)` is highly inefficient because it allocates a brand new Python set, copies all items, and hashes them again. Using set length arithmetic `len(a) + len(b) - len(a & b)` yields a massive 2.5x speedup per similarity calculation by eliminating set allocation/copy overhead entirely.
2. The $O(N^2)$ candidate pairs can be aggressively pruned before any set operations by comparing the shingle set size ratios. Since the Jaccard similarity is upper-bounded by `min(|A|, |B|) / max(|A|, |B|)`, and both field similarities must meet a minimum mathematical bound to reach the combined threshold, we can skip over 90% of costly set operations in $O(1)$ float checks, yielding an overall 4.3x speedup on 2,000 records, and even higher on larger datasets.

**Action:** Always precalculate and store set/collection lengths to enable $O(1)$ bounding/pruning before executing expensive set operations (like union or intersection) or heavy nested loops. When computing Jaccard similarity between Python sets, prefer set length arithmetic over explicit set union operations.

## 2026-04-01 - [PII Redaction Redundant Iterative Substitution]
**Learning:**
In the dataset sanitization pipeline, the PII redaction step `redact_pii` ran all 8 complex regex substitution patterns sequentially on every record field regardless of whether any PII actually existed. This caused massive redundant C-level regex loop traversals for clean data (which represents 95%+ of typical distillation datasets). We discovered that applying a fast, unified single-regex precheck `_PII_CANDIDATE_RE = re.compile(r'[@0-9+]|http|www\.', re.IGNORECASE)` to perform a single `.search()` check can bypass the 8 costly `subn` operations entirely for clean records. This yields a massive ~8x speedup on clean text, making the dataset sanitization significantly more efficient. We also noted that case-insensitivity checks with `re.IGNORECASE` have some overhead, but are mathematically required here to safely cover all variations of URL indicators.

**Action:** Before running an expensive series of sequential regex substitutions/modifications on strings, always implement a single-pass, cheap regex/substring precheck to exit early if no work is needed.

## 2026-04-02 - [Exact-Match Deduplication Cryptographic Hash Removal]
**Learning:** For exact-match deduplication in pipelines or exporters, computing cryptographic hashes (such as `hashlib.sha256`) on concatenated text keys is highly expensive due to Python-to-C buffer encoding (UTF-8 bytes conversion) and cryptographic hash computation. Since Python's internal string hashing and `set` membership are already $O(1)$ and written in highly optimized C, using plain-string set keys avoids all cryptographic overhead. This results in a massive ~6x speedup on exact deduplication checks and a ~1.35x speedup end-to-end on dataset deduplication.

**Action:** When implementing exact deduplication in memory, prefer plain-string set/dict keys over cryptographic hash keys unless the string keys are excessively large and memory bounds are an issue.
