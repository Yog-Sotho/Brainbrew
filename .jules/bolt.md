# Bolt's Journal

Critical performance-related learnings specific to this codebase's architecture.

## 2026-03-31 - [Text Processing Pipeline Fast-Paths and Validation Allocations]
**Learning:** Text sanitization and preprocessing pipeline throughput can be significantly improved by introducing fast-path checks for invariant or absent criteria:
1. `unicodedata.normalize('NFKC')` is a complete no-op on pure-ASCII text, but incurs huge C-level translation overhead. Bypassing it via `text.isascii()` on clean strings eliminates this overhead entirely. Similarly, skipping manual bytes encoding when `text.isascii()` is True yields a fast $O(1)$-like quality ratio check.
2. Checking `'<' not in text` allows bypassing the `strip_html` regular expression substitution completely for standard texts.
3. Cheap single-pattern regex pre-checks (e.g., checking for potential PII candidates like `@` or digits before executing an 8-pass list of complex regexes) saves thousands of redundant regex sweeps, accelerating clean dataset processing by over 2.67x end-to-end.
4. Input validation patterns like `not text.strip()` can allocate significant string memory buffers on large source material. Replacing them with `not text or text.isspace()` avoids copying the underlying buffer entirely.

**Action:** Before performing regex replacements or multi-pass filters, always check if a fast search/character scan can completely skip the logic. Prefer `str.isascii()` and fast character membership tests over heavy transformations and regexes for clean paths.

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
