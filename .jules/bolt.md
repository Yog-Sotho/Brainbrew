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
