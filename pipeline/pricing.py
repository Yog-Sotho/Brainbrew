"""
API prices, used for the estimate before a run and the actual cost after it.

USD per 1M tokens (input, output) for hosted models whose price is public.
Matched by substring of the model name, longest key first, so "gpt-4o-mini"
wins over "gpt-4o". Local endpoints cost nothing; unknown hosted models have
no actual cost (None) and a conservative estimate.
"""
from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

# Source: https://openai.com/api/pricing/
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}
UNKNOWN_ESTIMATE = (2.50, 10.00)  # conservative default for estimates only
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}  # noqa: S104 - hosts we classify, not a bind address


def is_local(base_url: str | None) -> bool:
    return base_url is not None and urlsplit(base_url).hostname in LOCAL_HOSTS


def price(model: str) -> tuple[float, float] | None:
    name = model.lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if key in name:
            return PRICES[key]
    return None


def usage_cost(model: str, usage: Mapping[str, int], local: bool = False) -> float | None:
    """Actual cost in USD of *usage* (prompt/completion tokens), or None if unknown."""
    if local:
        return 0.0
    p = price(model)
    if p is None:
        return None
    return round((usage.get("prompt_tokens", 0) * p[0] + usage.get("completion_tokens", 0) * p[1]) / 1e6, 6)


def run_cost(usage: Mapping[str, Mapping[str, int]], local: bool) -> dict[str, float | None]:
    """Per-client cost for a manifest `usage` block ({"teacher:gpt-4o-mini": {...}, ...}) plus a total.

    The total is None when any hosted model's price is unknown.
    """
    costs: dict[str, float | None] = {
        key: usage_cost(key.split(":", 1)[-1], u, local) for key, u in usage.items()
    }
    values = list(costs.values())
    costs["total"] = None if any(v is None for v in values) else round(sum(v or 0.0 for v in values), 6)
    return costs


def estimate_cost(model: str, total_tokens: int, local: bool) -> float | None:
    """Rough cost of *total_tokens*, assuming 3 input tokens per output token."""
    if local:
        return 0.0
    inp, out = price(model) or UNKNOWN_ESTIMATE
    return total_tokens * (0.75 * inp + 0.25 * out) / 1e6
