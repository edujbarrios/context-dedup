"""Core implementation for context-dedup."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from numbers import Real
from typing import Any


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.lower().split())


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be at least 1")
    return n


def _validate_threshold(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a number between 0.0 and 1.0")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0.0 and 1.0 and finite")
    return result


def _word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    size = _validate_n(n)
    words = _normalize(text).split()
    if not words:
        return set()
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _scores(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> tuple[float, float, float]:
    if not left and not right:
        return 1.0, 1.0, 1.0
    shared = len(left & right)
    union = len(left | right)
    jaccard = shared / union if union else 1.0
    left_containment = shared / len(left) if left else 0.0
    right_containment = shared / len(right) if right else 0.0
    return jaccard, left_containment, right_containment


def similarity(text_a: str, text_b: str, *, n: int = 3) -> float:
    """Return word n-gram Jaccard similarity from 0.0 to 1.0.

    Text is lowercased and whitespace is collapsed. Trigrams are used by
    default; a non-empty text shorter than ``n`` is treated as one n-gram.
    Two empty or whitespace-only strings have similarity 1.0.
    """
    return _scores(_word_ngrams(text_a, n), _word_ngrams(text_b, n))[0]


def _find_groups(
    chunks: Iterable[Any],
    *,
    key: Callable[[Any], str],
    similarity_threshold: float,
    containment_threshold: float,
    n: int,
) -> tuple[list[Any], list[set[tuple[str, ...]]], list[dict[str, Any]], list[list[int]]]:
    items = list(chunks)
    texts = [key(item) for item in items]
    grams = [_word_ngrams(text, n) for text in texts]
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    pairs = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            jaccard, left_containment, right_containment = _scores(grams[left], grams[right])
            if jaccard >= similarity_threshold or max(left_containment, right_containment) >= containment_threshold:
                pairs.append({
                    "indices": [left, right],
                    "similarity": jaccard,
                    "containment": [left_containment, right_containment],
                })
                union(left, right)

    components: dict[int, list[int]] = {}
    for index in range(len(items)):
        components.setdefault(find(index), []).append(index)
    groups = [indices for indices in components.values() if len(indices) > 1]
    return items, grams, pairs, groups
