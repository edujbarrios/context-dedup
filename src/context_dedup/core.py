"""Core implementation for context-dedup."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from numbers import Real
from typing import Any

DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_CONTAINMENT_THRESHOLD = 0.8


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
) -> tuple[list[Any], list[str], list[dict[str, Any]], list[list[int]]]:
    items = list(chunks)
    texts = [key(item) for item in items]
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("key must return a string for every chunk")
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
    return items, texts, pairs, groups


def _identity(value: Any) -> Any:
    return value


def _prepare(
    chunks: Iterable[Any],
    key: Callable[[Any], str] | None,
    similarity_threshold: float,
    containment_threshold: float,
    n: int,
) -> tuple[list[Any], list[str], list[dict[str, Any]], list[list[int]]]:
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be a non-string iterable")
    try:
        iter(chunks)
    except TypeError as error:
        raise TypeError("chunks must be a non-string iterable") from error
    extractor = _identity if key is None else key
    if not callable(extractor):
        raise TypeError("key must be callable or None")
    similarity_limit = _validate_threshold(similarity_threshold, "similarity_threshold")
    containment_limit = _validate_threshold(containment_threshold, "containment_threshold")
    items, texts, pairs, groups = _find_groups(
        chunks,
        key=extractor,
        similarity_threshold=similarity_limit,
        containment_threshold=containment_limit,
        n=n,
    )
    return items, texts, pairs, groups


def _representative(indices: list[int], texts: list[str], strategy: str) -> int:
    if strategy == "first":
        return indices[0]
    if strategy == "longest":
        return max(indices, key=lambda index: len(texts[index]))
    raise ValueError("strategy must be 'longest' or 'first'")


def inspect_context(
    chunks: Iterable[Any],
    *,
    strategy: str = "longest",
    key: Callable[[Any], str] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    n: int = 3,
) -> dict[str, Any]:
    """Inspect chunks and return a serializable lexical-redundancy report.

    A pair is redundant when its Jaccard score reaches ``similarity_threshold``
    or either directional containment score reaches ``containment_threshold``.
    The defaults are practical heuristics, not universal statistical cutoffs.
    """
    if strategy not in {"longest", "first"}:
        raise ValueError("strategy must be 'longest' or 'first'")
    items, texts, pairs, components = _prepare(
        chunks, key, similarity_threshold, containment_threshold, n
    )
    groups = []
    removable: list[int] = []
    for indices in components:
        keep = _representative(indices, texts, strategy)
        remove = [index for index in indices if index != keep]
        removable.extend(remove)
        group_pairs = [
            pair for pair in pairs if pair["indices"][0] in indices and pair["indices"][1] in indices
        ]
        groups.append({
            "indices": indices,
            "representative": keep,
            "remove_indices": remove,
            "max_similarity": max(pair["similarity"] for pair in group_pairs),
            "max_containment": max(max(pair["containment"]) for pair in group_pairs),
        })
    removable.sort()
    total = len(items)
    return {
        "total_chunks": total,
        "redundant_pairs": len(pairs),
        "redundancy_groups": len(groups),
        "redundant_chunks": len(removable),
        "redundancy_ratio": len(removable) / total if total else 0.0,
        "estimated_redundant_characters": sum(len(texts[index]) for index in removable),
        "estimated_redundant_words": sum(len(texts[index].split()) for index in removable),
        "removable_indices": removable,
        "pairs": pairs,
        "groups": groups,
    }


def deduplicate(
    chunks: Iterable[Any],
    *,
    strategy: str = "longest",
    key: Callable[[Any], str] | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    n: int = 3,
) -> list[Any]:
    """Return chunks with lexical duplicates removed, preserving input order.

    ``longest`` keeps the longest text in each group (and the first on ties).
    ``first`` keeps the earliest chunk. Original objects are returned unchanged.
    """
    items = list(chunks) if not isinstance(chunks, (str, bytes)) else chunks
    report = inspect_context(
        items,
        strategy=strategy,
        key=key,
        similarity_threshold=similarity_threshold,
        containment_threshold=containment_threshold,
        n=n,
    )
    remove = set(report["removable_indices"])
    return [item for index, item in enumerate(items) if index not in remove]
