import math

import pytest

from context_dedup import deduplicate, inspect_context, similarity
from context_dedup.core import _find_groups


def test_exact_capitalization_and_whitespace_duplicates():
    assert similarity("Python is a language", "Python is a language") == 1.0
    assert similarity("  PYTHON  is\na language ", "python is a language") == 1.0


def test_different_strings_and_symmetry():
    left = "Madrid is the capital of Spain"
    right = "Whales live in the ocean"
    assert similarity(left, right) == 0.0
    assert similarity(left, right) == similarity(right, left)


def test_near_duplicate_has_partial_similarity():
    score = similarity("one two three four five", "one two three four six")
    assert 0.0 < score < 1.0


def test_empty_and_short_strings():
    assert similarity("", " \n") == 1.0
    assert similarity("", "content") == 0.0
    assert similarity("short text", "SHORT   TEXT") == 1.0


def test_containment_and_transitive_groups():
    chunks = [
        "alpha beta gamma delta",
        "alpha beta gamma delta epsilon zeta",
        "gamma delta epsilon zeta eta",
    ]
    _, _, pairs, groups = _find_groups(
        chunks,
        key=lambda value: value,
        similarity_threshold=0.9,
        containment_threshold=0.65,
        n=3,
    )
    assert [pair["indices"] for pair in pairs] == [[0, 1], [1, 2]]
    assert groups == [[0, 1, 2]]


@pytest.mark.parametrize("n,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_invalid_ngram_size(n, error):
    with pytest.raises(error):
        similarity("a", "a", n=n)


@pytest.mark.parametrize("value", [None, 1, []])
def test_non_string_input(value):
    with pytest.raises(TypeError):
        similarity(value, "text")


def test_report_consistency_and_longest_selection():
    chunks = [
        "The Eiffel Tower is located in Paris.",
        "The Eiffel Tower is located in Paris. It was completed in 1889.",
        "Madrid is the capital of Spain.",
    ]
    report = inspect_context(chunks)
    assert report["total_chunks"] == 3
    assert report["redundant_pairs"] == 1
    assert report["redundancy_groups"] == 1
    assert report["redundant_chunks"] == 1
    assert report["removable_indices"] == [0]
    assert report["groups"][0]["representative"] == 1
    assert report["redundancy_ratio"] == pytest.approx(1 / 3)
    assert report["estimated_redundant_characters"] == len(chunks[0])
    assert report["estimated_redundant_words"] == len(chunks[0].split())
    assert deduplicate(chunks) == chunks[1:]


def test_default_containment_matches_overlapping_rag_chunks():
    chunks = [
        "Python is a programming language.",
        "Python is a programming language used for data science.",
        "Madrid is the capital of Spain.",
    ]
    assert inspect_context(chunks)["redundant_pairs"] == 1
    assert deduplicate(chunks) == chunks[1:]


def test_first_strategy_and_no_mutation():
    chunks = ["alpha beta gamma", "alpha beta gamma delta"]
    original = chunks.copy()
    assert deduplicate(chunks, strategy="first") == [chunks[0]]
    assert chunks == original


def test_configurable_thresholds():
    chunks = ["one two three four five", "one two three four six"]
    assert deduplicate(chunks, similarity_threshold=0.49, containment_threshold=1.0) == [chunks[0]]
    assert deduplicate(chunks, similarity_threshold=0.51, containment_threshold=1.0) == chunks


def test_metadata_key_preserves_original_objects():
    chunks = [
        {"text": "same useful context", "page": 1},
        {"text": "same useful context with detail", "page": 2},
    ]
    result = deduplicate(chunks, key=lambda item: item["text"])
    assert result == [chunks[1]]
    assert result[0] is chunks[1]


def test_key_is_called_once_per_item():
    calls = []

    def extract(item):
        calls.append(item)
        return item["text"]

    chunks = [{"text": "one"}, {"text": "two"}]
    inspect_context(chunks, key=extract)
    assert calls == chunks


def test_transitive_deduplicate_is_deterministic():
    chunks = [
        "alpha beta gamma delta",
        "alpha beta gamma delta epsilon zeta",
        "gamma delta epsilon zeta eta",
    ]
    options = {"similarity_threshold": 0.9, "containment_threshold": 0.65}
    first = inspect_context(chunks, **options)
    assert first == inspect_context(chunks, **options)
    assert first["groups"][0]["indices"] == [0, 1, 2]
    assert deduplicate(chunks, **options) == [chunks[1]]


def test_empty_context_report():
    report = inspect_context([])
    assert report["total_chunks"] == 0
    assert report["redundancy_ratio"] == 0.0
    assert report["groups"] == []
    assert deduplicate([]) == []


@pytest.mark.parametrize("strategy", ["last", "shortest", ""])
def test_invalid_strategy(strategy):
    with pytest.raises(ValueError):
        deduplicate(["a"], strategy=strategy)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        inspect_context(["a"], similarity_threshold=threshold)


def test_invalid_chunks_and_key():
    with pytest.raises(TypeError):
        inspect_context("not a chunk collection")
    with pytest.raises(TypeError):
        inspect_context([{"text": "x"}], key=None)
    with pytest.raises(TypeError):
        inspect_context(["x"], key="text")
