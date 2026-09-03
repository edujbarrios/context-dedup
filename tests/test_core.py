import math

import pytest

from context_dedup import similarity
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
