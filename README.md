# context-dedup

[![PyPI version](https://img.shields.io/pypi/v/context-dedup.svg?label=PyPI&logo=pypi&cacheSeconds=300)](https://pypi.org/project/context-dedup/)
[![License: MPL 2.0](https://img.shields.io/badge/license-MPL%202.0-blue.svg)](https://github.com/edujbarrios/context-dedup/blob/main/LICENSE)

Detect and remove redundant context before sending it to an LLM.

context-dedup is a small, deterministic Python library for finding exact and near-duplicate chunks in RAG results, agent pipelines, and assembled prompts. It uses lexical word n-gram overlap: no embeddings, no LLM calls, and no runtime dependencies.

## Installation

```bash
pip install context-dedup
```

## Usage

```python
from context_dedup import deduplicate, inspect_context, similarity

chunks = [
    "The Eiffel Tower is located in Paris.",
    "The Eiffel Tower is located in Paris. It was completed in 1889.",
    "Madrid is the capital of Spain.",
]

print(similarity(chunks[0], chunks[1]))
print(inspect_context(chunks))
print(deduplicate(chunks))
```

## Why context-dedup?

Retrieval and prompt assembly can repeat the same facts across overlapping passages, wasting context-window space and attention. context-dedup provides a transparent heuristic between raw retrieval and an LLM call, without adding a model, service, database, or provider SDK.

It is intentionally focused:

* No embeddings or transformer models
* No LLM calls or external APIs
* No runtime dependencies or infrastructure

## Inspect context

`inspect_context` returns an ordinary, serializable dictionary with pair scores, connected duplicate groups, recommended representatives, removable indices, and aggregate redundancy estimates:

```python
report = inspect_context(chunks)

print(report["redundant_pairs"])
print(report["groups"])
print(report["estimated_redundant_words"])
```

Thresholds are configurable. Their defaults are practical heuristics, not statistically universal values:

```python
report = inspect_context(
    chunks,
    similarity_threshold=0.75,
    containment_threshold=0.85,
    n=3,
)
```

## Deduplicate context

The default strategy keeps the longest chunk in each duplicate group, preserving more information. Use `first` to keep the earliest chunk instead:

```python
clean_chunks = deduplicate(chunks)
first_chunks = deduplicate(chunks, strategy="first")
```

Objects with metadata are supported through `key`; returned objects are the originals:

```python
retrieved = [
    {"text": "Refunds are available within 30 days.", "source": "policy.pdf", "page": 4},
    {"text": "Refunds are available within 30 days. Contact support to begin.", "source": "faq.pdf", "page": 2},
]

clean = deduplicate(retrieved, key=lambda item: item["text"])
```

## Algorithm

Text is lowercased, stripped, and normalized to single spaces. The library builds sets of word trigrams by default, then calculates Jaccard similarity and containment in both directions. A pair is redundant when either configured threshold is reached. Connected components combine transitive pairs into groups, and selection is deterministic.

## Limitations

This is a lexical heuristic, not a semantic-equivalence detector. Paraphrases with different wording may not match, while repeated wording can match despite different meaning. Punctuation remains part of word tokens. Version 0.1.0 compares every pair in `O(n²)` time, which is appropriate for contexts with tens or hundreds of chunks but not large document collections.

## Use cases

* Remove overlap from RAG retrieval results before prompt construction
* Inspect context redundancy in LLM and agent pipelines
* Deduplicate passages assembled from multiple document sources
* Reduce repeated prompt content without provider-specific infrastructure

## Features

* Deterministic word n-gram Jaccard similarity and directional containment
* Transitive duplicate groups and serializable inspection reports
* `longest` and `first` representative strategies
* Metadata-preserving `key` support
* Configurable thresholds and n-gram size
* Standard-library implementation with no runtime dependencies

## Issues

Report issues in the [GitHub issue tracker](https://github.com/edujbarrios/context-dedup/issues).

## Author

Eduardo J. Barrios — [edujbarrios@outlook.com](mailto:edujbarrios@outlook.com)

## License

[Mozilla Public License 2.0](https://github.com/edujbarrios/context-dedup/blob/main/LICENSE)
