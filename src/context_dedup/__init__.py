"""Small, deterministic tools for removing redundant LLM context."""

from .core import deduplicate, inspect_context, similarity

__all__ = ["similarity", "inspect_context", "deduplicate"]
