"""Text utility helpers."""

import re


def normalize_whitespace(text: str) -> str:
    """Collapse extra whitespace while preserving content."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Basic text cleanup used before chunking and indexing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
