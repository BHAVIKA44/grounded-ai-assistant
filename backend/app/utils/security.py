"""Security helpers for prompt and user input hardening."""

import re

INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s*:\s*",
    r"developer\s*:\s*",
    r"jailbreak",
]


def has_prompt_injection(text: str) -> bool:
    """Detect common prompt-injection phrases."""
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in INJECTION_PATTERNS)


def sanitize_user_text(text: str) -> str:
    """Remove dangerous control characters and trim length."""
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return clean.strip()
