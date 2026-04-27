"""
Configuration access layer.
Provides cached settings + helper functions for environment access.
"""

import os
from functools import lru_cache

from app.core.settings import Settings


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_env(key: str, default: str | None = None) -> str | None:
    """Read a raw environment variable."""
    return os.getenv(key, default)
