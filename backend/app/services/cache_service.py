"""
Redis caching service for responses and retrieval results.
"""

import json
import hashlib
from typing import Any, Optional

import redis.asyncio as redis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global Redis client
_client = None


class CacheService:
    """Redis-based caching service."""

    def __init__(self):
        self._client = None
        self._ttl = 3600  # Default 1 hour TTL

    @property
    def client(self) -> redis.Redis:
        """Lazy initialize Redis client."""
        global _client
        if _client is None:
            logger.info("connecting_to_redis", url=settings.redis_url)
            _client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return _client

    def _hash_key(self, key: str) -> str:
        """Generate a hash for the key."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            hashed_key = self._hash_key(key)
            value = await self.client.get(hashed_key)

            if value:
                logger.debug("cache_hit", key=key[:50])
                return json.loads(value)

            logger.debug("cache_miss", key=key[:50])
            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            hashed_key = self._hash_key(key)
            ttl = ttl or self._ttl

            serialized = json.dumps(value)
            await self.client.setex(hashed_key, ttl, serialized)

            logger.debug("cache_set", key=key[:50], ttl=ttl)
            return True

        except Exception as e:
            logger.error("cache_set_error", error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            hashed_key = self._hash_key(key)
            await self.client.delete(hashed_key)
            logger.debug("cache_deleted", key=key[:50])
            return True

        except Exception as e:
            logger.error("cache_delete_error", error=str(e))
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "rag:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self.client.delete(*keys)
                logger.info("cache_cleared", pattern=pattern, count=len(keys))

            return len(keys)

        except Exception as e:
            logger.error("cache_clear_error", error=str(e))
            return 0

    async def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            await self.client.ping()
            return True
        except Exception:
            return False


# Singleton instance
cache_service = CacheService()


async def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    return cache_service


# Cache key generators
def cache_key_question(question: str) -> str:
    """Generate cache key for a question."""
    return f"rag:question:{hashlib.md5(question.encode()).hexdigest()}"


def cache_key_retrieval(query: str, top_k: int) -> str:
    """Generate cache key for retrieval results."""
    return f"rag:retrieval:{hashlib.md5(f'{query}:{top_k}'.encode()).hexdigest()}"