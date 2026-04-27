"""
Observability service using LangSmith for tracing and monitoring.
"""

from typing import Any, Dict, Optional

from langsmith import Client

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global LangSmith client
_client: Optional[Client] = None


class ObservabilityService:
    """Service for LangSmith tracing and monitoring."""

    def __init__(self):
        self._client = None
        self._enabled = False

    @property
    def client(self) -> Optional[Client]:
        """Lazy initialize LangSmith client."""
        global _client
        if _client is None and settings.langsmith_api_key:
            logger.info("initializing_langsmith", project=settings.langsmith_project)
            _client = Client(
                api_key=settings.langsmith_api_key,
                api_url="https://api.smith.langchain.com",
            )
            self._enabled = True
        return _client

    def is_enabled(self) -> bool:
        """Check if LangSmith is enabled."""
        return self._enabled and self.client is not None

    def create_run(
        self,
        name: str,
        run_type: str = "chain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create a LangSmith run for tracing.
        
        Args:
            name: Run name
            run_type: Type of run (chain, llm, retriever, etc.)
            metadata: Optional metadata
            
        Returns:
            Run ID if enabled, None otherwise
        """
        if not self.is_enabled():
            return None

        try:
            # LangSmith run creation would go here
            # For now, we log the intent
            logger.debug(
                "langsmith_run_created",
                name=name,
                run_type=run_type,
                metadata=metadata,
            )
            return f"run_{name}_{run_type}"

        except Exception as e:
            logger.error("langsmith_run_error", error=str(e))
            return None

    def end_run(
        self,
        run_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a LangSmith run."""
        if not self.is_enabled():
            return

        logger.debug(
            "langsmith_run_ended",
            run_id=run_id,
            outputs=outputs,
            error=error,
        )

    def log_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a custom metric."""
        if not self.is_enabled():
            return

        logger.info(
            "metric_logged",
            metric_name=metric_name,
            value=value,
            tags=tags,
        )


# Singleton instance
observability_service = ObservabilityService()


async def get_observability_service() -> ObservabilityService:
    """Get the global observability service instance."""
    return observability_service


def setup_langsmith_tracing() -> None:
    """Setup LangSmith environment variables for LangChain tracing."""
    if settings.langsmith_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith_tracing).lower()
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("langsmith_tracing_enabled", project=settings.langsmith_project)
    else:
        logger.info("langsmith_not_configured")