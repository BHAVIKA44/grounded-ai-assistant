"""
FastAPI application entrypoint.
Configures middleware, routes, and lifecycle management.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.observability import setup_langsmith_tracing

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan manager for startup/shutdown."""
    settings = get_settings()
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
    )

    # Startup: Initialize connections, load models, etc.
    try:
        # Setup observability
        setup_langsmith_tracing()

        # Initialize database
        from app.db.session import init_db
        await init_db()

        # Initialize Redis connection check
        from app.services.cache_service import get_cache_service
        cache = await get_cache_service()
        redis_available = await cache.ping()
        logger.info("redis_status", available=redis_available)

        logger.info("application_started_successfully")
    except Exception as e:
        logger.error("application_startup_failed", error=str(e))
        # Don't raise - allow app to start in degraded mode

    yield

    # Shutdown: Clean up resources
    logger.info("application_shutting_down")
    try:
        from app.db.session import close_db
        await close_db()
        logger.info("application_shutdown_complete")
    except Exception as e:
        logger.error("application_shutdown_error", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade RAG system with hybrid retrieval",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next) -> Response:
        """Add processing time to response headers."""
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        """Log all incoming requests."""
        logger.info(
            "request_received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )
        response = await call_next(request)
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )
        return response

    return app


app = create_app()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Dict containing health status and version info.
    """
    settings = get_settings()

    # Check Redis
    redis_status = "unknown"
    try:
        from app.services.cache_service import get_cache_service
        cache = await get_cache_service()
        redis_status = "healthy" if await cache.ping() else "unhealthy"
    except Exception:
        redis_status = "error"

    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "timestamp": time.time(),
        "services": {
            "redis": redis_status,
        },
    }


@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint.
    Returns 503 if the service is not ready to accept traffic.
    """
    # Check database
    try:
        from app.db.session import get_session
        async with get_session() as session:
            await session.execute("SELECT 1")
    except Exception:
        return {"status": "not_ready", "reason": "database unavailable"}

    return {"status": "ready"}


# Import and include routers
from app.api import documents, chat, evaluation

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(evaluation.router)
