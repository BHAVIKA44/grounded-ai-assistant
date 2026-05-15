from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
import httpx

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.db.session import get_session
from app.fine_tuning.trainer import get_fine_tuning_service
from app.services.cache_service import get_cache_service
from app.services.llm_service import LLMProvider, get_llm_service
from app.services.observability import get_observability_service
from app.workflow.admin_orchestrator import AdminOrchestrator

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
_fine_tune_state: Dict[str, Any] = {"status": "idle", "dataset_path": None, "output_path": None, "error": None}
SUPPORTED_OLLAMA_MODELS = ["llama3.2", "phi3:mini", "qwen2.5:3b"]
SUPPORTED_GROQ_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
admin_orchestrator = AdminOrchestrator()


@router.get("/status")
async def status() -> Dict[str, Any]:
    settings = get_settings()
    redis_ok = False
    db_ok = False
    try:
        cache = await get_cache_service()
        redis_ok = await cache.ping()
    except Exception:
        redis_ok = False
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        db_ok = False
    obs = await get_observability_service()
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "services": {"db": db_ok, "redis": redis_ok, "langsmith": obs.is_enabled()},
        "llm": {
            "provider": "ollama",
            "model": settings.ollama_model,
            "openai_model": settings.openai_model,
            "groq_model": settings.groq_model,
        },
    }


@router.get("/cache")
async def cache_status() -> Dict[str, Any]:
    cache = await get_cache_service()
    return {"available": await cache.ping()}


@router.delete("/cache")
async def clear_cache(pattern: str = Query(default="*")) -> Dict[str, Any]:
    cache = await get_cache_service()
    deleted = await cache.clear_pattern(pattern)
    return {"deleted": deleted, "pattern": pattern}


@router.get("/llm")
async def llm_status() -> Dict[str, Any]:
    svc = await get_llm_service()
    return {
        "provider": svc.provider,
        "model": svc.model,
        "supported_providers": ["ollama", "groq"],
        "supported_models": SUPPORTED_OLLAMA_MODELS,
        "supported_models_by_provider": {
            "ollama": SUPPORTED_OLLAMA_MODELS,
            "groq": SUPPORTED_GROQ_MODELS,
        },
    }


@router.post("/llm")
async def set_llm(provider: str = "ollama", model: str = "") -> Dict[str, Any]:
    svc = await get_llm_service()
    try:
        state = await admin_orchestrator.set_llm_graph.ainvoke(
            {
                "provider": provider,
                "model": model,
                "svc": svc,
                "supported_ollama": SUPPORTED_OLLAMA_MODELS,
                "supported_groq": SUPPORTED_GROQ_MODELS,
            }
        )
    except (ValueError,) as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "validation_error", "message": str(exc)},
        )
    return {
        "provider": state["response"]["provider"],
        "model": state["response"]["model"],
        "supported_models": SUPPORTED_OLLAMA_MODELS,
    }


@router.post("/llm/pull")
async def pull_llm_model(model: str) -> Dict[str, Any]:
    try:
        state = await admin_orchestrator.pull_graph.ainvoke(
            {"model": model, "supported_ollama": SUPPORTED_OLLAMA_MODELS}
        )
        return state["response"]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "validation_error", "message": str(exc)},
        )
    except ExternalServiceError as exc:
        logger.error("admin_llm_pull_failed", model=model, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        )
    except httpx.HTTPError as exc:
        logger.error("admin_llm_pull_failed", model=model, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail={"code": "external_service_error", "message": str(exc)},
        )


@router.post("/fine-tuning/prepare")
async def prepare_fine_tuning(data: List[Dict[str, str]]) -> Dict[str, Any]:
    svc = await get_fine_tuning_service()
    state = await admin_orchestrator.ft_prepare_graph.ainvoke(
        {"fine_tuning_service": svc, "dataset": data, "fine_tune_state": _fine_tune_state}
    )
    return state["response"]


@router.post("/fine-tuning/run")
async def run_fine_tuning(dataset_path: str | None = None) -> Dict[str, Any]:
    svc = await get_fine_tuning_service()
    active_dataset = dataset_path or _fine_tune_state.get("dataset_path")
    if not active_dataset:
        raise HTTPException(
            status_code=400,
            detail={"code": "validation_error", "message": "No dataset prepared"},
        )
    _fine_tune_state["status"] = "running"
    _fine_tune_state["error"] = None
    try:
        state = await admin_orchestrator.ft_run_graph.ainvoke(
            {
                "fine_tuning_service": svc,
                "dataset_path": active_dataset,
                "fine_tune_state": _fine_tune_state,
            }
        )
        return state["response"]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "validation_error", "message": str(exc)},
        )
    except Exception as exc:
        _fine_tune_state["status"] = "failed"
        _fine_tune_state["error"] = str(exc)
        svc_logger = getattr(svc, "logger", None)
        if svc_logger is not None:
            svc_logger.exception("fine_tuning_run_failed", dataset_path=active_dataset)
        raise HTTPException(
            status_code=500,
            detail={"code": "fine_tuning_failed", "message": str(exc)},
        )


@router.get("/fine-tuning/status")
async def fine_tuning_status() -> Dict[str, Any]:
    return _fine_tune_state
