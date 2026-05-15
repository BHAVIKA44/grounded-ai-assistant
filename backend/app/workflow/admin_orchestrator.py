"""LangGraph orchestration for admin action flows."""

from __future__ import annotations

from typing import Any, Dict, TypedDict

import httpx
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.services.llm_service import LLMProvider, LLMService

settings = get_settings()


class AdminState(TypedDict, total=False):
    provider: str
    model: str
    svc: LLMService
    supported_ollama: list[str]
    supported_groq: list[str]
    response: Dict[str, Any]
    fine_tune_state: Dict[str, Any]
    fine_tuning_service: Any
    dataset: Any
    dataset_path: str
    output_path: str


class AdminOrchestrator:
    def __init__(self) -> None:
        self.set_llm_graph = self._build_set_llm_graph()
        self.pull_graph = self._build_pull_graph()
        self.ft_prepare_graph = self._build_ft_prepare_graph()
        self.ft_run_graph = self._build_ft_run_graph()

    def _build_set_llm_graph(self):
        async def validate(state: AdminState) -> AdminState:
            provider = state["provider"]
            model = state.get("model", "")
            if provider == "ollama" and model and model not in state["supported_ollama"]:
                raise ValueError(f"Unsupported model '{model}'. Use one of: {', '.join(state['supported_ollama'])}")
            if provider == "groq" and model and model not in state["supported_groq"]:
                raise ValueError(f"Unsupported model '{model}'. Use one of: {', '.join(state['supported_groq'])}")
            return state

        async def apply(state: AdminState) -> AdminState:
            svc = state["svc"]
            svc.provider = LLMProvider(state["provider"])
            if state.get("model"):
                svc.model = state["model"]
            state["response"] = {"provider": svc.provider, "model": svc.model}
            return state

        graph = StateGraph(AdminState)
        graph.add_node("validate", validate)
        graph.add_node("apply", apply)
        graph.set_entry_point("validate")
        graph.add_edge("validate", "apply")
        graph.add_edge("apply", END)
        return graph.compile()

    def _build_pull_graph(self):
        async def validate(state: AdminState) -> AdminState:
            if state["model"] not in state["supported_ollama"]:
                raise ValueError(f"Unsupported model '{state['model']}'. Use one of: {', '.join(state['supported_ollama'])}")
            return state

        async def pull(state: AdminState) -> AdminState:
            async with httpx.AsyncClient(timeout=1200.0) as client:
                response = await client.post(
                    f"{settings.ollama_base_url}/api/pull",
                    json={"name": state["model"], "stream": False},
                )
            if response.status_code != 200:
                raise ExternalServiceError(f"Ollama pull failed: {response.text}")
            state["response"] = {"pulled": True, "model": state["model"]}
            return state

        graph = StateGraph(AdminState)
        graph.add_node("validate", validate)
        graph.add_node("pull", pull)
        graph.set_entry_point("validate")
        graph.add_edge("validate", "pull")
        graph.add_edge("pull", END)
        return graph.compile()

    def _build_ft_prepare_graph(self):
        async def prepare(state: AdminState) -> AdminState:
            path = state["fine_tuning_service"].prepare_dataset(
                data=state["dataset"],
                output_path="/app/data/fine_tuning/train.jsonl",
            )
            state["dataset_path"] = path
            state["fine_tune_state"]["dataset_path"] = path
            state["fine_tune_state"]["status"] = "prepared"
            state["response"] = {"prepared": True, "dataset_path": path, "samples": len(state["dataset"])}
            return state

        graph = StateGraph(AdminState)
        graph.add_node("prepare", prepare)
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", END)
        return graph.compile()

    def _build_ft_run_graph(self):
        async def validate(state: AdminState) -> AdminState:
            if not state.get("dataset_path"):
                raise ValueError("No dataset prepared")
            return state

        async def run(state: AdminState) -> AdminState:
            state["fine_tune_state"]["status"] = "running"
            state["fine_tune_state"]["error"] = None
            output = await state["fine_tuning_service"].fine_tune(state["dataset_path"])
            state["output_path"] = output
            state["fine_tune_state"]["status"] = "completed"
            state["fine_tune_state"]["output_path"] = output
            state["response"] = {"status": "completed", "dataset_path": state["dataset_path"], "output_path": output}
            return state

        graph = StateGraph(AdminState)
        graph.add_node("validate", validate)
        graph.add_node("run", run)
        graph.set_entry_point("validate")
        graph.add_edge("validate", "run")
        graph.add_edge("run", END)
        return graph.compile()

