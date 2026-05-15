"""LangGraph orchestration for document write flows."""

from __future__ import annotations

from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from app.schemas.document import DocumentType
from app.services.document_service import DocumentService


class DocumentState(TypedDict, total=False):
    service: DocumentService
    session: Any
    filename: str
    title: str
    content_bytes: bytes
    content_text: str
    document_type: str
    document: Any
    document_id: str
    success: bool


class DocumentOrchestrator:
    def __init__(self) -> None:
        self.upload_graph = self._build_upload_graph()
        self.delete_graph = self._build_delete_graph()

    def _build_upload_graph(self):
        async def determine_type(state: DocumentState) -> DocumentState:
            ext = state["filename"].split(".")[-1].lower() if "." in state["filename"] else ""
            if ext == "pdf":
                state["document_type"] = DocumentType.PDF.value
            elif ext == "docx":
                state["document_type"] = DocumentType.DOCX.value
            else:
                state["document_type"] = DocumentType.TEXT.value
            return state

        async def normalize_content(state: DocumentState) -> DocumentState:
            if state["document_type"] == DocumentType.TEXT.value:
                try:
                    state["content_text"] = state["content_bytes"].decode("utf-8")
                except UnicodeDecodeError:
                    state["content_text"] = state["content_bytes"].decode("latin-1")
            else:
                state["content_text"] = state["filename"]
            return state

        async def create_document(state: DocumentState) -> DocumentState:
            state["document"] = await state["service"].create_document(
                session=state["session"],
                title=state["title"],
                content=state["content_text"],
                document_type=state["document_type"],
                filename=state["filename"],
                file_bytes=state["content_bytes"],
            )
            return state

        graph = StateGraph(DocumentState)
        graph.add_node("determine_type", determine_type)
        graph.add_node("normalize_content", normalize_content)
        graph.add_node("create_document", create_document)
        graph.set_entry_point("determine_type")
        graph.add_edge("determine_type", "normalize_content")
        graph.add_edge("normalize_content", "create_document")
        graph.add_edge("create_document", END)
        return graph.compile()

    def _build_delete_graph(self):
        async def delete_document(state: DocumentState) -> DocumentState:
            state["success"] = await state["service"].delete_document(
                state["session"], state["document_id"]
            )
            return state

        graph = StateGraph(DocumentState)
        graph.add_node("delete_document", delete_document)
        graph.set_entry_point("delete_document")
        graph.add_edge("delete_document", END)
        return graph.compile()

