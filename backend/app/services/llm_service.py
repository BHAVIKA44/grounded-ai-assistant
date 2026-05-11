"""LLM service for answer generation with Ollama and OpenAI support."""

import time
from enum import Enum
from typing import List, Optional

import httpx
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, GenerationError
from app.core.logging import get_logger
from app.services.prompt_service import PromptTemplates, extract_citations

logger = get_logger(__name__)
settings = get_settings()


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMService:
    """Service for generating answers using LLMs."""

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OLLAMA,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model or settings.ollama_model
        self._openai_client = None

    async def generate(
        self,
        question: str,
        context_chunks: List[str],
        system_prompt: Optional[str] = None,
    ) -> tuple[str, float]:
        """Generate an answer based on question and context."""
        start_time = time.perf_counter()

        system = system_prompt or PromptTemplates.SYSTEM_PROMPT
        user_prompt = PromptTemplates.build_prompt(question, context_chunks)

        logger.info(
            "generating_answer",
            provider=self.provider,
            model=self.model,
            context_chunks=len(context_chunks),
        )

        try:
            if self.provider == LLMProvider.OLLAMA:
                answer = await self._generate_ollama(system, user_prompt)
            elif self.provider == LLMProvider.OPENAI:
                answer = await self._generate_openai(system, user_prompt)
            else:
                raise GenerationError(f"Unknown provider: {self.provider}")

            generation_time = (time.perf_counter() - start_time) * 1000

            logger.info(
                "answer_generated",
                provider=self.provider,
                model=self.model,
                generation_time_ms=generation_time,
                answer_length=len(answer),
            )
            return answer, generation_time

        except Exception as exc:
            logger.error(
                "generation_failed",
                provider=self.provider,
                model=self.model,
                error=str(exc),
            )
            if isinstance(exc, (GenerationError, ExternalServiceError)):
                raise
            raise GenerationError(str(exc))

    async def _generate_ollama(self, system: str, user_prompt: str) -> str:
        """Generate answer using Ollama."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"System: {system}\n\nUser: {user_prompt}",
                    "stream": False,
                },
            )

            if response.status_code != 200:
                raise ExternalServiceError(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            return result.get("response", "")

    async def _generate_openai(self, system: str, user_prompt: str) -> str:
        """Generate answer using OpenAI."""
        if not settings.openai_api_key:
            raise ExternalServiceError("OpenAI API key not configured")

        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
            max_tokens=1000,
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.agenerate([messages])
        return response.generations[0][0].text

    async def generate_with_citations(
        self,
        question: str,
        context_chunks: List[str],
        sources: List[str],
    ) -> tuple[str, List[str], float]:
        """Generate answer with citation enforcement."""
        normalized_sources = [s for s in dict.fromkeys(sources) if s]
        system = (
            PromptTemplates.SYSTEM_PROMPT
            + "\n\n"
            + PromptTemplates.CITATION_INSTRUCTION
        )

        answer, gen_time = await self.generate(
            question, context_chunks, system_prompt=system
        )
        cited = extract_citations(answer)
        lower_cited = {c.strip().lower() for c in cited}
        generic = {"source", "sources", "document", "doc", "context"}
        needs_rewrite = (not cited) or any(token in generic for token in lower_cited)
        if needs_rewrite and normalized_sources:
            source_label = normalized_sources[0]
            if answer.endswith((".", "!", "?")):
                answer = f"{answer} [{source_label}]"
            else:
                answer = f"{answer}. [{source_label}]"
            cited = [source_label]
        elif normalized_sources:
            valid_sources = {s.lower(): s for s in normalized_sources}
            filtered = []
            for c in cited:
                key = c.strip().lower()
                if key in valid_sources:
                    filtered.append(valid_sources[key])
            cited = list(dict.fromkeys(filtered)) or normalized_sources[:1]
        return answer, cited, gen_time


llm_service = LLMService(provider=LLMProvider.OLLAMA)


async def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    return llm_service
