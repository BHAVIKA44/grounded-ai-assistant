"""
LLM service for answer generation with Ollama and OpenAI support.
"""

import time
from enum import Enum
from typing import List, Optional

import httpx
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.prompts import PromptTemplates

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
        """
        Generate an answer based on question and context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            system_prompt: Optional custom system prompt
            
        Returns:
            Tuple of (generated_answer, generation_time_ms)
        """
        start_time = time.perf_counter()

        # Build prompt
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
                raise ValueError(f"Unknown provider: {self.provider}")

            generation_time = (time.perf_counter() - start_time) * 1000

            logger.info(
                "answer_generated",
                provider=self.provider,
                model=self.model,
                generation_time_ms=generation_time,
                answer_length=len(answer),
            )

            return answer, generation_time

        except Exception as e:
            logger.error(
                "generation_failed",
                provider=self.provider,
                model=self.model,
                error=str(e),
            )
            raise

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
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")

            result = response.json()
            return result.get("response", "")

    async def _generate_openai(self, system: str, user_prompt: str) -> str:
        """Generate answer using OpenAI."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")

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
        """
        Generate answer with citation enforcement.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            sources: Source document names
            
        Returns:
            Tuple of (answer, cited_sources, generation_time_ms)
        """
        # Add citation instruction to system prompt
        system = PromptTemplates.SYSTEM_PROMPT + "\n\n" + PromptTemplates.CITATION_INSTRUCTION

        answer, gen_time = await self.generate(question, context_chunks, system_prompt=system)

        # Extract cited sources
        from app.services.prompts import extract_citations
        cited = extract_citations(answer)

        return answer, cited, gen_time


# Singleton instance
llm_service = LLMService(provider=LLMProvider.OLLAMA)


async def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    return llm_service