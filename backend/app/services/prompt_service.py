"""Prompt templates and helpers for RAG generation."""

import re
from typing import List


class PromptTemplates:
    """Collection of prompt templates for RAG."""

    SYSTEM_PROMPT = """You are a precise AI assistant that answers based ONLY on the provided context.

Your answers must:
1. Be based ONLY on the provided context.
2. Be short and direct (3-6 lines unless user asks for detail).
3. Include citations using the format [source].
4. If context is insufficient or unrelated, reply exactly:
   "I can't answer this from the uploaded documents."

Do NOT:
- Make up information not in the context.
- Mention internal context labels like "Context 1", "Context 2", etc.
- Add repeated citations or duplicated source names."""

    USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context above
- Include citations using exact document names from the context, e.g. [my_file.pdf]
- If the answer cannot be determined from the context, say "I cannot answer this based on the provided context."

Answer:"""

    CITATION_INSTRUCTION = """For each factual statement in your answer, include a citation in brackets using exact source names found in context. Do not use generic placeholders like [source]."""

    NO_CONTEXT_PROMPT = """The question asked does not appear to be related to any uploaded documents.

Question: {question}

Response: I don't have enough context to answer this question. Please upload relevant documents or rephrase your question."""

    @staticmethod
    def build_prompt(question: str, context_chunks: List[str]) -> str:
        """Build the full prompt with context."""
        if not context_chunks:
            return PromptTemplates.NO_CONTEXT_PROMPT.format(question=question)

        context = "\n\n---\n\n".join(
            f"[Context {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        return PromptTemplates.USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    @staticmethod
    def build_sources_list(sources: List[str]) -> str:
        """Build a formatted sources list."""
        if not sources:
            return "No sources"
        unique_sources = list(set(sources))
        return "Sources: " + ", ".join(f"[{s}]" for s in unique_sources)


CITATION_PATTERNS = {
    "bracket": r"\[([^\]]+)\]",
    "parenthesis": r"\(([^\)]+)\)",
    "superscript": r"\^(\d+)",
}


def extract_citations(text: str) -> List[str]:
    """Extract all citations from generated text."""
    citations: List[str] = []
    for pattern in CITATION_PATTERNS.values():
        citations.extend(re.findall(pattern, text))
    return list(set(citations))


def format_citation(source: str, page: int | None = None) -> str:
    """Format a citation string."""
    if page:
        return f"[{source} p.{page}]"
    return f"[{source}]"
