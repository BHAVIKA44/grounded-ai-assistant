"""
Prompt templates for RAG answer generation.
"""

from typing import List


class PromptTemplates:
    """Collection of prompt templates for RAG."""

    # System prompt for grounded answering
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

Your answers must:
1. Be based ONLY on the provided context
2. Include citations for all factual claims using the format [source]
3. Be concise and accurate
4. Admit when you don't know or when context is insufficient

Do NOT:
- Make up information not in the context
- Hallucinate facts or figures
- Answer questions that cannot be answered from the context"""

    # User prompt template
    USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context above
- Include citations using [source] format
- If the answer cannot be determined from the context, say "I cannot answer this based on the provided context."

Answer:"""

    # Citation format instruction
    CITATION_INSTRUCTION = """For each factual statement in your answer, include a citation in brackets showing the source document, e.g., [doc1], [document.pdf], etc."""

    # Fallback prompt when no context is found
    NO_CONTEXT_PROMPT = """The question asked does not appear to be related to any uploaded documents.

Question: {question}

Response: I don't have enough context to answer this question. Please upload relevant documents or rephrase your question."""

    @staticmethod
    def build_prompt(question: str, context_chunks: List[str]) -> str:
        """
        Build the full prompt with context.
        
        Args:
            question: User question
            context_chunks: List of context chunks
            
        Returns:
            Formatted prompt string
        """
        if not context_chunks:
            return PromptTemplates.NO_CONTEXT_PROMPT.format(question=question)

        # Format context with source markers
        context = "\n\n---\n\n".join(
            f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
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


# Citation extraction patterns
CITATION_PATTERNS = {
    "bracket": r"\[([^\]]+)\]",  # [source]
    "parenthesis": r"\(([^\)]+)\)",  # (source)
    "superscript": r"\^(\d+)",  # ^1
}


def extract_citations(text: str) -> List[str]:
    """
    Extract all citations from generated text.
    
    Args:
        text: Generated text with citations
        
    Returns:
        List of cited sources
    """
    import re

    citations = []
    for pattern in CITATION_PATTERNS.values():
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return list(set(citations))


def format_citation(source: str, page: int = None) -> str:
    """
    Format a citation string.
    
    Args:
        source: Source document name
        page: Optional page number
        
    Returns:
        Formatted citation
    """
    if page:
        return f"[{source} p.{page}]"
    return f"[{source}]"