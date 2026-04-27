"""
Document chunking service with multiple strategies.
"""

import re
import uuid
from enum import Enum
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"


class ChunkingConfig:
    """Configuration for document chunking."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size


class Chunk:
    """Represents a document chunk."""

    def __init__(
        self,
        content: str,
        chunk_index: int,
        document_id: str,
        source: str,
        page: Optional[int] = None,
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.chunk_index = chunk_index
        self.document_id = document_id
        self.source = source
        self.page = page

    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
        }


class DocumentChunker:
    """Service for chunking documents with various strategies."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk_document(
        self, text: str, document_id: str, source: str, page: Optional[int] = None
    ) -> List[Chunk]:
        """
        Chunk document based on configured strategy.
        
        Args:
            text: Document text to chunk
            document_id: ID of the parent document
            source: Source filename
            page: Page number (for PDFs)
            
        Returns:
            List of Chunk objects
        """
        strategy_map = {
            ChunkingStrategy.FIXED_SIZE: self._chunk_fixed_size,
            ChunkingStrategy.RECURSIVE: self._chunk_recursive,
            ChunkingStrategy.SEMANTIC: self._chunk_semantic,
            ChunkingStrategy.MARKDOWN: self._chunk_markdown,
        }

        chunker = strategy_map.get(self.config.strategy, self._chunk_recursive)
        chunks = chunker(text, document_id, source, page)

        logger.info(
            "document_chunked",
            document_id=document_id,
            chunk_count=len(chunks),
            strategy=self.config.strategy.value,
        )

        return chunks

    def _chunk_fixed_size(
        self, text: str, document_id: str, source: str, page: Optional[int]
    ) -> List[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        index = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for space within last 50 chars
                last_space = text.rfind(" ", start, end + 50)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_index=index,
                        document_id=document_id,
                        source=source,
                        page=page,
                    )
                )
                index += 1

            start = end - overlap
            if start < 0:
                start = 0

        return self._filter_small_chunks(chunks)

    def _chunk_recursive(
        self, text: str, document_id: str, source: str, page: Optional[int]
    ) -> List[Chunk]:
        """Recursive text splitting respecting boundaries."""
        chunks = []
        index = 0

        # Split by paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If single paragraph exceeds chunk size, split it
            if len(para) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(
                        Chunk(
                            content=current_chunk.strip(),
                            chunk_index=index,
                            document_id=document_id,
                            source=source,
                            page=page,
                        )
                    )
                    index += 1
                    current_chunk = ""

                # Split long paragraph
                sub_chunks = self._split_long_text(para, document_id, source, page, index)
                chunks.extend(sub_chunks)
                index += len(sub_chunks)
            elif len(current_chunk) + len(para) + 2 > self.config.chunk_size:
                # Current chunk is full, start new one
                if current_chunk:
                    chunks.append(
                        Chunk(
                            content=current_chunk.strip(),
                            chunk_index=index,
                            document_id=document_id,
                            source=source,
                            page=page,
                        )
                    )
                    index += 1
                current_chunk = para
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    chunk_index=index,
                    document_id=document_id,
                    source=source,
                    page=page,
                )
            )

        return self._filter_small_chunks(chunks)

    def _split_long_text(
        self, text: str, document_id: str, source: str, page: Optional[int], start_index: int
    ) -> List[Chunk]:
        """Split long text into smaller chunks."""
        chunks = []
        index = start_index

        # Try splitting by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.config.chunk_size:
                if current_chunk:
                    chunks.append(
                        Chunk(
                            content=current_chunk.strip(),
                            chunk_index=index,
                            document_id=document_id,
                            source=source,
                            page=page,
                        )
                    )
                    index += 1
                    current_chunk = ""

            if len(sentence) > self.config.chunk_size:
                # Very long sentence, split by words
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 > self.config.chunk_size:
                        if current_chunk:
                            chunks.append(
                                Chunk(
                                    content=current_chunk.strip(),
                                    chunk_index=index,
                                    document_id=document_id,
                                    source=source,
                                    page=page,
                                )
                            )
                            index += 1
                            current_chunk = ""
                    current_chunk += " " + word if current_chunk else word
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(
                Chunk(
                    content=current_chunk.strip(),
                    chunk_index=index,
                    document_id=document_id,
                    source=source,
                    page=page,
                )
            )

        return chunks

    def _chunk_semantic(
        self, text: str, document_id: str, source: str, page: Optional[int]
    ) -> List[Chunk]:
        """Semantic chunking based on topic changes (simplified)."""
        # This is a simplified version - in production, could use embeddings
        # to detect topic changes
        return self._chunk_recursive(text, document_id, source, page)

    def _chunk_markdown(
        self, text: str, document_id: str, source: str, page: Optional[int]
    ) -> List[Chunk]:
        """Chunk based on markdown headers."""
        chunks = []
        index = 0

        # Split by headers
        lines = text.split("\n")
        current_section = ""
        current_header = ""

        for line in lines:
            # Check if it's a header
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save previous section
                if current_section.strip():
                    chunks.append(
                        Chunk(
                            content=f"{current_header}\n\n{current_section.strip()}",
                            chunk_index=index,
                            document_id=document_id,
                            source=source,
                            page=page,
                        )
                    )
                    index += 1

                current_header = line
                current_section = ""
            else:
                current_section += "\n" + line

        # Add final section
        if current_section.strip():
            chunks.append(
                Chunk(
                    content=f"{current_header}\n\n{current_section.strip()}",
                    chunk_index=index,
                    document_id=document_id,
                    source=source,
                    page=page,
                )
            )

        # If no headers found, fall back to recursive
        if not chunks:
            return self._chunk_recursive(text, document_id, source, page)

        return self._filter_small_chunks(chunks)

    def _filter_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filter out chunks that are too small."""
        if not self.config.min_chunk_size:
            return chunks

        filtered = []
        for chunk in chunks:
            if len(chunk.content) >= self.config.min_chunk_size:
                filtered.append(chunk)
            elif filtered:
                # Merge small chunk into previous
                filtered[-1].content += "\n\n" + chunk.content

        return filtered