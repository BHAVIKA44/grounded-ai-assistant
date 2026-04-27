"""
Document parsing service for PDF, TXT, and DOCX files.
"""

import io
import re
from typing import List, Optional

from pypdf import PdfReader
from python_docx import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """Parse various document formats into text."""

    @staticmethod
    def parse_pdf(file_bytes: bytes, filename: str) -> List[tuple[str, int]]:
        """
        Parse PDF file and extract text with page numbers.
        
        Args:
            file_bytes: Raw PDF file bytes
            filename: Original filename for logging
            
        Returns:
            List of tuples (text, page_number)
        """
        logger.info("parsing_pdf", filename=filename)
        pages = []

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    # Clean up extracted text
                    text = DocumentParser._clean_text(text)
                    pages.append((text, page_num))

            logger.info("pdf_parsed", filename=filename, page_count=len(pages))
            return pages

        except Exception as e:
            logger.error("pdf_parse_error", filename=filename, error=str(e))
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    @staticmethod
    def parse_txt(file_bytes: bytes, filename: str) -> List[tuple[str, Optional[int]]]:
        """
        Parse plain text file.
        
        Args:
            file_bytes: Raw text file bytes
            filename: Original filename for logging
            
        Returns:
            List of tuples (text, None for page)
        """
        logger.info("parsing_txt", filename=filename)

        try:
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    text = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # Fallback to utf-8 with error replacement
                text = file_bytes.decode("utf-8", errors="replace")

            text = DocumentParser._clean_text(text)
            logger.info("txt_parsed", filename=filename, length=len(text))
            return [(text, None)]

        except Exception as e:
            logger.error("txt_parse_error", filename=filename, error=str(e))
            raise ValueError(f"Failed to parse text file: {str(e)}")

    @staticmethod
    def parse_docx(file_bytes: bytes, filename: str) -> List[tuple[str, Optional[int]]]:
        """
        Parse DOCX file and extract text.
        
        Args:
            file_bytes: Raw DOCX file bytes
            filename: Original filename for logging
            
        Returns:
            List of tuples (text, None for page)
        """
        logger.info("parsing_docx", filename=filename)

        try:
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            # Combine all paragraphs
            text = "\n\n".join(paragraphs)
            text = DocumentParser._clean_text(text)

            logger.info("docx_parsed", filename=filename, paragraph_count=len(paragraphs))
            return [(text, None)]

        except Exception as e:
            logger.error("docx_parse_error", filename=filename, error=str(e))
            raise ValueError(f"Failed to parse DOCX: {str(e)}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")

        # Normalize dashes
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        return text.strip()


def parse_document(file_bytes: bytes, filename: str, file_extension: str) -> List[tuple[str, Optional[int]]]:
    """
    Parse document based on file extension.
    
    Args:
        file_bytes: Raw file bytes
        filename: Original filename
        file_extension: File extension (pdf, txt, docx)
        
    Returns:
        List of tuples (text, page_number)
        
    Raises:
        ValueError: If file type is not supported
    """
    extension = file_extension.lower().lstrip(".")

    if extension == "pdf":
        return DocumentParser.parse_pdf(file_bytes, filename)
    elif extension == "txt":
        return DocumentParser.parse_txt(file_bytes, filename)
    elif extension in ["docx", "doc"]:
        return DocumentParser.parse_docx(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {extension}")