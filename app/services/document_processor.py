"""Smart document processing with structure-aware chunking."""

import io
import re
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from app.config import settings
from app.services.embedding import embedding_service


@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""

    content: str
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    token_count: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.token_count:
            self.token_count = len(self.content.split())


@dataclass
class ProcessedDocument:
    """Result of document processing."""

    chunks: list[DocumentChunk]
    total_pages: int
    total_tokens: int
    metadata: dict


class DocumentProcessor:
    """Smart PDF processor with structure-aware chunking."""

    def __init__(
        self,
        max_chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.max_chunk_size = max_chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    async def process_pdf(
        self, file_content: bytes, filename: str
    ) -> ProcessedDocument:
        """Process a PDF file into structured chunks.

        Args:
            file_content: PDF file bytes
            filename: Original filename

        Returns:
            ProcessedDocument with chunks and metadata
        """
        # Extract text with structure using PyMuPDF
        doc = fitz.open(stream=file_content, filetype="pdf")

        pages_content = []
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            page_data = self._extract_page_content(page, page_num + 1)
            pages_content.append(page_data)

        doc.close()

        # Extract tables using pdfplumber
        tables = self._extract_tables(file_content)

        # Create chunks with structure awareness
        chunks = self._create_chunks(pages_content, tables)

        # Calculate total tokens
        total_tokens = sum(c.token_count for c in chunks)

        return ProcessedDocument(
            chunks=chunks,
            total_pages=total_pages,
            total_tokens=total_tokens,
            metadata={
                "filename": filename,
                "page_count": total_pages,
                "chunk_count": len(chunks),
            },
        )

    def _extract_page_content(
        self, page: fitz.Page, page_number: int
    ) -> dict:
        """Extract structured content from a page."""
        blocks = page.get_text("dict")["blocks"]

        content_parts = []
        headers = []

        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    text = ""
                    font_size = 0
                    is_bold = False

                    for span in line.get("spans", []):
                        text += span["text"]
                        font_size = max(font_size, span["size"])
                        if "bold" in span.get("font", "").lower():
                            is_bold = True

                    text = text.strip()
                    if not text:
                        continue

                    # Detect headers based on font size and style
                    if font_size > 14 or is_bold:
                        if len(text) < 200:  # Likely a header
                            headers.append({
                                "text": text,
                                "font_size": font_size,
                                "position": len(content_parts),
                            })

                    content_parts.append({
                        "text": text,
                        "is_header": font_size > 14 or is_bold,
                        "font_size": font_size,
                    })

        return {
            "page_number": page_number,
            "content_parts": content_parts,
            "headers": headers,
        }

    def _extract_tables(self, file_content: bytes) -> list[dict]:
        """Extract tables from PDF using pdfplumber."""
        tables = []

        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()

                    for table in page_tables:
                        if table and len(table) > 1:
                            # Convert table to markdown format
                            markdown = self._table_to_markdown(table)
                            if markdown:
                                tables.append({
                                    "page_number": page_num + 1,
                                    "content": markdown,
                                    "rows": len(table),
                                    "cols": len(table[0]) if table else 0,
                                })
        except Exception:
            # If table extraction fails, continue without tables
            pass

        return tables

    def _table_to_markdown(self, table: list[list]) -> str:
        """Convert table data to markdown format."""
        if not table or not table[0]:
            return ""

        lines = []

        # Header row
        header = [str(cell or "").strip() for cell in table[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Data rows
        for row in table[1:]:
            cells = [str(cell or "").strip() for cell in row]
            # Pad row if needed
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells[:len(header)]) + " |")

        return "\n".join(lines)

    def _create_chunks(
        self, pages_content: list[dict], tables: list[dict]
    ) -> list[DocumentChunk]:
        """Create chunks with structure awareness."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_header = None
        current_page = 1

        # Process page content
        for page_data in pages_content:
            page_number = page_data["page_number"]

            for part in page_data["content_parts"]:
                text = part["text"]
                tokens = len(text.split())

                # Update current header if this is a header
                if part["is_header"] and len(text) < 200:
                    current_header = text

                # Check if adding this would exceed chunk size
                if current_tokens + tokens > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        page_number=current_page,
                        section_header=current_header,
                        token_count=current_tokens,
                    ))

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_tokens = len(overlap_text.split()) if overlap_text else 0
                    current_page = page_number

                current_chunk.append(text)
                current_tokens += tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(
                content=chunk_text,
                page_number=current_page,
                section_header=current_header,
                token_count=current_tokens,
            ))

        # Add tables as separate chunks
        for table in tables:
            chunks.append(DocumentChunk(
                content=table["content"],
                page_number=table["page_number"],
                section_header="Table",
                metadata={"type": "table", "rows": table["rows"], "cols": table["cols"]},
            ))

        return chunks

    def _get_overlap(self, chunk_parts: list[str]) -> str:
        """Get overlap text from the end of a chunk."""
        if not chunk_parts:
            return ""

        # Join and split to get word-level overlap
        full_text = " ".join(chunk_parts)
        words = full_text.split()

        if len(words) <= self.chunk_overlap:
            return full_text

        overlap_words = words[-self.chunk_overlap:]
        return " ".join(overlap_words)

    async def process_and_embed(
        self, file_content: bytes, filename: str
    ) -> tuple[ProcessedDocument, list[list[float]]]:
        """Process document and generate embeddings for all chunks.

        Args:
            file_content: PDF file bytes
            filename: Original filename

        Returns:
            Tuple of (ProcessedDocument, list of embeddings)
        """
        # Process document
        processed = await self.process_pdf(file_content, filename)

        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in processed.chunks]
        embeddings = await embedding_service.get_embeddings_batch(texts)

        return processed, embeddings


# Clean text utility
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s.,!?;:\-'\"()\[\]{}]", "", text)
    return text.strip()


# Global processor instance
document_processor = DocumentProcessor()
