"""
PDF Extractor

Extracts text, structure, and metadata from PDF documents.
Stdlib only — uses pdftotext (poppler) via subprocess.
macOS: brew install poppler. Linux: apt install poppler-utils.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails"""
    pass


@dataclass
class Section:
    """Represents a document section"""
    heading: str
    level: int
    content: str
    line_number: int
    subsections: List['Section']


@dataclass
class CodeBlock:
    """Represents a code block"""
    language: Optional[str]
    code: str
    line_number: Optional[int]
    context: str


@dataclass
class ExtractedContent:
    """Structured extracted content"""
    title: str
    sections: List[Section]
    code_blocks: List[CodeBlock]
    metadata: Dict[str, Any]
    source_url: Optional[str]
    extraction_date: datetime
    raw_text: str


class PDFExtractor:
    """Extracts content from PDF files with structure preservation"""

    def __init__(self):
        """Initialize PDF extractor"""
        self.heading_patterns = [
            re.compile(r'^(\d+\.)+\s+[A-Z]'),  # 1.1 Title
            re.compile(r'^[A-Z][A-Z\s]+$'),     # ALL CAPS TITLE
            re.compile(r'^Abstract\s*$', re.IGNORECASE),
            re.compile(r'^Introduction\s*$', re.IGNORECASE),
            re.compile(r'^Conclusion\s*$', re.IGNORECASE),
            re.compile(r'^References\s*$', re.IGNORECASE),
        ]

        self.code_indicators = [
            'algorithm', 'procedure', 'function', 'def ', 'class ',
            'import ', 'for(', 'while(', 'if(', '{', '}', ';'
        ]

    def extract(self, pdf_path: str) -> ExtractedContent:
        """
        Extract content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractedContent object with structured data

        Raises:
            PDFExtractionError: If extraction fails
            FileNotFoundError: If PDF file doesn't exist
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not path.suffix.lower() == '.pdf':
            raise PDFExtractionError(f"Not a PDF file: {pdf_path}")

        logger.info(f"Extracting content from PDF: {pdf_path}")

        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=120, check=False,
            )
            if result.returncode != 0:
                raise PDFExtractionError(
                    f"pdftotext failed (code {result.returncode}). "
                    "Install: brew install poppler (macOS) or apt install poppler-utils (Linux)"
                )
            raw_text = result.stdout
        except FileNotFoundError:
            raise PDFExtractionError(
                "pdftotext not found. "
                "Install: brew install poppler (macOS) or apt install poppler-utils (Linux)"
            )

        if not raw_text.strip():
            raise PDFExtractionError("No text content extracted from PDF (may be image-based)")

        logger.info(f"Extracted {len(raw_text)} characters from PDF")

        # Extract metadata via pdfinfo if available
        metadata = self._extract_pdfinfo(pdf_path)

        return self._process_extracted_text(raw_text, metadata, pdf_path)

    def _extract_pdfinfo(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata using pdfinfo (poppler)."""
        metadata = {}
        try:
            result = subprocess.run(
                ["pdfinfo", str(pdf_path)],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, _, value = line.partition(':')
                        key = key.strip().lower().replace(' ', '_')
                        metadata[key] = value.strip()
        except FileNotFoundError:
            pass  # pdfinfo not available — not critical
        return metadata

    def _process_extracted_text(
        self,
        raw_text: str,
        metadata: Dict[str, Any],
        pdf_path: str
    ) -> ExtractedContent:
        """Process raw extracted text into structured content"""

        title = self._extract_title(raw_text, metadata)
        sections = self._extract_sections(raw_text)
        code_blocks = self._extract_code_blocks(raw_text)

        full_metadata = {
            **metadata,
            'file_name': Path(pdf_path).name,
            'file_path': pdf_path,
            'num_sections': len(sections),
            'num_code_blocks': len(code_blocks),
        }

        return ExtractedContent(
            title=title,
            sections=sections,
            code_blocks=code_blocks,
            metadata=full_metadata,
            source_url=None,
            extraction_date=datetime.now(),
            raw_text=raw_text
        )

    def _extract_title(self, text: str, metadata: Dict[str, Any]) -> str:
        """Extract document title"""
        if metadata.get('title'):
            title = metadata['title'].strip()
            if title and title.lower() != 'untitled':
                return title

        lines = text.split('\n')
        for line in lines[:20]:
            line = line.strip()
            if 10 < len(line) < 200:
                if not line.startswith('---'):
                    return line

        return "Untitled Document"

    def _extract_sections(self, text: str) -> List[Section]:
        """Extract document sections with headings"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            is_heading, level = self._is_heading(stripped)

            if is_heading:
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                current_section = Section(
                    heading=stripped,
                    level=level,
                    content='',
                    line_number=i,
                    subsections=[]
                )
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        logger.info(f"Extracted {len(sections)} sections")
        return sections

    def _is_heading(self, line: str) -> Tuple[bool, int]:
        """Determine if a line is a heading and its level."""
        if not line or len(line) < 3:
            return False, 0

        for pattern in self.heading_patterns:
            if pattern.match(line):
                if line[0].isdigit():
                    level = line.split()[0].count('.') + 1
                else:
                    level = 1
                return True, level

        if line.isupper() and 3 < len(line) < 50 and ' ' in line:
            return True, 1

        return False, 0

    def _extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """Extract code blocks from text"""
        code_blocks = []
        lines = text.split('\n')

        in_code_block = False
        current_code = []
        code_start_line = 0
        context = ''

        for i, line in enumerate(lines):
            is_code = self._is_code_line(line)

            if is_code and not in_code_block:
                in_code_block = True
                code_start_line = i
                current_code = [line]
                if i > 0:
                    context = lines[i - 1].strip()
            elif is_code and in_code_block:
                current_code.append(line)
            elif not is_code and in_code_block:
                if len(current_code) > 2:
                    code_blocks.append(CodeBlock(
                        language=self._detect_language('\n'.join(current_code)),
                        code='\n'.join(current_code),
                        line_number=code_start_line,
                        context=context
                    ))
                in_code_block = False
                current_code = []
                context = ''

        if in_code_block and len(current_code) > 2:
            code_blocks.append(CodeBlock(
                language=self._detect_language('\n'.join(current_code)),
                code='\n'.join(current_code),
                line_number=code_start_line,
                context=context
            ))

        logger.info(f"Extracted {len(code_blocks)} code blocks")
        return code_blocks

    def _is_code_line(self, line: str) -> bool:
        """Check if a line looks like code"""
        stripped = line.strip()

        if not stripped:
            return False

        for indicator in self.code_indicators:
            if indicator in stripped.lower():
                return True

        if line.startswith('    ') or line.startswith('\t'):
            return True

        if re.search(r'[=\+\-\*\/]{2,}', stripped):
            return True
        if re.search(r'[\(\)\{\}\[\];]', stripped):
            return True
        if re.search(r'^\s*\d+[\.\)]\s+', stripped):
            return True

        return False

    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language from code"""
        code_lower = code.lower()

        language_indicators = {
            'python': ['def ', 'import ', 'from ', 'print(', '__init__', 'self.'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'console.'],
            'java': ['public class', 'private ', 'void ', 'System.out'],
            'c++': ['#include', 'cout', 'std::', 'namespace'],
            'c': ['#include', 'printf', 'int main'],
            'rust': ['fn ', 'let mut', 'impl ', 'pub '],
            'go': ['func ', 'package ', 'import (', ':='],
            'pseudocode': ['algorithm', 'procedure', 'begin', 'end', 'step '],
        }

        scores = {lang: 0 for lang in language_indicators}

        for lang, indicators in language_indicators.items():
            for indicator in indicators:
                if indicator in code_lower:
                    scores[lang] += 1

        max_score = max(scores.values())
        if max_score > 0:
            detected = max(scores, key=scores.get)
            logger.debug(f"Detected language: {detected} (score: {max_score})")
            return detected

        return None

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract only metadata from PDF."""
        return self._extract_pdfinfo(pdf_path)
