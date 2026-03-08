"""
Notebook Extractor

Parses Jupyter notebooks and extracts code, markdown, and outputs.
Stdlib only — notebooks are JSON files, parsed with json module.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .pdf_extractor import ExtractedContent, Section, CodeBlock

logger = logging.getLogger(__name__)


class NotebookExtractionError(Exception):
    """Raised when notebook extraction fails"""
    pass


class NotebookExtractor:
    """Extracts content from Jupyter notebooks"""

    def __init__(self):
        """Initialize notebook extractor"""
        pass

    def extract(self, notebook_path: str) -> ExtractedContent:
        """
        Extract content from a Jupyter notebook.

        Args:
            notebook_path: Path to the .ipynb file

        Returns:
            ExtractedContent object with cells and outputs

        Raises:
            NotebookExtractionError: If parsing fails
        """
        path = Path(notebook_path)
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        if not path.suffix.lower() == '.ipynb':
            raise NotebookExtractionError(f"Not a notebook file: {notebook_path}")

        logger.info(f"Extracting notebook: {notebook_path}")

        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            raise NotebookExtractionError(f"Failed to read notebook: {e}")

        cells = nb.get("cells", [])

        # Extract title from metadata or first markdown cell
        title = self._extract_title(nb, cells)

        # Extract sections from markdown cells
        sections = []
        code_blocks = []
        raw_text_parts = []

        for i, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", "")
            # source can be a string or list of strings
            if isinstance(source, list):
                source = "".join(source)

            if cell_type == 'markdown':
                section = self._process_markdown_cell(source, i)
                if section:
                    sections.append(section)
                    raw_text_parts.append(f"## {section.heading}\n{section.content}")

            elif cell_type == 'code':
                code_block = self._process_code_cell(cell, source, i)
                if code_block:
                    code_blocks.append(code_block)
                    raw_text_parts.append(f"```python\n{code_block.code}\n```")

        # Extract metadata
        metadata = self._extract_metadata(nb, notebook_path)

        # Extract dependencies from code cells
        dependencies = self._extract_dependencies(cells)
        metadata['dependencies'] = dependencies

        raw_text = '\n\n'.join(raw_text_parts)

        logger.info(f"Extracted {len(sections)} sections and {len(code_blocks)} code blocks")

        return ExtractedContent(
            title=title,
            sections=sections,
            code_blocks=code_blocks,
            metadata=metadata,
            source_url=None,
            extraction_date=datetime.now(),
            raw_text=raw_text
        )

    def _extract_title(self, nb: dict, cells: list) -> str:
        """Extract title from notebook"""
        # Try metadata first
        nb_meta = nb.get("metadata", {})
        if "title" in nb_meta:
            return nb_meta["title"]

        # Look for title in first markdown cell
        for cell in cells:
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                for line in source.split('\n'):
                    if line.startswith('#'):
                        title = line.lstrip('#').strip()
                        if title:
                            return title

        return "Untitled Notebook"

    def _process_markdown_cell(self, source: str, cell_num: int) -> Optional[Section]:
        """Process markdown cell into a section"""
        content = source.strip()

        if not content:
            return None

        lines = content.split('\n')
        if lines[0].startswith('#'):
            heading_line = lines[0]
            level = len(heading_line) - len(heading_line.lstrip('#'))
            heading = heading_line.lstrip('#').strip()
            body = '\n'.join(lines[1:]).strip()

            return Section(
                heading=heading,
                level=level,
                content=body,
                line_number=cell_num,
                subsections=[]
            )

        return Section(
            heading=f"Cell {cell_num}",
            level=3,
            content=content,
            line_number=cell_num,
            subsections=[]
        )

    def _process_code_cell(self, cell: dict, source: str, cell_num: int) -> Optional[CodeBlock]:
        """Process code cell into a code block"""
        code = source.strip()

        if not code:
            return None

        # Extract language from cell metadata
        language = 'python'  # Default for Jupyter
        cell_meta = cell.get("metadata", {})
        if "language" in cell_meta:
            language = cell_meta["language"]

        # Get output as context
        context = ''
        outputs = cell.get("outputs", [])
        if outputs:
            output_texts = []
            for output in outputs[:3]:
                if isinstance(output, dict):
                    if "text" in output:
                        text = output["text"]
                        if isinstance(text, list):
                            text = "".join(text)
                        output_texts.append(str(text)[:100])
                    elif "data" in output and "text/plain" in output["data"]:
                        text = output["data"]["text/plain"]
                        if isinstance(text, list):
                            text = "".join(text)
                        output_texts.append(str(text)[:100])

            if output_texts:
                context = ' | '.join(output_texts)

        return CodeBlock(
            language=language,
            code=code,
            line_number=cell_num,
            context=context
        )

    def _extract_metadata(self, nb: dict, notebook_path: str) -> Dict[str, Any]:
        """Extract notebook metadata"""
        cells = nb.get("cells", [])
        metadata = {
            'file_name': Path(notebook_path).name,
            'file_path': notebook_path,
            'num_cells': len(cells),
        }

        nb_meta = nb.get("metadata", {})

        if 'kernelspec' in nb_meta:
            kernel = nb_meta['kernelspec']
            metadata['kernel_name'] = kernel.get('name', 'unknown')
            metadata['kernel_display_name'] = kernel.get('display_name', 'unknown')

        if 'language_info' in nb_meta:
            lang_info = nb_meta['language_info']
            metadata['language'] = lang_info.get('name', 'unknown')
            metadata['language_version'] = lang_info.get('version', 'unknown')

        return metadata

    def extract_code_cells(self, notebook_path: str) -> List[CodeBlock]:
        """Extract only code cells"""
        content = self.extract(notebook_path)
        return content.code_blocks

    def _extract_dependencies(self, cells: list) -> List[str]:
        """Extract imported libraries and dependencies."""
        dependencies = set()
        import_pattern = re.compile(
            r'^\s*(?:from\s+(\S+)\s+)?import\s+(\S+)',
            re.MULTILINE
        )

        for cell in cells:
            if cell.get("cell_type") == 'code':
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                matches = import_pattern.findall(source)
                for match in matches:
                    dep = match[0] if match[0] else match[1]
                    root_dep = dep.split('.')[0]
                    dependencies.add(root_dep)

        logger.debug(f"Extracted dependencies: {dependencies}")
        return sorted(list(dependencies))

    def extract_dependencies(self, notebook_path: str) -> List[str]:
        """Extract dependencies from a notebook file path."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read notebook for dependencies: {e}")
            return []
        return self._extract_dependencies(nb.get("cells", []))
