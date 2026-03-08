"""
Web Extractor

Fetches and extracts content from web pages and online documentation.
Removes boilerplate, extracts code blocks, and preserves article structure.
Stdlib only — uses urllib and html.parser.
"""

import logging
import re
import time
from html.parser import HTMLParser
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urlparse, urljoin
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass

from .pdf_extractor import ExtractedContent, Section, CodeBlock

logger = logging.getLogger(__name__)


class WebExtractionError(Exception):
    """Raised when web extraction fails"""
    pass


# ---------------------------------------------------------------------------
# Lightweight HTML parser using stdlib html.parser
# ---------------------------------------------------------------------------

class _HTMLContentParser(HTMLParser):
    """Parses HTML into a simple tree of elements for content extraction."""

    SKIP_TAGS = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'svg'}
    BLOCK_TAGS = {'p', 'div', 'section', 'article', 'main', 'li', 'tr', 'blockquote', 'figcaption'}

    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta = {}         # name/property → content
        self.headings = []     # (level, text)
        self.code_blocks = []  # (language, code, context)
        self.text_parts = []   # all visible text in order

        self._tag_stack = []
        self._skip_depth = 0
        self._in_title = False
        self._in_heading = 0   # 0 = not in heading, 1-6 = h level
        self._heading_text = []
        self._in_pre = False
        self._in_code = False
        self._code_text = []
        self._code_lang = None
        self._pre_context = ""

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_dict = dict(attrs)
        self._tag_stack.append(tag)

        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        if tag == 'title':
            self._in_title = True

        if tag == 'meta':
            name = attrs_dict.get('name', attrs_dict.get('property', '')).lower()
            content = attrs_dict.get('content', '')
            if name and content:
                self.meta[name] = content

        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self._in_heading = int(tag[1])
            self._heading_text = []

        if tag == 'pre':
            self._in_pre = True
            self._code_text = []
            self._code_lang = None
            # Capture last text part as context
            if self.text_parts:
                self._pre_context = self.text_parts[-1][:100]

        if tag == 'code':
            self._in_code = True
            # Detect language from class
            classes = attrs_dict.get('class', '')
            for cls in classes.split():
                if cls.startswith('language-'):
                    self._code_lang = cls[9:]
                    break
                elif cls.startswith('lang-'):
                    self._code_lang = cls[5:]
                    break

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

        if self._skip_depth > 0:
            if self._tag_stack and self._tag_stack[-1] == tag:
                self._tag_stack.pop()
            return

        if tag == 'title':
            self._in_title = False

        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6') and self._in_heading:
            heading_text = ' '.join(self._heading_text).strip()
            if heading_text:
                self.headings.append((self._in_heading, heading_text))
                self.text_parts.append(heading_text)
            self._in_heading = 0
            self._heading_text = []

        if tag == 'pre':
            code = '\n'.join(self._code_text).strip()
            if code and len(code) > 10:
                self.code_blocks.append((self._code_lang, code, self._pre_context))
            self._in_pre = False
            self._code_text = []
            self._pre_context = ""

        if tag == 'code':
            if not self._in_pre:
                # Inline code — skip for now
                pass
            self._in_code = False

        if tag in self.BLOCK_TAGS:
            self.text_parts.append("")  # paragraph separator

        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

    def handle_data(self, data):
        if self._skip_depth > 0:
            return

        if self._in_title and not self.title:
            self.title = data.strip()

        if self._in_heading:
            self._heading_text.append(data)
            return

        if self._in_pre or self._in_code:
            self._code_text.append(data)
            return

        text = data.strip()
        if text:
            self.text_parts.append(text)

    def handle_entityref(self, name):
        entities = {'amp': '&', 'lt': '<', 'gt': '>', 'quot': '"', 'apos': "'",
                     'nbsp': ' ', 'mdash': '—', 'ndash': '–'}
        self.handle_data(entities.get(name, f'&{name};'))

    def handle_charref(self, name):
        try:
            if name.startswith('x'):
                char = chr(int(name[1:], 16))
            else:
                char = chr(int(name))
            self.handle_data(char)
        except (ValueError, OverflowError):
            pass


class WebExtractor:
    """Extracts content from web pages with boilerplate removal"""

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; ml-paper-research-skill/1.0)"

    def extract(self, url: str) -> ExtractedContent:
        """Extract content from a web page."""
        logger.info(f"Extracting content from URL: {url}")

        if not self._is_valid_url(url):
            raise WebExtractionError(f"Invalid URL: {url}")

        html = self._fetch_html(url)

        parser = _HTMLContentParser()
        parser.feed(html)

        title = parser.title or parser.meta.get('og:title', 'Untitled Article')

        # Build sections from headings + text between them
        sections = self._build_sections(parser.headings, parser.text_parts)

        # Build code blocks
        code_blocks = []
        for lang, code, context in parser.code_blocks:
            code_blocks.append(CodeBlock(
                language=lang,
                code=code,
                line_number=None,
                context=context
            ))

        # Build metadata
        metadata = {
            'url': url,
            'title': title,
            'author': parser.meta.get('author', parser.meta.get('og:author', '')),
            'description': parser.meta.get('description', parser.meta.get('og:description', '')),
        }

        raw_text = '\n'.join(part for part in parser.text_parts if part)

        return ExtractedContent(
            title=title,
            sections=sections,
            code_blocks=code_blocks,
            metadata=metadata,
            source_url=url,
            extraction_date=datetime.now(),
            raw_text=raw_text
        )

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            return False

    def _fetch_html(self, url: str) -> str:
        """Fetch HTML content with retries."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Fetching URL (attempt {attempt}/{self.max_retries})")
                req = Request(url, headers={'User-Agent': self.user_agent})
                with urlopen(req, timeout=self.timeout) as resp:
                    charset = resp.headers.get_content_charset() or 'utf-8'
                    html = resp.read().decode(charset, errors='replace')
                logger.info(f"Successfully fetched {len(html)} characters")
                return html

            except HTTPError as e:
                if e.code == 404:
                    raise WebExtractionError(f"Page not found (404): {url}")
                elif e.code == 403:
                    raise WebExtractionError(f"Access forbidden (403): {url}")
                elif e.code >= 500:
                    last_error = e
                    logger.warning(f"Server error {e.code} on attempt {attempt}")
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                else:
                    raise WebExtractionError(f"HTTP error {e.code}: {url}")

            except URLError as e:
                last_error = e
                logger.warning(f"Request failed on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

            except Exception as e:
                last_error = e
                logger.warning(f"Unexpected error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise WebExtractionError(f"Failed to fetch URL after {self.max_retries} attempts: {last_error}")

    def _build_sections(self, headings: list, text_parts: list) -> List[Section]:
        """Build sections from extracted headings and text."""
        if not headings:
            # No headings found — return one big section
            text = '\n'.join(part for part in text_parts if part)
            if text:
                return [Section(
                    heading="Content",
                    level=1,
                    content=text[:2000],
                    line_number=0,
                    subsections=[]
                )]
            return []

        sections = []
        full_text = '\n'.join(text_parts)

        for i, (level, heading_text) in enumerate(headings):
            # Find content between this heading and the next
            start_idx = full_text.find(heading_text)
            if start_idx < 0:
                continue
            start_idx += len(heading_text)

            if i + 1 < len(headings):
                end_idx = full_text.find(headings[i + 1][1], start_idx)
                if end_idx < 0:
                    end_idx = len(full_text)
            else:
                end_idx = len(full_text)

            content = full_text[start_idx:end_idx].strip()

            sections.append(Section(
                heading=heading_text,
                level=level,
                content=content[:2000],
                line_number=0,
                subsections=[]
            ))

        logger.info(f"Extracted {len(sections)} sections")
        return sections

    def extract_code_blocks(self, url: str) -> List[CodeBlock]:
        """Extract only code blocks from a web page."""
        logger.info(f"Extracting code blocks from: {url}")
        content = self.extract(url)
        return content.code_blocks

    def crawl_documentation(
        self,
        base_url: str,
        max_pages: int = 10,
        follow_pattern: Optional[str] = None
    ) -> List[ExtractedContent]:
        """Crawl multi-page documentation."""
        logger.info(f"Starting documentation crawl from: {base_url}")

        visited = set()
        to_visit = [base_url]
        results = []

        pattern = re.compile(follow_pattern) if follow_pattern else None

        while to_visit and len(results) < max_pages:
            url = to_visit.pop(0)

            if url in visited:
                continue

            visited.add(url)

            try:
                content = self.extract(url)
                results.append(content)
                logger.info(f"Crawled {len(results)}/{max_pages}: {url}")

                # Find links to follow
                if pattern:
                    html = self._fetch_html(url)
                    for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
                        href = match.group(1)
                        absolute_url = urljoin(url, href)
                        if absolute_url not in visited and pattern.match(absolute_url):
                            to_visit.append(absolute_url)

                time.sleep(1)

            except Exception as e:
                logger.error(f"Failed to crawl {url}: {e}")
                continue

        logger.info(f"Crawling complete. Extracted {len(results)} pages")
        return results
