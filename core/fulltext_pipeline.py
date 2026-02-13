"""
Full-Text PDF Pipeline - Download, extract, and chunk academic papers.

Replicates SciSpace's ability to read and understand full paper content.
Sources for full text (in priority order):
1. CORE API (free full text for 140M+ papers)
2. Unpaywall (open access PDF links via DOI)
3. arXiv (direct PDF download)
4. PubMed Central (free XML full text)

Pipeline: DOI/URL -> Download PDF -> Extract text -> Chunk for LLM analysis
"""

import asyncio
import aiohttp
import re
import os
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class FullTextPaper:
    """A paper with extracted full text content."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    doi: str
    abstract: str
    full_text: str = ""
    sections: List[Dict] = field(default_factory=list)  # [{heading, content}]
    source_type: str = ""  # core, unpaywall, arxiv, pmc
    pdf_url: str = ""
    word_count: int = 0
    chunks: List[str] = field(default_factory=list)  # Pre-chunked for LLM

    def to_dict(self) -> Dict:
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'doi': self.doi,
            'abstract': self.abstract,
            'full_text_available': bool(self.full_text),
            'source_type': self.source_type,
            'word_count': self.word_count,
            'section_count': len(self.sections),
            'chunk_count': len(self.chunks)
        }


class FullTextPipeline:
    """
    Downloads and extracts full text from academic papers.
    Tries multiple open access sources in priority order.
    """

    def __init__(self, cache_dir: str = ".cache/fulltext"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = 2000  # tokens per chunk (approx words)
        self.chunk_overlap = 200

    async def get_full_text(self, paper: Dict,
                            session: aiohttp.ClientSession = None) -> Optional[FullTextPaper]:
        """
        Try to get full text for a paper from multiple sources.
        Returns FullTextPaper with extracted text and chunks.
        """
        doi = paper.get('doi', '')
        arxiv_id = paper.get('arxiv_id', '')
        pubmed_id = paper.get('pubmed_id', '')
        title = paper.get('title', '')

        # Check cache first
        cache_key = doi or arxiv_id or hashlib.md5(title.encode()).hexdigest()[:16]
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        _session = session or aiohttp.ClientSession()
        full_text = None
        source_type = ""
        pdf_url = ""

        try:
            # Strategy 1: CORE API (free full text)
            if not full_text:
                full_text, pdf_url = await self._try_core(doi, title, _session)
                if full_text:
                    source_type = "core"

            # Strategy 2: Unpaywall (OA PDF link)
            if not full_text and doi:
                full_text, pdf_url = await self._try_unpaywall(doi, _session)
                if full_text:
                    source_type = "unpaywall"

            # Strategy 3: arXiv direct PDF
            if not full_text and arxiv_id:
                full_text, pdf_url = await self._try_arxiv(arxiv_id, _session)
                if full_text:
                    source_type = "arxiv"

            # Strategy 4: PubMed Central XML
            if not full_text and pubmed_id:
                full_text, pdf_url = await self._try_pmc(pubmed_id, _session)
                if full_text:
                    source_type = "pmc"

        finally:
            if not session:
                await _session.close()

        if not full_text:
            return None

        # Build FullTextPaper
        sections = self._extract_sections(full_text)
        chunks = self._chunk_text(full_text)

        ft_paper = FullTextPaper(
            paper_id=paper.get('paper_id', ''),
            title=title,
            authors=paper.get('authors', []),
            year=paper.get('year', 0),
            doi=doi,
            abstract=paper.get('abstract', ''),
            full_text=full_text,
            sections=sections,
            source_type=source_type,
            pdf_url=pdf_url,
            word_count=len(full_text.split()),
            chunks=chunks
        )

        # Cache result
        self._save_cache(cache_key, ft_paper)

        return ft_paper

    async def batch_get_full_text(self, papers: List[Dict],
                                   max_concurrent: int = 5,
                                   session: aiohttp.ClientSession = None) -> List[FullTextPaper]:
        """Get full text for multiple papers concurrently."""
        _session = session or aiohttp.ClientSession()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _fetch_one(paper):
            async with semaphore:
                try:
                    return await self.get_full_text(paper, _session)
                except Exception:
                    return None

        try:
            results = await asyncio.gather(
                *[_fetch_one(p) for p in papers],
                return_exceptions=True
            )
            return [r for r in results if isinstance(r, FullTextPaper)]
        finally:
            if not session:
                await _session.close()

    # --- Source strategies ---

    async def _try_core(self, doi: str, title: str,
                        session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
        """Try CORE API for full text."""
        try:
            # Search by DOI first, then title
            query = doi if doi else title
            async with session.get(
                "https://api.core.ac.uk/v3/search/works",
                params={'q': query, 'limit': 1},
                headers={'Accept': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status != 200:
                    return None, ""
                data = await resp.json()
                results = data.get('results', [])
                if not results:
                    return None, ""

                work = results[0]
                full_text = work.get('fullText', '')
                download_url = work.get('downloadUrl', '')

                if full_text and len(full_text) > 500:
                    return full_text, download_url

        except Exception:
            pass
        return None, ""

    async def _try_unpaywall(self, doi: str,
                              session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
        """Try Unpaywall for OA PDF link, then download + extract."""
        try:
            async with session.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={'email': 'research-agent@example.com'},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return None, ""
                data = await resp.json()
                best = data.get('best_oa_location', {})
                if not best:
                    return None, ""

                pdf_url = best.get('url_for_pdf') or best.get('url', '')
                if not pdf_url:
                    return None, ""

                # Download and extract PDF
                text = await self._download_and_extract_pdf(pdf_url, session)
                if text and len(text) > 500:
                    return text, pdf_url

        except Exception:
            pass
        return None, ""

    async def _try_arxiv(self, arxiv_id: str,
                          session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
        """Download arXiv PDF and extract text."""
        try:
            # Clean arxiv ID
            clean_id = arxiv_id.replace('arxiv:', '').strip()
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

            text = await self._download_and_extract_pdf(pdf_url, session)
            if text and len(text) > 500:
                return text, pdf_url

        except Exception:
            pass
        return None, ""

    async def _try_pmc(self, pubmed_id: str,
                        session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
        """Try PubMed Central for free XML full text."""
        try:
            # First convert PMID to PMCID
            async with session.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={'ids': pubmed_id, 'format': 'json'},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return None, ""
                data = await resp.json()
                records = data.get('records', [])
                if not records:
                    return None, ""
                pmc_id = records[0].get('pmcid', '')
                if not pmc_id:
                    return None, ""

            # Fetch full text from PMC
            async with session.get(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={'db': 'pmc', 'id': pmc_id, 'rettype': 'xml'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return None, ""
                content = await resp.text()

                # Simple XML text extraction
                text = self._extract_text_from_xml(content)
                if text and len(text) > 500:
                    return text, f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"

        except Exception:
            pass
        return None, ""

    async def _download_and_extract_pdf(self, pdf_url: str,
                                         session: aiohttp.ClientSession) -> Optional[str]:
        """Download a PDF and extract text using PyMuPDF if available, else basic."""
        try:
            async with session.get(
                pdf_url,
                timeout=aiohttp.ClientTimeout(total=60),
                headers={'User-Agent': 'ResearchAgent/2.0'}
            ) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get('Content-Type', '')
                if 'pdf' not in content_type and 'octet' not in content_type:
                    # Might be HTML landing page
                    return None

                pdf_bytes = await resp.read()
                if len(pdf_bytes) < 1000:
                    return None

                return self._extract_text_from_pdf_bytes(pdf_bytes)

        except Exception:
            return None

    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes. Tries PyMuPDF first, then pdfplumber."""
        # Try PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = '\n'.join(text_parts)
            if len(text.strip()) > 200:
                return self._clean_extracted_text(text)
        except ImportError:
            pass
        except Exception:
            pass

        # Try pdfplumber
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            text = '\n'.join(text_parts)
            if len(text.strip()) > 200:
                return self._clean_extracted_text(text)
        except ImportError:
            pass
        except Exception:
            pass

        return None

    def _extract_text_from_xml(self, xml_content: str) -> Optional[str]:
        """Extract text from PMC XML."""
        try:
            # Remove XML tags, keep text
            text = re.sub(r'<[^>]+>', ' ', xml_content)
            text = re.sub(r'\s+', ' ', text).strip()
            return self._clean_extracted_text(text)
        except Exception:
            return None

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text: fix whitespace, remove artifacts."""
        # Fix multiple spaces/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        # Remove page numbers/headers
        text = re.sub(r'\n\d+\n', '\n', text)
        # Remove citation numbers like [1], [2,3]
        # Keep them - useful for analysis
        return text.strip()

    def _extract_sections(self, text: str) -> List[Dict]:
        """Split text into sections based on common heading patterns."""
        sections = []

        # Common academic paper section patterns
        patterns = [
            r'\n(Abstract)\s*\n',
            r'\n(\d+\.?\s*Introduction)\s*\n',
            r'\n(\d+\.?\s*Background)\s*\n',
            r'\n(\d+\.?\s*Related Work)\s*\n',
            r'\n(\d+\.?\s*Literature Review)\s*\n',
            r'\n(\d+\.?\s*Methodology|Methods?)\s*\n',
            r'\n(\d+\.?\s*Data|Dataset)\s*\n',
            r'\n(\d+\.?\s*Results?)\s*\n',
            r'\n(\d+\.?\s*Discussion)\s*\n',
            r'\n(\d+\.?\s*Conclusion)\s*\n',
            r'\n(\d+\.?\s*References?)\s*\n',
        ]

        # Find all section boundaries
        boundaries = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append((match.start(), match.group(1).strip()))

        boundaries.sort(key=lambda x: x[0])

        if not boundaries:
            # No clear sections - treat as single block
            return [{'heading': 'Full Text', 'content': text[:5000]}]

        # Extract section content
        for i, (pos, heading) in enumerate(boundaries):
            end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            content = text[pos:end_pos].strip()
            # Remove the heading from content
            content = re.sub(r'^.*?\n', '', content, count=1).strip()

            if content:
                sections.append({
                    'heading': heading,
                    'content': content[:5000]  # Cap per section
                })

        return sections

    def _chunk_text(self, text: str, chunk_size: int = None,
                    overlap: int = None) -> List[str]:
        """Split text into overlapping chunks for LLM processing."""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    # --- Caching ---

    def _cache_path(self, key: str) -> Path:
        safe_key = re.sub(r'[^\w\-.]', '_', key)[:100]
        return self.cache_dir / f"{safe_key}.txt"

    def _load_cache(self, key: str) -> Optional[FullTextPaper]:
        """Load cached full text."""
        path = self._cache_path(key)
        if path.exists():
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return FullTextPaper(**data)
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, paper: FullTextPaper):
        """Cache full text result."""
        path = self._cache_path(key)
        try:
            import json
            data = {
                'paper_id': paper.paper_id,
                'title': paper.title,
                'authors': paper.authors,
                'year': paper.year,
                'doi': paper.doi,
                'abstract': paper.abstract,
                'full_text': paper.full_text[:50000],  # Cap cache size
                'sections': paper.sections,
                'source_type': paper.source_type,
                'pdf_url': paper.pdf_url,
                'word_count': paper.word_count,
                'chunks': paper.chunks[:50]  # Cap chunks
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass
