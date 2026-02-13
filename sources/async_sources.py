"""
Async Source Wrappers - Upgrade #1 & #2: Async + Pagination for all 6 sources.
Wraps existing sync sources with async support and adds pagination.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import re
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base_source import Paper
from config.settings import settings


class AsyncSemanticScholar:
    """Async Semantic Scholar with pagination. Supports up to 10,000 papers."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = 'paperId,title,authors,year,abstract,citationCount,influentialCitationCount,venue,externalIds,url,tldr,references,citations'

    def __init__(self):
        self.api_key = settings.SEMANTIC_SCHOLAR_API_KEY
        self.headers = {'Accept': 'application/json'}
        if self.api_key:
            self.headers['x-api-key'] = self.api_key

    async def search(self, query: str, limit: int = 100, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            params = {
                'query': query,
                'limit': min(limit, 100),
                'offset': offset,
                'fields': self.FIELDS
            }
            async with _session.get(
                f"{self.BASE_URL}/paper/search",
                params=params, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(30)
                    return await self.search(query, limit, offset, _session)
                if resp.status != 200:
                    return []
                data = await resp.json()
                return [self._parse(item) for item in data.get('data', [])]
        except Exception as e:
            print(f"  [AsyncS2] Error: {e}")
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 500,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        """Fetch up to target papers via pagination."""
        all_papers = []
        offset = 0
        page_size = 100  # S2 max per page

        _session = session or aiohttp.ClientSession()
        try:
            while len(all_papers) < target:
                papers = await self.search(query, page_size, offset, _session)
                if not papers:
                    break
                all_papers.extend(papers)
                offset += len(papers)
                if len(papers) < page_size:
                    break
                await asyncio.sleep(0.3)
        finally:
            if not session:
                await _session.close()
        return all_papers[:target]

    async def get_references(self, paper_id: str, limit: int = 100,
                              session: aiohttp.ClientSession = None) -> List[Dict]:
        """Get papers referenced BY this paper (snowball backward)."""
        _session = session or aiohttp.ClientSession()
        try:
            fields = 'paperId,title,authors,year,abstract,citationCount,venue,externalIds,url'
            async with _session.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={'fields': fields, 'limit': min(limit, 1000)},
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                papers = []
                for item in data.get('data', []):
                    cited = item.get('citedPaper', {})
                    if cited and cited.get('paperId'):
                        papers.append(self._parse(cited))
                return papers
        except Exception as e:
            return []
        finally:
            if not session:
                await _session.close()

    async def get_citations(self, paper_id: str, limit: int = 100,
                             session: aiohttp.ClientSession = None) -> List[Dict]:
        """Get papers that CITE this paper (snowball forward)."""
        _session = session or aiohttp.ClientSession()
        try:
            fields = 'paperId,title,authors,year,abstract,citationCount,venue,externalIds,url'
            async with _session.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={'fields': fields, 'limit': min(limit, 1000)},
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                papers = []
                for item in data.get('data', []):
                    citing = item.get('citingPaper', {})
                    if citing and citing.get('paperId'):
                        papers.append(self._parse(citing))
                return papers
        except Exception as e:
            return []
        finally:
            if not session:
                await _session.close()

    def _parse(self, item: dict) -> Dict:
        ext_ids = item.get('externalIds', {}) or {}
        tldr = item.get('tldr', {}) or {}
        return {
            'paper_id': item.get('paperId', ''),
            'title': item.get('title', 'Untitled'),
            'authors': [a.get('name', '') for a in (item.get('authors') or [])],
            'year': item.get('year', 0) or 0,
            'abstract': item.get('abstract', '') or '',
            'venue': item.get('venue', '') or '',
            'venue_type': 'journal' if item.get('venue') else 'unknown',
            'doi': ext_ids.get('DOI', ''),
            'arxiv_id': ext_ids.get('ArXiv', ''),
            'pubmed_id': ext_ids.get('PubMed', ''),
            'url': item.get('url', ''),
            'citation_count': item.get('citationCount', 0) or 0,
            'influential_citation_count': item.get('influentialCitationCount', 0) or 0,
            'tldr': tldr.get('text', ''),
            'source': 'semantic_scholar',
            'verified': bool(ext_ids.get('DOI'))
        }


class AsyncOpenAlex:
    """Async OpenAlex with cursor pagination. Supports unlimited papers."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self):
        self.email = "research-agent@example.com"
        self.headers = {
            'User-Agent': 'ResearchAgent/2.0 (mailto:research-agent@example.com)'
        }

    async def search(self, query: str, limit: int = 50, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            params = {
                'search': query,
                'per_page': min(limit, 200),
                'page': (offset // max(limit, 1)) + 1,
                'mailto': self.email,
                'select': 'id,doi,title,authorships,abstract_inverted_index,publication_year,cited_by_count,primary_location,type'
            }
            async with _session.get(
                f"{self.BASE_URL}/works",
                params=params, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return [self._parse(item) for item in data.get('results', [])]
        except Exception as e:
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 500,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        """Use cursor pagination for large result sets."""
        all_papers = []
        _session = session or aiohttp.ClientSession()
        cursor = '*'

        try:
            while len(all_papers) < target and cursor:
                params = {
                    'search': query,
                    'per_page': min(200, target - len(all_papers)),
                    'mailto': self.email,
                    'cursor': cursor,
                    'select': 'id,doi,title,authorships,abstract_inverted_index,publication_year,cited_by_count,primary_location,type'
                }
                async with _session.get(
                    f"{self.BASE_URL}/works",
                    params=params, headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    results = data.get('results', [])
                    if not results:
                        break
                    all_papers.extend([self._parse(item) for item in results])
                    cursor = data.get('meta', {}).get('next_cursor')
                    await asyncio.sleep(0.1)
        finally:
            if not session:
                await _session.close()

        return all_papers[:target]

    def _parse(self, item: dict) -> Dict:
        authors = []
        for authorship in item.get('authorships', []):
            name = authorship.get('author', {}).get('display_name', '')
            if name:
                authors.append(name)

        abstract = self._reconstruct_abstract(item.get('abstract_inverted_index', {}))
        doi = item.get('doi', '') or ''
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]

        openalex_id = (item.get('id', '') or '').split('/')[-1]
        location = item.get('primary_location', {}) or {}
        url = location.get('landing_page_url', '')

        return {
            'paper_id': openalex_id or doi,
            'title': item.get('title', 'Unknown') or 'Unknown',
            'authors': authors or ['Unknown'],
            'year': item.get('publication_year') or 0,
            'abstract': abstract,
            'doi': doi,
            'url': url or f"https://openalex.org/works/{openalex_id}",
            'citation_count': item.get('cited_by_count', 0) or 0,
            'source': 'openalex',
            'venue': '',
            'venue_type': item.get('type', ''),
            'verified': bool(doi)
        }

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        if not inverted_index:
            return ''
        try:
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort(key=lambda x: x[0])
            return ' '.join(w for _, w in word_positions)
        except Exception:
            return ''


class AsyncArXiv:
    """Async arXiv with pagination. Supports up to 30,000 results."""

    BASE_URL = "http://export.arxiv.org/api/query"
    ATOM_NS = '{http://www.w3.org/2005/Atom}'
    ARXIV_NS = '{http://arxiv.org/schemas/atom}'

    async def search(self, query: str, limit: int = 100, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            params = {
                'search_query': f'all:{query}',
                'start': offset,
                'max_results': min(limit, 200),
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            async with _session.get(
                self.BASE_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                content = await resp.read()
                return self._parse_xml(content)
        except Exception as e:
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 500,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        all_papers = []
        offset = 0
        page_size = 200

        _session = session or aiohttp.ClientSession()
        try:
            while len(all_papers) < target:
                papers = await self.search(query, page_size, offset, _session)
                if not papers:
                    break
                all_papers.extend(papers)
                offset += len(papers)
                if len(papers) < page_size:
                    break
                await asyncio.sleep(3)  # arXiv requires 3s between requests
        finally:
            if not session:
                await _session.close()
        return all_papers[:target]

    def _parse_xml(self, content: bytes) -> List[Dict]:
        papers = []
        try:
            root = ET.fromstring(content)
            for entry in root.findall(f'{self.ATOM_NS}entry'):
                id_elem = entry.find(f'{self.ATOM_NS}id')
                arxiv_url = id_elem.text if id_elem is not None else ''
                arxiv_id = arxiv_url.split('/abs/')[-1] if '/abs/' in arxiv_url else ''

                title_elem = entry.find(f'{self.ATOM_NS}title')
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else 'Untitled'

                authors = []
                for author in entry.findall(f'{self.ATOM_NS}author'):
                    name_elem = author.find(f'{self.ATOM_NS}name')
                    if name_elem is not None:
                        authors.append(name_elem.text)

                summary_elem = entry.find(f'{self.ATOM_NS}summary')
                abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ''

                published_elem = entry.find(f'{self.ATOM_NS}published')
                year = 0
                if published_elem is not None:
                    try:
                        year = int(published_elem.text[:4])
                    except (ValueError, TypeError):
                        pass

                doi_elem = entry.find(f'{self.ARXIV_NS}doi')
                doi = doi_elem.text if doi_elem is not None else ''

                papers.append({
                    'paper_id': arxiv_id,
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'abstract': abstract,
                    'venue': 'arXiv Preprint',
                    'venue_type': 'preprint',
                    'doi': doi,
                    'arxiv_id': arxiv_id,
                    'url': arxiv_url,
                    'citation_count': 0,
                    'source': 'arxiv',
                    'verified': True
                })
        except ET.ParseError:
            pass
        return papers


class AsyncPubMed:
    """Async PubMed with pagination. Supports up to 10,000 results."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.api_key = settings.NCBI_API_KEY

    async def search(self, query: str, limit: int = 100, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            # Step 1: Search for IDs
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': min(limit, 200),
                'retstart': offset,
                'sort': 'relevance',
                'retmode': 'json'
            }
            if self.api_key:
                params['api_key'] = self.api_key

            async with _session.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                if not id_list:
                    return []

            # Step 2: Fetch details
            return await self._fetch_details(id_list, _session)
        except Exception as e:
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 500,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        all_papers = []
        offset = 0
        page_size = 200

        _session = session or aiohttp.ClientSession()
        try:
            while len(all_papers) < target:
                papers = await self.search(query, page_size, offset, _session)
                if not papers:
                    break
                all_papers.extend(papers)
                offset += len(papers)
                if len(papers) < page_size:
                    break
                await asyncio.sleep(0.5)
        finally:
            if not session:
                await _session.close()
        return all_papers[:target]

    async def _fetch_details(self, pmids: List[str],
                              session: aiohttp.ClientSession) -> List[Dict]:
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        if self.api_key:
            params['api_key'] = self.api_key

        async with session.get(
            f"{self.BASE_URL}/efetch.fcgi",
            params=params,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status != 200:
                return []
            content = await resp.read()

        papers = []
        try:
            root = ET.fromstring(content)
            for article in root.findall('.//PubmedArticle'):
                p = self._parse_article(article)
                if p:
                    papers.append(p)
        except ET.ParseError:
            pass
        return papers

    def _parse_article(self, article) -> Optional[Dict]:
        try:
            medline = article.find('.//MedlineCitation')
            if medline is None:
                return None

            pmid_elem = medline.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ''

            art = medline.find('.//Article')
            if art is None:
                return None

            title_elem = art.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else 'Untitled'

            abstract_parts = []
            for abs_text in art.findall('.//AbstractText'):
                if abs_text.text:
                    abstract_parts.append(abs_text.text)
            abstract = ' '.join(abstract_parts)

            authors = []
            for author in art.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    name = last_name.text
                    if fore_name is not None:
                        name = f"{fore_name.text} {name}"
                    authors.append(name)

            year = 0
            pub_date = art.find('.//PubDate')
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                if year_elem is not None:
                    try:
                        year = int(year_elem.text)
                    except (ValueError, TypeError):
                        pass

            journal_elem = art.find('.//Journal/Title')
            venue = journal_elem.text if journal_elem is not None else ''

            doi = ''
            for id_elem in article.findall('.//ArticleIdList/ArticleId'):
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
                    break

            return {
                'paper_id': pmid,
                'title': title,
                'authors': authors,
                'year': year,
                'abstract': abstract,
                'venue': venue,
                'venue_type': 'journal',
                'doi': doi,
                'pubmed_id': pmid,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'citation_count': 0,
                'source': 'pubmed',
                'verified': True
            }
        except Exception:
            return None


class AsyncCrossRef:
    """Async CrossRef with pagination."""

    BASE_URL = "https://api.crossref.org"

    def __init__(self):
        self.headers = {
            'User-Agent': 'AcademicResearchAgent/2.0 (mailto:research@example.com)'
        }

    async def search(self, query: str, limit: int = 100, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            params = {
                'query': query,
                'rows': min(limit, 1000),
                'offset': offset,
                'sort': 'relevance',
                'order': 'desc'
            }
            async with _session.get(
                f"{self.BASE_URL}/works",
                params=params, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                items = data.get('message', {}).get('items', [])
                return [self._parse(item) for item in items if self._parse(item)]
        except Exception as e:
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 500,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        all_papers = []
        offset = 0
        page_size = 200

        _session = session or aiohttp.ClientSession()
        try:
            while len(all_papers) < target:
                papers = await self.search(query, page_size, offset, _session)
                if not papers:
                    break
                all_papers.extend(papers)
                offset += len(papers)
                if len(papers) < page_size:
                    break
                await asyncio.sleep(0.3)
        finally:
            if not session:
                await _session.close()
        return all_papers[:target]

    def _parse(self, item: dict) -> Optional[Dict]:
        try:
            doi = item.get('DOI', '')
            titles = item.get('title', [])
            title = titles[0] if titles else 'Untitled'

            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    authors.append(f"{given} {family}".strip())

            year = 0
            published = item.get('published', {})
            date_parts = published.get('date-parts', [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]

            container = item.get('container-title', [])
            venue = container[0] if container else ''

            abstract = item.get('abstract', '')
            if abstract:
                abstract = re.sub('<[^<]+?>', '', abstract)

            return {
                'paper_id': doi,
                'title': title,
                'authors': authors,
                'year': year,
                'abstract': abstract,
                'venue': venue,
                'venue_type': 'journal' if 'journal' in item.get('type', '') else item.get('type', ''),
                'doi': doi,
                'url': f"https://doi.org/{doi}" if doi else '',
                'citation_count': item.get('is-referenced-by-count', 0),
                'source': 'crossref',
                'verified': True
            }
        except Exception:
            return None


class AsyncSSRN:
    """Async SSRN via CrossRef proxy."""

    def __init__(self):
        self.crossref = AsyncCrossRef()

    async def search(self, query: str, limit: int = 50, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            params = {
                'query': query,
                'filter': 'prefix:10.2139',
                'rows': min(limit, 100),
                'offset': offset,
                'select': 'DOI,title,author,abstract,published-print,container-title'
            }
            headers = {'User-Agent': 'ResearchAgent/2.0 (Academic Research Tool)'}

            async with _session.get(
                "https://api.crossref.org/works",
                params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                items = data.get('message', {}).get('items', [])
                papers = []
                for item in items:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                return papers
        except Exception:
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 200,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        all_papers = []
        offset = 0
        _session = session or aiohttp.ClientSession()
        try:
            while len(all_papers) < target:
                papers = await self.search(query, 100, offset, _session)
                if not papers:
                    break
                all_papers.extend(papers)
                offset += len(papers)
                if len(papers) < 100:
                    break
                await asyncio.sleep(0.5)
        finally:
            if not session:
                await _session.close()
        return all_papers[:target]

    def _parse(self, item: dict) -> Optional[Dict]:
        try:
            authors = []
            for author in item.get('author', []):
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    authors.append(name)

            published = item.get('published-print', {}) or item.get('published-online', {})
            date_parts = published.get('date-parts', [[None]])[0]
            year = date_parts[0] if date_parts else 0

            title = item.get('title', [''])[0] if item.get('title') else 'Unknown'
            doi = item.get('DOI', '')

            return {
                'paper_id': doi,
                'title': title,
                'authors': authors or ['Unknown'],
                'year': year or 0,
                'abstract': item.get('abstract', ''),
                'doi': doi,
                'url': f"https://ssrn.com/abstract={doi.split('.')[-1]}" if doi else '',
                'citation_count': 0,
                'source': 'ssrn',
                'venue': 'SSRN',
                'venue_type': 'working_paper',
                'verified': bool(doi)
            }
        except Exception:
            return None


class AsyncUnpaywall:
    """Async Unpaywall for finding open access full text. Upgrade #8."""

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: str = "research-agent@example.com"):
        self.email = email

    async def get_open_access_url(self, doi: str,
                                   session: aiohttp.ClientSession = None) -> Optional[str]:
        """Get open access PDF URL for a DOI."""
        if not doi:
            return None
        _session = session or aiohttp.ClientSession()
        try:
            async with _session.get(
                f"{self.BASE_URL}/{doi}",
                params={'email': self.email},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                best = data.get('best_oa_location', {})
                if best:
                    return best.get('url_for_pdf') or best.get('url')
                return None
        except Exception:
            return None
        finally:
            if not session:
                await _session.close()

    async def batch_check(self, dois: List[str],
                           session: aiohttp.ClientSession = None) -> Dict[str, Optional[str]]:
        """Check multiple DOIs for open access."""
        _session = session or aiohttp.ClientSession()
        results = {}
        try:
            tasks = [self.get_open_access_url(doi, _session) for doi in dois]
            urls = await asyncio.gather(*tasks, return_exceptions=True)
            for doi, url in zip(dois, urls):
                results[doi] = url if isinstance(url, str) else None
        finally:
            if not session:
                await _session.close()
        return results


class AsyncCORE:
    """Async CORE API for 140M+ open access papers with full text."""

    BASE_URL = "https://api.core.ac.uk/v3"

    def __init__(self):
        self.api_key = settings.CORE_API_KEY if hasattr(settings, 'CORE_API_KEY') else ""
        self.headers = {'Accept': 'application/json'}
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'

    async def search(self, query: str, limit: int = 100, offset: int = 0,
                     session: aiohttp.ClientSession = None) -> List[Dict]:
        _session = session or aiohttp.ClientSession()
        try:
            params = {
                'q': query,
                'limit': min(limit, 100),
                'offset': offset,
                'scroll': 'false'
            }
            async with _session.get(
                f"{self.BASE_URL}/search/works",
                params=params, headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(60)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json()
                results = data.get('results', [])
                papers = []
                for item in results:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                return papers
        except Exception:
            return []
        finally:
            if not session:
                await _session.close()

    async def search_paginated(self, query: str, target: int = 200,
                                session: aiohttp.ClientSession = None) -> List[Dict]:
        all_papers = []
        offset = 0
        _session = session or aiohttp.ClientSession()
        try:
            while len(all_papers) < target:
                papers = await self.search(query, 100, offset, _session)
                if not papers:
                    break
                all_papers.extend(papers)
                offset += len(papers)
                if len(papers) < 100:
                    break
                await asyncio.sleep(6)  # Respect CORE free tier rate limit
        finally:
            if not session:
                await _session.close()
        return all_papers[:target]

    async def get_full_text(self, paper_id: str,
                             session: aiohttp.ClientSession = None) -> Optional[str]:
        """Get full text content for a CORE paper."""
        _session = session or aiohttp.ClientSession()
        try:
            async with _session.get(
                f"{self.BASE_URL}/works/{paper_id}",
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get('fullText', '')
        except Exception:
            return None
        finally:
            if not session:
                await _session.close()

    def _parse(self, item: dict) -> Optional[Dict]:
        try:
            title = item.get('title', '')
            if not title or title == 'Unknown':
                return None

            authors = []
            for author in item.get('authors', []):
                if isinstance(author, dict):
                    name = author.get('name', '')
                elif isinstance(author, str):
                    name = author
                else:
                    continue
                if name:
                    authors.append(name)

            year = 0
            year_published = item.get('yearPublished')
            if year_published:
                try:
                    year = int(year_published)
                except (ValueError, TypeError):
                    pass

            doi = item.get('doi', '') or ''
            if doi.startswith('https://doi.org/'):
                doi = doi[16:]

            download_url = item.get('downloadUrl', '')
            source_urls = item.get('sourceFulltextUrls', [])
            source_url = source_urls[0] if source_urls else ''
            url = download_url or source_url or ''

            core_id = str(item.get('id', ''))

            return {
                'paper_id': core_id or doi or f"core-{hash(title)}",
                'title': title,
                'authors': authors or ['Unknown'],
                'year': year or 0,
                'abstract': item.get('abstract', '') or '',
                'doi': doi,
                'url': url,
                'citation_count': item.get('citationCount', 0) or 0,
                'source': 'core',
                'venue': '',
                'has_full_text': bool(item.get('fullText')),
                'verified': bool(doi)
            }
        except Exception:
            return None
