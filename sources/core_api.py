"""
CORE API - The world's largest collection of open access research papers.
140M+ open access papers with full-text content.
https://core.ac.uk/
Free API (rate-limited), API key for higher limits.
"""

import requests
from typing import List, Optional
from .base_source import BaseSource, Paper


class COREApi(BaseSource):
    """
    CORE API - 140M+ open access papers with full-text.
    Free tier: 10 requests/minute. With API key: higher limits.
    """

    def __init__(self, api_key: str = ""):
        super().__init__("core")
        self.base_url = "https://api.core.ac.uk/v3"
        self.api_key = api_key
        self.headers = {
            'Accept': 'application/json'
        }
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'

    def search(self, query: str, limit: int = 30) -> List[Paper]:
        """Search CORE for open access papers."""
        papers = []

        try:
            url = f"{self.base_url}/search/works"
            params = {
                'q': query,
                'limit': min(limit, 100),
                'scroll': 'false'
            }

            response = requests.get(
                url, params=params, headers=self.headers, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                for item in results:
                    try:
                        paper = self._parse(item)
                        if paper:
                            papers.append(paper)
                    except Exception:
                        continue

            print(f"  CORE API: Found {len(papers)} papers for '{query[:50]}...'")

        except requests.exceptions.Timeout:
            print("  CORE API search timed out")
        except Exception as e:
            print(f"  CORE API error: {e}")

        return papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by CORE ID."""
        try:
            url = f"{self.base_url}/works/{paper_id}"
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                item = response.json()
                return self._parse(item)
        except Exception:
            pass
        return None

    def get_full_text(self, paper_id: str) -> Optional[str]:
        """Get full text content of a paper."""
        try:
            url = f"{self.base_url}/works/{paper_id}"
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data.get('fullText', '')
        except Exception:
            pass
        return None

    def _parse(self, item: dict) -> Optional[Paper]:
        """Parse CORE API response into Paper object."""
        try:
            title = item.get('title', '')
            if not title or title == 'Unknown':
                return None

            # Authors
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

            # Year
            year = 0
            year_published = item.get('yearPublished')
            if year_published:
                try:
                    year = int(year_published)
                except (ValueError, TypeError):
                    pass

            # DOI
            doi = item.get('doi', '') or ''
            if doi.startswith('https://doi.org/'):
                doi = doi[16:]

            # URLs
            download_url = item.get('downloadUrl', '')
            source_url = item.get('sourceFulltextUrls', [''])[0] if item.get('sourceFulltextUrls') else ''
            url = download_url or source_url or ''

            # Abstract
            abstract = item.get('abstract', '') or ''

            # Full text available?
            has_full_text = bool(item.get('fullText'))

            core_id = str(item.get('id', ''))

            return Paper(
                paper_id=core_id or doi or f"core-{hash(title)}",
                title=title,
                authors=authors or ['Unknown'],
                abstract=abstract,
                year=year,
                doi=doi,
                url=url,
                source='core',
                citation_count=item.get('citationCount', 0) or 0,
                verified=bool(doi)
            )
        except Exception:
            return None
