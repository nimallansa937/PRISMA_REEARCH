"""
SSRN (Social Science Research Network) data source.
Uses SSRN's public search API for working papers.
"""

import requests
from typing import List, Optional
from .base_source import BaseSource, Paper


class SSRN(BaseSource):
    """
    SSRN working paper search.
    Focus: Finance, Economics, Law, and Social Sciences.
    """
    
    def __init__(self):
        super().__init__("ssrn")
        self.base_url = "https://api.ssrn.com/content/v1"
        self.search_url = "https://www.ssrn.com/index.cfm/en/search/results/"
        
    def search(self, query: str, limit: int = 30) -> List[Paper]:
        """
        Search SSRN for papers.
        
        Note: SSRN doesn't have a public API, so we use CrossRef as a proxy
        for SSRN-hosted content (DOIs with ssrn prefix).
        """
        papers = []
        
        try:
            # Use CrossRef to find SSRN papers by DOI prefix
            url = "https://api.crossref.org/works"
            params = {
                'query': query,
                'filter': 'prefix:10.2139',  # SSRN DOI prefix
                'rows': limit,
                'select': 'DOI,title,author,abstract,published-print,container-title'
            }
            
            headers = {
                'User-Agent': 'ResearchAgent/1.0 (Academic Research Tool)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                for item in items:
                    try:
                        # Parse authors
                        authors = []
                        for author in item.get('author', []):
                            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                            if name:
                                authors.append(name)
                        
                        # Get year
                        published = item.get('published-print', {}) or item.get('published-online', {})
                        date_parts = published.get('date-parts', [[None]])[0]
                        year = date_parts[0] if date_parts else None
                        
                        # Get title
                        title = item.get('title', [''])[0] if item.get('title') else 'Unknown'
                        
                        # Get DOI and URL
                        doi = item.get('DOI', '')
                        url = f"https://ssrn.com/abstract={doi.split('.')[-1]}" if doi else ''
                        
                        paper = Paper(
                            title=title,
                            authors=authors or ['Unknown'],
                            abstract=item.get('abstract', ''),
                            year=year,
                            doi=doi,
                            url=url,
                            source='ssrn',
                            citation_count=0  # CrossRef doesn't provide this
                        )
                        papers.append(paper)
                        
                    except Exception as e:
                        continue
            
            print(f"✓ SSRN: Found {len(papers)} papers for '{query[:50]}...'")
            
        except requests.exceptions.Timeout:
            print("⚠️ SSRN search timed out")
        except Exception as e:
            print(f"⚠️ SSRN error: {e}")
        
        return papers[:limit]
