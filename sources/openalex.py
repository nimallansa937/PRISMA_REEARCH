"""
OpenAlex - Free, open academic data source.
Indexes 250M+ works including IEEE, ACM, Springer, etc.
https://openalex.org/
"""

import requests
from typing import List, Optional
from .base_source import BaseSource, Paper


class OpenAlex(BaseSource):
    """
    OpenAlex academic search.
    Free API, indexes 250M+ works from all publishers.
    """
    
    def __init__(self):
        super().__init__("openalex")
        self.base_url = "https://api.openalex.org"
        # Polite pool email for higher rate limits
        self.email = "research-agent@example.com"
        
    def search(self, query: str, limit: int = 30) -> List[Paper]:
        """
        Search OpenAlex for papers.
        """
        papers = []
        
        try:
            url = f"{self.base_url}/works"
            params = {
                'search': query,
                'per_page': min(limit, 50),
                'mailto': self.email,
                'select': 'id,doi,title,authorships,abstract_inverted_index,publication_year,cited_by_count,primary_location'
            }
            
            headers = {
                'User-Agent': 'ResearchAgent/1.0 (mailto:research-agent@example.com)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                for item in results:
                    try:
                        # Parse authors
                        authors = []
                        for authorship in item.get('authorships', []):
                            author = authorship.get('author', {})
                            name = author.get('display_name', '')
                            if name:
                                authors.append(name)
                        
                        # Reconstruct abstract from inverted index
                        abstract = self._reconstruct_abstract(
                            item.get('abstract_inverted_index', {})
                        )
                        
                        # Get URL
                        primary_location = item.get('primary_location', {}) or {}
                        source = primary_location.get('source', {}) or {}
                        url = primary_location.get('landing_page_url', '')
                        
                        # Get DOI
                        doi = item.get('doi', '')
                        if doi and doi.startswith('https://doi.org/'):
                            doi = doi.replace('https://doi.org/', '')
                        
                        # Get OpenAlex ID
                        openalex_id = item.get('id', '').split('/')[-1]
                        
                        paper = Paper(
                            paper_id=openalex_id or doi or f"openalex-{hash(item.get('title', ''))}",
                            title=item.get('title', 'Unknown'),
                            authors=authors or ['Unknown'],
                            abstract=abstract,
                            year=item.get('publication_year') or 0,
                            doi=doi,
                            url=url or f"https://openalex.org/works/{openalex_id}",
                            source='openalex',
                            citation_count=item.get('cited_by_count', 0)
                        )
                        papers.append(paper)
                        
                    except Exception as e:
                        continue
            
            print(f"✓ OpenAlex: Found {len(papers)} papers for '{query[:50]}...'")
            
        except requests.exceptions.Timeout:
            print("⚠️ OpenAlex search timed out")
        except Exception as e:
            print(f"⚠️ OpenAlex error: {e}")
        
        return papers[:limit]
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by OpenAlex ID"""
        try:
            url = f"{self.base_url}/works/{paper_id}"
            params = {'mailto': self.email}
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                item = response.json()
                # Parse similar to search
                authors = [a.get('author', {}).get('display_name', '') 
                          for a in item.get('authorships', [])]
                doi = item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else ''
                
                return Paper(
                    paper_id=paper_id,
                    title=item.get('title', 'Unknown'),
                    authors=authors or ['Unknown'],
                    abstract=self._reconstruct_abstract(item.get('abstract_inverted_index', {})),
                    year=item.get('publication_year') or 0,
                    doi=doi,
                    url=f"https://openalex.org/works/{paper_id}",
                    source='openalex',
                    citation_count=item.get('cited_by_count', 0)
                )
        except:
            pass
        return None
    
    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format"""
        if not inverted_index:
            return ''
        
        try:
            # Build word position map
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            
            # Sort by position and join
            word_positions.sort(key=lambda x: x[0])
            abstract = ' '.join([word for pos, word in word_positions])
            return abstract
        except:
            return ''
