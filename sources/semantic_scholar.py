"""
Semantic Scholar API client.
Free API with 100 requests/minute (or more with API key).
Covers 200M+ papers across all disciplines.
"""

import requests
from typing import List, Optional
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base_source import BaseSource, Paper
from config.settings import settings


class SemanticScholar(BaseSource):
    """Semantic Scholar API client"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        super().__init__(rate_limit=settings.SEMANTIC_SCHOLAR_RATE_LIMIT)
        self.api_key = settings.SEMANTIC_SCHOLAR_API_KEY
        self.headers = {
            'Accept': 'application/json'
        }
        if self.api_key:
            self.headers['x-api-key'] = self.api_key
    
    def search(self, query: str, limit: int = 25) -> List[Paper]:
        """
        Search Semantic Scholar for papers.
        
        Args:
            query: Search query
            limit: Maximum results (max 100 per request)
            
        Returns:
            List of Paper objects
        """
        fields = 'paperId,title,authors,year,abstract,citationCount,influentialCitationCount,venue,externalIds,url'
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            'query': query,
            'limit': min(limit, 100),
            'fields': fields
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if response.status_code == 429:
                print(f"⚠️  Rate limited by Semantic Scholar. Waiting 60s...")
                time.sleep(60)
                response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if not response.ok:
                print(f"❌ Semantic Scholar API error: {response.status_code}")
                return []
            
            data = response.json()
            papers = []
            
            for item in data.get('data', []):
                paper = Paper(
                    paper_id=item.get('paperId', ''),
                    title=item.get('title', 'Untitled'),
                    authors=[a.get('name', '') for a in item.get('authors', [])],
                    year=item.get('year', 0) or 0,
                    abstract=item.get('abstract', '') or '',
                    venue=item.get('venue', '') or '',
                    venue_type='journal' if item.get('venue') else 'unknown',
                    doi=item.get('externalIds', {}).get('DOI', ''),
                    arxiv_id=item.get('externalIds', {}).get('ArXiv', ''),
                    url=item.get('url', ''),
                    citation_count=item.get('citationCount', 0) or 0,
                    influential_citation_count=item.get('influentialCitationCount', 0) or 0,
                    source='semantic_scholar',
                    verified=bool(item.get('externalIds', {}).get('DOI'))
                )
                papers.append(paper)
            
            print(f"✓ Semantic Scholar: Found {len(papers)} papers for '{query[:50]}...'")
            return papers
            
        except requests.exceptions.Timeout:
            print(f"❌ Semantic Scholar timeout for query: {query[:50]}")
            return []
        except Exception as e:
            print(f"❌ Semantic Scholar error: {e}")
            return []
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by Semantic Scholar ID"""
        fields = 'paperId,title,authors,year,abstract,citationCount,influentialCitationCount,venue,externalIds,url'
        
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {'fields': fields}
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if not response.ok:
                return None
            
            item = response.json()
            
            return Paper(
                paper_id=item.get('paperId', ''),
                title=item.get('title', 'Untitled'),
                authors=[a.get('name', '') for a in item.get('authors', [])],
                year=item.get('year', 0) or 0,
                abstract=item.get('abstract', '') or '',
                venue=item.get('venue', '') or '',
                doi=item.get('externalIds', {}).get('DOI', ''),
                url=item.get('url', ''),
                citation_count=item.get('citationCount', 0) or 0,
                source='semantic_scholar'
            )
            
        except Exception as e:
            print(f"❌ Error fetching paper {paper_id}: {e}")
            return None


def test_semantic_scholar():
    """Test the Semantic Scholar client"""
    client = SemanticScholar()
    
    print("\n" + "="*80)
    print("Testing Semantic Scholar API")
    print("="*80)
    
    papers = client.search("cryptocurrency liquidation cascades", limit=5)
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Year: {paper.year} | Citations: {paper.citation_count}")
        print(f"   DOI: {paper.doi or 'N/A'}")


if __name__ == "__main__":
    test_semantic_scholar()
