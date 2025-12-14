"""
CrossRef API client.
Free API for DOI verification and metadata lookup.
"""

import requests
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base_source import BaseSource, Paper
from config.settings import settings


class CrossRef(BaseSource):
    """CrossRef API client for DOI verification and metadata"""
    
    BASE_URL = "https://api.crossref.org"
    
    def __init__(self):
        super().__init__(rate_limit=settings.CROSSREF_RATE_LIMIT)
        self.headers = {
            'User-Agent': 'AcademicResearchAgent/1.0 (mailto:research@example.com)'
        }
    
    def search(self, query: str, limit: int = 25) -> List[Paper]:
        """
        Search CrossRef for papers.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of Paper objects
        """
        url = f"{self.BASE_URL}/works"
        params = {
            'query': query,
            'rows': min(limit, 100),
            'sort': 'relevance',
            'order': 'desc'
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if not response.ok:
                print(f"❌ CrossRef API error: {response.status_code}")
                return []
            
            data = response.json()
            items = data.get('message', {}).get('items', [])
            papers = []
            
            for item in items:
                paper = self._parse_item(item)
                if paper:
                    papers.append(paper)
            
            print(f"✓ CrossRef: Found {len(papers)} papers for '{query[:50]}...'")
            return papers
            
        except Exception as e:
            print(f"❌ CrossRef error: {e}")
            return []
    
    def _parse_item(self, item: dict) -> Optional[Paper]:
        """Parse a CrossRef work item into a Paper object"""
        try:
            # DOI
            doi = item.get('DOI', '')
            
            # Title
            titles = item.get('title', [])
            title = titles[0] if titles else 'Untitled'
            
            # Authors
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    name = f"{given} {family}".strip()
                    authors.append(name)
            
            # Year
            year = 0
            published = item.get('published', {})
            date_parts = published.get('date-parts', [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
            
            # Venue
            container = item.get('container-title', [])
            venue = container[0] if container else ''
            
            # Type
            item_type = item.get('type', '')
            venue_type = 'journal' if 'journal' in item_type else item_type
            
            # Citation count
            citation_count = item.get('is-referenced-by-count', 0)
            
            # Abstract
            abstract = item.get('abstract', '')
            if abstract:
                # Remove HTML tags
                import re
                abstract = re.sub('<[^<]+?>', '', abstract)
            
            return Paper(
                paper_id=doi,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract[:500],
                venue=venue,
                venue_type=venue_type,
                doi=doi,
                url=f"https://doi.org/{doi}" if doi else '',
                citation_count=citation_count,
                source='crossref',
                verified=True  # CrossRef is authoritative
            )
            
        except Exception as e:
            print(f"❌ Error parsing CrossRef item: {e}")
            return None
    
    def get_paper(self, doi: str) -> Optional[Paper]:
        """Get a specific paper by DOI"""
        # Clean DOI
        doi = doi.replace('https://doi.org/', '')
        
        url = f"{self.BASE_URL}/works/{doi}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if not response.ok:
                return None
            
            data = response.json()
            item = data.get('message', {})
            
            return self._parse_item(item)
            
        except Exception as e:
            print(f"❌ Error fetching DOI {doi}: {e}")
            return None
    
    def verify_doi(self, doi: str) -> bool:
        """Verify that a DOI exists"""
        doi = doi.replace('https://doi.org/', '')
        url = f"{self.BASE_URL}/works/{doi}"
        
        try:
            response = requests.head(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except:
            return False


def test_crossref():
    """Test the CrossRef client"""
    client = CrossRef()
    
    print("\n" + "="*80)
    print("Testing CrossRef API")
    print("="*80)
    
    # Test search
    papers = client.search("deep learning image recognition", limit=5)
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title[:80]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Year: {paper.year} | Citations: {paper.citation_count}")
        print(f"   DOI: {paper.doi}")
    
    # Test DOI verification
    print("\n" + "-"*40)
    print("Testing DOI verification:")
    test_doi = "10.1038/nature14539"  # Famous deep learning paper
    is_valid = client.verify_doi(test_doi)
    print(f"DOI {test_doi}: {'✓ Valid' if is_valid else '✗ Invalid'}")


if __name__ == "__main__":
    test_crossref()
