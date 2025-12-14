"""
arXiv API client.
Free API, no auth required.
Covers physics, math, CS, biology, and more preprints.
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base_source import BaseSource, Paper
from config.settings import settings


class ArXiv(BaseSource):
    """arXiv API client"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # XML namespaces
    ATOM_NS = '{http://www.w3.org/2005/Atom}'
    ARXIV_NS = '{http://arxiv.org/schemas/atom}'
    
    def __init__(self):
        super().__init__(rate_limit=settings.ARXIV_RATE_LIMIT)
    
    def search(self, query: str, limit: int = 25) -> List[Paper]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of Paper objects
        """
        # Format query for arXiv API
        formatted_query = f"all:{query}"
        
        params = {
            'search_query': formatted_query,
            'start': 0,
            'max_results': min(limit, 100),
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if not response.ok:
                print(f"❌ arXiv API error: {response.status_code}")
                return []
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            for entry in root.findall(f'{self.ATOM_NS}entry'):
                # Extract ID (arXiv ID from URL)
                id_elem = entry.find(f'{self.ATOM_NS}id')
                arxiv_url = id_elem.text if id_elem is not None else ''
                arxiv_id = arxiv_url.split('/abs/')[-1] if '/abs/' in arxiv_url else ''
                
                # Title
                title_elem = entry.find(f'{self.ATOM_NS}title')
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else 'Untitled'
                
                # Authors
                authors = []
                for author in entry.findall(f'{self.ATOM_NS}author'):
                    name_elem = author.find(f'{self.ATOM_NS}name')
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                # Abstract
                summary_elem = entry.find(f'{self.ATOM_NS}summary')
                abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ''
                
                # Published date -> year
                published_elem = entry.find(f'{self.ATOM_NS}published')
                year = 0
                if published_elem is not None:
                    try:
                        year = int(published_elem.text[:4])
                    except:
                        pass
                
                # DOI (if available)
                doi_elem = entry.find(f'{self.ARXIV_NS}doi')
                doi = doi_elem.text if doi_elem is not None else ''
                
                # Categories
                categories = []
                for cat in entry.findall(f'{self.ARXIV_NS}primary_category'):
                    term = cat.get('term', '')
                    if term:
                        categories.append(term)
                
                paper = Paper(
                    paper_id=arxiv_id,
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract[:500],  # Limit abstract length
                    venue='arXiv Preprint',
                    venue_type='preprint',
                    doi=doi,
                    arxiv_id=arxiv_id,
                    url=arxiv_url,
                    citation_count=0,  # arXiv doesn't provide this
                    source='arxiv',
                    verified=True  # arXiv is a verified source
                )
                papers.append(paper)
            
            print(f"✓ arXiv: Found {len(papers)} papers for '{query[:50]}...'")
            return papers
            
        except requests.exceptions.Timeout:
            print(f"❌ arXiv timeout for query: {query[:50]}")
            return []
        except ET.ParseError as e:
            print(f"❌ arXiv XML parse error: {e}")
            return []
        except Exception as e:
            print(f"❌ arXiv error: {e}")
            return []
    
    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Get a specific paper by arXiv ID"""
        params = {
            'id_list': arxiv_id
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if not response.ok:
                return None
            
            root = ET.fromstring(response.content)
            entries = root.findall(f'{self.ATOM_NS}entry')
            
            if not entries:
                return None
            
            # Parse first entry
            entry = entries[0]
            
            title_elem = entry.find(f'{self.ATOM_NS}title')
            title = title_elem.text.strip() if title_elem is not None else 'Untitled'
            
            authors = []
            for author in entry.findall(f'{self.ATOM_NS}author'):
                name_elem = author.find(f'{self.ATOM_NS}name')
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            summary_elem = entry.find(f'{self.ATOM_NS}summary')
            abstract = summary_elem.text.strip() if summary_elem is not None else ''
            
            return Paper(
                paper_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                arxiv_id=arxiv_id,
                source='arxiv'
            )
            
        except Exception as e:
            print(f"❌ Error fetching arXiv paper {arxiv_id}: {e}")
            return None


def test_arxiv():
    """Test the arXiv client"""
    client = ArXiv()
    
    print("\n" + "="*80)
    print("Testing arXiv API")
    print("="*80)
    
    papers = client.search("machine learning optimization", limit=5)
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title[:80]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Year: {paper.year}")
        print(f"   arXiv ID: {paper.arxiv_id}")


if __name__ == "__main__":
    test_arxiv()
