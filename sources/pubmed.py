"""
PubMed/NCBI API client.
Free API for biomedical and life sciences literature.
Requires API key for higher rate limits.
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


class PubMed(BaseSource):
    """PubMed/NCBI API client"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self):
        super().__init__(rate_limit=settings.PUBMED_RATE_LIMIT)
        self.api_key = settings.NCBI_API_KEY
    
    def search(self, query: str, limit: int = 25) -> List[Paper]:
        """
        Search PubMed for papers.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of Paper objects
        """
        # Step 1: Search for IDs
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(limit, 100),
            'sort': 'relevance',
            'retmode': 'json'
        }
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            
            if not response.ok:
                print(f"❌ PubMed search error: {response.status_code}")
                return []
            
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                print(f"⚠️  PubMed: No results for '{query[:50]}...'")
                return []
            
            # Step 2: Fetch details for IDs
            papers = self._fetch_details(id_list)
            print(f"✓ PubMed: Found {len(papers)} papers for '{query[:50]}...'")
            return papers
            
        except Exception as e:
            print(f"❌ PubMed error: {e}")
            return []
    
    def _fetch_details(self, pmids: List[str]) -> List[Paper]:
        """Fetch detailed info for a list of PubMed IDs"""
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(fetch_url, params=params, timeout=60)
            
            if not response.ok:
                return []
            
            root = ET.fromstring(response.content)
            papers = []
            
            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"❌ PubMed fetch error: {e}")
            return []
    
    def _parse_article(self, article) -> Optional[Paper]:
        """Parse a PubmedArticle XML element into a Paper object"""
        try:
            medline = article.find('.//MedlineCitation')
            if medline is None:
                return None
            
            # PMID
            pmid_elem = medline.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ''
            
            # Article info
            art = medline.find('.//Article')
            if art is None:
                return None
            
            # Title
            title_elem = art.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else 'Untitled'
            
            # Abstract
            abstract_parts = []
            for abs_text in art.findall('.//AbstractText'):
                if abs_text.text:
                    abstract_parts.append(abs_text.text)
            abstract = ' '.join(abstract_parts)
            
            # Authors
            authors = []
            for author in art.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    name = last_name.text
                    if fore_name is not None:
                        name = f"{fore_name.text} {name}"
                    authors.append(name)
            
            # Year
            year = 0
            pub_date = art.find('.//PubDate')
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                if year_elem is not None:
                    try:
                        year = int(year_elem.text)
                    except:
                        pass
            
            # Journal
            journal_elem = art.find('.//Journal/Title')
            venue = journal_elem.text if journal_elem is not None else ''
            
            # DOI
            doi = ''
            for id_elem in article.findall('.//ArticleIdList/ArticleId'):
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
                    break
            
            return Paper(
                paper_id=pmid,
                title=title,
                authors=authors,
                year=year,
                abstract=abstract[:500],
                venue=venue,
                venue_type='journal',
                doi=doi,
                pubmed_id=pmid,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source='pubmed',
                verified=True
            )
            
        except Exception as e:
            print(f"❌ Error parsing PubMed article: {e}")
            return None
    
    def get_paper(self, pmid: str) -> Optional[Paper]:
        """Get a specific paper by PubMed ID"""
        papers = self._fetch_details([pmid])
        return papers[0] if papers else None


def test_pubmed():
    """Test the PubMed client"""
    client = PubMed()
    
    print("\n" + "="*80)
    print("Testing PubMed API")
    print("="*80)
    
    papers = client.search("CRISPR gene editing cancer", limit=5)
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title[:80]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Year: {paper.year} | Journal: {paper.venue[:40]}")
        print(f"   PMID: {paper.pubmed_id}")


if __name__ == "__main__":
    test_pubmed()
