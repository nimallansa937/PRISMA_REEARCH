"""
Deduplicator - Removes duplicate papers across sources.
"""

from typing import List, Set
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base_source import Paper


class Deduplicator:
    """Remove duplicate papers based on DOI, title, or other identifiers"""
    
    def __init__(self):
        self.seen_dois: Set[str] = set()
        self.seen_titles: Set[str] = set()
        self.seen_arxiv: Set[str] = set()
        self.seen_pubmed: Set[str] = set()
    
    def deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove duplicates from a list of papers.
        Prioritizes papers with more metadata (DOI, citations, etc.)
        
        Args:
            papers: List of papers (may contain duplicates)
            
        Returns:
            List of unique papers
        """
        self._reset()
        unique = []
        
        # Sort by quality indicators (prefer papers with more info)
        sorted_papers = sorted(
            papers,
            key=lambda p: (
                bool(p.doi),           # Has DOI
                p.citation_count or 0,  # More citations
                len(p.abstract or ''),  # Longer abstract
                len(p.authors or [])    # More authors
            ),
            reverse=True
        )
        
        for paper in sorted_papers:
            if not self._is_duplicate(paper):
                unique.append(paper)
                self._mark_seen(paper)
        
        print(f"Deduplication: {len(unique)}/{len(papers)} unique papers")
        return unique
    
    def _reset(self):
        """Reset seen sets for new deduplication run"""
        self.seen_dois.clear()
        self.seen_titles.clear()
        self.seen_arxiv.clear()
        self.seen_pubmed.clear()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ''
        # Lowercase, remove punctuation, collapse whitespace
        import re
        normalized = title.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _is_duplicate(self, paper: Paper) -> bool:
        """Check if paper is a duplicate of one already seen"""
        # Check DOI (most reliable)
        if paper.doi and paper.doi.lower() in self.seen_dois:
            return True
        
        # Check arXiv ID
        if paper.arxiv_id and paper.arxiv_id in self.seen_arxiv:
            return True
        
        # Check PubMed ID
        if paper.pubmed_id and paper.pubmed_id in self.seen_pubmed:
            return True
        
        # Check normalized title
        norm_title = self._normalize_title(paper.title)
        if norm_title and norm_title in self.seen_titles:
            return True
        
        return False
    
    def _mark_seen(self, paper: Paper):
        """Mark paper identifiers as seen"""
        if paper.doi:
            self.seen_dois.add(paper.doi.lower())
        
        if paper.arxiv_id:
            self.seen_arxiv.add(paper.arxiv_id)
        
        if paper.pubmed_id:
            self.seen_pubmed.add(paper.pubmed_id)
        
        norm_title = self._normalize_title(paper.title)
        if norm_title:
            self.seen_titles.add(norm_title)


def test_deduplicator():
    """Test the deduplicator"""
    dedup = Deduplicator()
    
    # Create test papers with duplicates
    papers = [
        Paper(paper_id='1', title='Deep Learning for NLP', doi='10.1234/test1', 
              citation_count=100, source='semantic_scholar'),
        Paper(paper_id='2', title='Deep Learning for NLP', doi='10.1234/test1', 
              citation_count=50, source='crossref'),  # Duplicate DOI
        Paper(paper_id='3', title='Deep learning for NLP!', doi='', 
              citation_count=0, source='arxiv'),  # Duplicate title (normalized)
        Paper(paper_id='4', title='Another Paper', doi='10.1234/test2',
              citation_count=75, source='semantic_scholar'),
        Paper(paper_id='5', title='Third Paper', arxiv_id='2401.00001',
              source='arxiv'),
        Paper(paper_id='6', title='Third Paper copy', arxiv_id='2401.00001',
              source='semantic_scholar'),  # Duplicate arXiv ID
    ]
    
    print("\n" + "="*80)
    print("Testing Deduplicator")
    print("="*80)
    
    print(f"\nInput: {len(papers)} papers")
    unique = dedup.deduplicate(papers)
    
    print(f"\nUnique papers:")
    for paper in unique:
        print(f"  - {paper.title} (source: {paper.source})")


if __name__ == "__main__":
    test_deduplicator()
