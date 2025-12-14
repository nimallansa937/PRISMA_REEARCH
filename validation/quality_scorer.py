"""
Quality Scorer - Scores papers based on multiple quality factors.
"""

from typing import List
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.base_source import Paper
from config.settings import settings


class QualityScorer:
    """Score paper quality based on multiple factors"""
    
    # Top venue keywords (partial matches)
    TOP_VENUES = [
        'nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'bmj',
        'ieee', 'acm', 'neurips', 'icml', 'iclr', 'cvpr', 'aaai',
        'pnas', 'plos', 'frontiers', 'proceedings', 'transactions',
        'journal of', 'review of', 'annual review'
    ]
    
    def __init__(self, min_score: float = None):
        self.min_score = min_score or settings.MIN_QUALITY_SCORE
        self.current_year = datetime.now().year
    
    def score(self, paper: Paper) -> float:
        """
        Calculate quality score for a paper (0.0 to 1.0).
        
        Scoring breakdown:
        - DOI presence: 0.20
        - Citations: 0.30 (graduated scale)
        - Venue quality: 0.20
        - Recency: 0.15
        - Abstract: 0.10
        - Authors: 0.05
        """
        score = 0.0
        
        # 1. DOI presence (0.20)
        if paper.doi:
            score += 0.20
        
        # 2. Citation count (0.30)
        citations = paper.citation_count or 0
        if citations >= 100:
            score += 0.30
        elif citations >= 50:
            score += 0.25
        elif citations >= 20:
            score += 0.20
        elif citations >= 10:
            score += 0.15
        elif citations >= 5:
            score += 0.10
        elif citations >= 1:
            score += 0.05
        
        # 3. Venue quality (0.20)
        venue = (paper.venue or '').lower()
        if any(top in venue for top in self.TOP_VENUES):
            score += 0.20
        elif venue and venue != 'arxiv preprint':
            score += 0.10
        elif paper.venue_type == 'preprint':
            score += 0.05  # Preprints get minimal venue score
        
        # 4. Recency (0.15)
        if paper.year:
            age = self.current_year - paper.year
            if age <= 2:
                score += 0.15
            elif age <= 5:
                score += 0.12
            elif age <= 10:
                score += 0.08
            elif age <= 20:
                score += 0.04
        
        # 5. Abstract presence (0.10)
        if paper.abstract and len(paper.abstract) >= 100:
            score += 0.10
        elif paper.abstract and len(paper.abstract) >= 50:
            score += 0.05
        
        # 6. Author presence (0.05)
        if paper.authors and len(paper.authors) >= 1:
            score += 0.05
        
        return min(score, 1.0)
    
    def score_batch(self, papers: List[Paper]) -> List[Paper]:
        """
        Score a batch of papers and update their quality_score field.
        Returns papers sorted by quality score (descending).
        """
        for paper in papers:
            paper.quality_score = self.score(paper)
        
        return sorted(papers, key=lambda p: p.quality_score, reverse=True)
    
    def filter_by_quality(self, papers: List[Paper]) -> List[Paper]:
        """
        Filter papers that meet minimum quality threshold.
        Also scores and sorts the papers.
        """
        scored = self.score_batch(papers)
        filtered = [p for p in scored if p.quality_score >= self.min_score]
        
        print(f"Quality filter: {len(filtered)}/{len(papers)} papers passed (threshold: {self.min_score})")
        
        return filtered


def test_quality_scorer():
    """Test the quality scorer"""
    scorer = QualityScorer(min_score=0.3)
    
    # Create test papers
    test_papers = [
        Paper(
            paper_id='1',
            title='High Quality Paper',
            authors=['John Doe', 'Jane Smith'],
            year=2023,
            venue='Nature',
            doi='10.1038/test',
            citation_count=150,
            abstract='This is a comprehensive abstract that provides detailed information about the research methodology and findings.'
        ),
        Paper(
            paper_id='2',
            title='Medium Quality Paper',
            authors=['Alice'],
            year=2020,
            venue='Some Journal',
            citation_count=25,
            abstract='Short abstract.'
        ),
        Paper(
            paper_id='3',
            title='Low Quality Paper',
            authors=[],
            year=2010,
            venue='',
            citation_count=0,
            abstract=''
        ),
        Paper(
            paper_id='4',
            title='arXiv Preprint',
            authors=['Bob'],
            year=2024,
            venue='arXiv Preprint',
            venue_type='preprint',
            arxiv_id='2401.12345',
            citation_count=0,
            abstract='New research on cutting edge topic with detailed methodology.'
        )
    ]
    
    print("\n" + "="*80)
    print("Testing Quality Scorer")
    print("="*80)
    
    for paper in test_papers:
        score = scorer.score(paper)
        print(f"\n{paper.title}")
        print(f"  Score: {score:.2f}")
        print(f"  Year: {paper.year} | Citations: {paper.citation_count}")
        print(f"  Venue: {paper.venue or 'N/A'}")
        print(f"  DOI: {'✓' if paper.doi else '✗'}")
    
    print("\n" + "-"*40)
    print("Filtering by quality threshold (0.3):")
    filtered = scorer.filter_by_quality(test_papers)
    print(f"Passed: {[p.title for p in filtered]}")


if __name__ == "__main__":
    test_quality_scorer()
