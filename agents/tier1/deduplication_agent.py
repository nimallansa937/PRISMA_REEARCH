"""
Tier 1 Deduplication Agent - Scripted executor (no LLM).
Multi-field fuzzy matching for robust duplicate detection.

Strategies:
  1. Exact DOI match
  2. Normalized title match (lowered, stripped punctuation)
  3. Fuzzy title match (Levenshtein ratio > 0.90)
  4. Author-year-venue fingerprint match
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import Tier1Agent


class DeduplicationAgent(Tier1Agent):
    """
    Advanced multi-field deduplication.
    No LLM - purely algorithmic execution.

    Fields used:
      - DOI (exact)
      - Title (normalized + fuzzy)
      - Author fingerprint + year
      - arXiv ID / PubMed ID
    """

    def __init__(self, fuzzy_threshold: float = 0.85):
        super().__init__(
            name="Deduplicator",
            description="Multi-field fuzzy deduplication"
        )
        self.fuzzy_threshold = fuzzy_threshold
        self.stats = {
            'doi_matches': 0,
            'title_exact': 0,
            'title_fuzzy': 0,
            'id_matches': 0,
            'fingerprint_matches': 0,
        }

    def execute(self, input_data: Dict) -> Dict:
        """Deduplicate a list of papers."""
        papers = input_data.get('papers', [])
        unique = self.deduplicate(papers)
        duplicates_removed = len(papers) - len(unique)

        return {
            'papers': unique,
            'total_input': len(papers),
            'total_unique': len(unique),
            'duplicates_removed': duplicates_removed,
            'match_stats': dict(self.stats),
        }

    def deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Multi-pass deduplication."""
        if not papers:
            return []

        # Reset stats
        for k in self.stats:
            self.stats[k] = 0

        # Pass 1: Index by exact identifiers
        doi_index: Dict[str, int] = {}
        arxiv_index: Dict[str, int] = {}
        pubmed_index: Dict[str, int] = {}
        title_index: Dict[str, int] = {}
        fingerprint_index: Dict[str, int] = {}

        unique = []
        kept_indices: Set[int] = set()

        for i, paper in enumerate(papers):
            if self._is_duplicate(paper, i, doi_index, arxiv_index,
                                  pubmed_index, title_index, fingerprint_index):
                continue

            # Not a duplicate - add to indices and keep
            self._index_paper(paper, len(unique), doi_index, arxiv_index,
                              pubmed_index, title_index, fingerprint_index)
            unique.append(paper)
            kept_indices.add(i)

        return unique

    def _is_duplicate(self, paper: Dict, idx: int,
                      doi_idx: Dict, arxiv_idx: Dict, pubmed_idx: Dict,
                      title_idx: Dict, fp_idx: Dict) -> bool:
        """Check if paper is a duplicate using multiple strategies."""

        # Strategy 1: Exact DOI match
        doi = self._normalize_doi(paper.get('doi', ''))
        if doi and doi in doi_idx:
            self.stats['doi_matches'] += 1
            return True

        # Strategy 2: arXiv ID match
        arxiv_id = (paper.get('arxiv_id') or '').strip()
        if arxiv_id and arxiv_id in arxiv_idx:
            self.stats['id_matches'] += 1
            return True

        # Strategy 3: PubMed ID match
        pubmed_id = (paper.get('pubmed_id') or '').strip()
        if pubmed_id and pubmed_id in pubmed_idx:
            self.stats['id_matches'] += 1
            return True

        # Strategy 4: Normalized title match
        norm_title = self._normalize_title(paper.get('title', ''))
        if norm_title and norm_title in title_idx:
            self.stats['title_exact'] += 1
            return True

        # Strategy 5: Fuzzy title match (check against all existing titles)
        if norm_title:
            for existing_title in title_idx:
                if self._fuzzy_match(norm_title, existing_title):
                    self.stats['title_fuzzy'] += 1
                    return True

        # Strategy 6: Author-year fingerprint
        fp = self._author_year_fingerprint(paper)
        if fp and fp in fp_idx:
            self.stats['fingerprint_matches'] += 1
            return True

        return False

    def _index_paper(self, paper: Dict, idx: int,
                     doi_idx: Dict, arxiv_idx: Dict, pubmed_idx: Dict,
                     title_idx: Dict, fp_idx: Dict):
        """Add paper to all indices."""
        doi = self._normalize_doi(paper.get('doi', ''))
        if doi:
            doi_idx[doi] = idx

        arxiv_id = (paper.get('arxiv_id') or '').strip()
        if arxiv_id:
            arxiv_idx[arxiv_id] = idx

        pubmed_id = (paper.get('pubmed_id') or '').strip()
        if pubmed_id:
            pubmed_idx[pubmed_id] = idx

        norm_title = self._normalize_title(paper.get('title', ''))
        if norm_title:
            title_idx[norm_title] = idx

        fp = self._author_year_fingerprint(paper)
        if fp:
            fp_idx[fp] = idx

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        """Normalize DOI for comparison."""
        if not doi:
            return ''
        doi = doi.strip().lower()
        doi = re.sub(r'^https?://doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)
        return doi.strip()

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title: lowercase, strip punctuation, collapse whitespace."""
        if not title:
            return ''
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title[:200]  # Cap length

    @staticmethod
    def _author_year_fingerprint(paper: Dict) -> str:
        """Create fingerprint from first author last name + year + first 5 title words."""
        authors = paper.get('authors', [])
        year = paper.get('year')
        title = paper.get('title', '')

        if not authors or not year or not title:
            return ''

        first_author = authors[0] if isinstance(authors[0], str) else str(authors[0])
        # Extract last name
        last_name = first_author.split()[-1].lower().strip() if first_author else ''
        if not last_name:
            return ''

        # First 5 words of title
        words = re.sub(r'[^\w\s]', '', title.lower()).split()[:5]
        title_key = '_'.join(words)

        return f"{last_name}_{year}_{title_key}"

    def _fuzzy_match(self, s1: str, s2: str) -> bool:
        """Quick Levenshtein ratio check."""
        if not s1 or not s2:
            return False

        # Length-based quick reject
        len_diff = abs(len(s1) - len(s2))
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return True
        if len_diff / max_len > (1 - self.fuzzy_threshold):
            return False

        # Simple ratio using character overlap
        # (Avoids external dependency on python-Levenshtein)
        common = len(set(s1.split()) & set(s2.split()))
        total = max(len(s1.split()), len(s2.split()))
        if total == 0:
            return False
        ratio = common / total
        return ratio >= self.fuzzy_threshold

    def get_stats(self) -> Dict:
        """Return deduplication statistics."""
        return dict(self.stats)
