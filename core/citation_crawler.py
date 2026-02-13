"""
Citation Network Crawler - Upgrade #7.
Implements snowball sampling for systematic reviews.
Finds papers that keyword searches miss via citation networks.
"""

import asyncio
import aiohttp
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sources.async_sources import AsyncSemanticScholar


@dataclass
class CitationNetwork:
    """Tracks the citation network built during crawling."""
    seed_papers: List[str] = field(default_factory=list)
    referenced_by: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    citing: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    co_citation_pairs: List[Tuple[str, str, int]] = field(default_factory=list)
    total_crawled: int = 0


class CitationCrawler:
    """
    Snowball sampling via citation networks.

    Strategy:
    1. Start with seed papers (top results from keyword search)
    2. Backward snowball: Get references OF seed papers
    3. Forward snowball: Get papers CITING seed papers
    4. Co-citation analysis: Find papers frequently cited together
    """

    def __init__(self, max_papers: int = 500):
        self.s2 = AsyncSemanticScholar()
        self.max_papers = max_papers
        self.seen_ids: Set[str] = set()

    async def snowball_search(
        self,
        seed_papers: List[Dict],
        depth: int = 1,
        top_n_seeds: int = 30,
        refs_per_paper: int = 50,
        cites_per_paper: int = 50,
        progress_callback=None,
        session: aiohttp.ClientSession = None
    ) -> Dict:
        """
        Full snowball sampling from seed papers.

        Args:
            seed_papers: Initial papers from keyword search
            depth: How many levels deep to crawl (1 = direct refs/cites only)
            top_n_seeds: Number of seed papers to use (highest cited)
            refs_per_paper: Max references to fetch per paper
            cites_per_paper: Max citations to fetch per paper
            progress_callback: async callback(phase, current, total, data)

        Returns:
            Dict with new papers and network info
        """
        _session = session or aiohttp.ClientSession()
        network = CitationNetwork()
        all_new_papers = []

        try:
            # Select top seeds by citation count
            seeds = sorted(
                seed_papers,
                key=lambda p: p.get('citation_count', 0) or 0,
                reverse=True
            )[:top_n_seeds]

            # Mark seed IDs as seen
            for p in seed_papers:
                pid = p.get('paper_id', '')
                if pid:
                    self.seen_ids.add(pid)

            seed_ids = [p.get('paper_id', '') for p in seeds if p.get('paper_id')]
            network.seed_papers = seed_ids

            for d in range(depth):
                if progress_callback:
                    await progress_callback('snowball_depth', d + 1, depth, {})

                # Backward snowball: references
                ref_papers = await self._crawl_references(
                    seed_ids, refs_per_paper, _session, progress_callback
                )
                all_new_papers.extend(ref_papers)

                # Forward snowball: citations
                cite_papers = await self._crawl_citations(
                    seed_ids, cites_per_paper, _session, progress_callback
                )
                all_new_papers.extend(cite_papers)

                # Update seeds for next depth
                new_ids = [p.get('paper_id', '') for p in ref_papers + cite_papers
                           if p.get('paper_id')]
                seed_ids = new_ids[:top_n_seeds]

                if len(all_new_papers) >= self.max_papers:
                    break

            # Co-citation analysis
            co_citations = self._analyze_co_citations(network)
            network.co_citation_pairs = co_citations
            network.total_crawled = len(all_new_papers)

        finally:
            if not session:
                await _session.close()

        # Deduplicate
        unique_papers = self._deduplicate(all_new_papers)

        return {
            'new_papers': unique_papers[:self.max_papers],
            'network': {
                'seed_count': len(network.seed_papers),
                'total_crawled': network.total_crawled,
                'unique_new': len(unique_papers),
                'co_citation_pairs': len(co_citations)
            },
            'co_citations': co_citations[:20]
        }

    async def _crawl_references(
        self, paper_ids: List[str], limit: int,
        session: aiohttp.ClientSession,
        progress_callback=None
    ) -> List[Dict]:
        """Backward snowball: Get references of papers."""
        all_refs = []
        total = len(paper_ids)

        # Process in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, total, batch_size):
            batch = paper_ids[i:i + batch_size]
            tasks = [self.s2.get_references(pid, limit, session) for pid in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for pid, result in zip(batch, results):
                if isinstance(result, list):
                    new_refs = [p for p in result
                                if p.get('paper_id') and p['paper_id'] not in self.seen_ids]
                    for p in new_refs:
                        self.seen_ids.add(p['paper_id'])
                        p['discovery_method'] = 'backward_snowball'
                        p['discovered_from'] = pid
                    all_refs.extend(new_refs)

            if progress_callback:
                await progress_callback(
                    'references', min(i + batch_size, total), total,
                    {'new_papers': len(all_refs)}
                )

            await asyncio.sleep(0.5)  # Rate limiting

        return all_refs

    async def _crawl_citations(
        self, paper_ids: List[str], limit: int,
        session: aiohttp.ClientSession,
        progress_callback=None
    ) -> List[Dict]:
        """Forward snowball: Get papers citing these papers."""
        all_cites = []
        total = len(paper_ids)

        batch_size = 5
        for i in range(0, total, batch_size):
            batch = paper_ids[i:i + batch_size]
            tasks = [self.s2.get_citations(pid, limit, session) for pid in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for pid, result in zip(batch, results):
                if isinstance(result, list):
                    new_cites = [p for p in result
                                 if p.get('paper_id') and p['paper_id'] not in self.seen_ids]
                    for p in new_cites:
                        self.seen_ids.add(p['paper_id'])
                        p['discovery_method'] = 'forward_snowball'
                        p['discovered_from'] = pid
                    all_cites.extend(new_cites)

            if progress_callback:
                await progress_callback(
                    'citations', min(i + batch_size, total), total,
                    {'new_papers': len(all_cites)}
                )

            await asyncio.sleep(0.5)

        return all_cites

    def _analyze_co_citations(self, network: CitationNetwork) -> List[Tuple[str, str, int]]:
        """Find papers frequently cited together."""
        # Track which papers are referenced together in the same paper
        ref_sets = defaultdict(set)
        for paper_id, refs in network.referenced_by.items():
            for ref in refs:
                ref_sets[ref].add(paper_id)

        # Find pairs that appear together in many reference lists
        co_cite_counts = defaultdict(int)
        refs_list = list(ref_sets.items())
        for i in range(len(refs_list)):
            for j in range(i + 1, len(refs_list)):
                paper_a, citing_a = refs_list[i]
                paper_b, citing_b = refs_list[j]
                overlap = len(citing_a & citing_b)
                if overlap >= 2:
                    co_cite_counts[(paper_a, paper_b)] = overlap

        # Sort by frequency
        sorted_pairs = sorted(co_cite_counts.items(), key=lambda x: x[1], reverse=True)
        return [(a, b, count) for (a, b), count in sorted_pairs[:50]]

    def _deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Deduplicate by paper_id."""
        seen = set()
        unique = []
        for p in papers:
            pid = p.get('paper_id', '')
            if pid and pid not in seen:
                seen.add(pid)
                unique.append(p)
        return unique
