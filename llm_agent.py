"""
LLM-Powered Research Agent - Next-gen agent with AI intelligence.

Uses LLMs (DeepSeek/Gemini) for:
1. Domain classification
2. Filter generation
3. Relevance scoring
"""

from typing import List, Dict, Optional
import json
import time
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.llm_client import LLMClient
from core.llm_domain_classifier import LLMDomainClassifier
from core.llm_filter_generator import LLMFilterGenerator
from core.llm_relevance_scorer import LLMRelevanceScorer
from sources.semantic_scholar import SemanticScholar
from sources.arxiv import ArXiv
from sources.pubmed import PubMed
from sources.crossref import CrossRef
from sources.base_source import Paper
from validation.deduplicator import Deduplicator
from config.settings import settings


class LLMPoweredAgent:
    """
    Next-gen research agent powered by LLMs.
    Intelligent, adaptive, domain-agnostic.
    """
    
    def __init__(self, primary_llm: str = "gemini", fallback_llm: str = "deepseek"):
        """
        Initialize LLM-powered agent.
        
        Args:
            primary_llm: Primary LLM ('gemini' or 'deepseek')
            fallback_llm: Fallback if primary fails
        """
        print("🚀 Initializing LLM-Powered Research Agent...")
        
        self.llm = LLMClient(primary=primary_llm, fallback=fallback_llm)
        self.domain_classifier = LLMDomainClassifier(self.llm)
        self.filter_generator = LLMFilterGenerator(self.llm)
        self.relevance_scorer = LLMRelevanceScorer(self.llm)
        self.deduplicator = Deduplicator()
        
        # Data sources
        self.sources = {
            'semantic_scholar': SemanticScholar(),
            'arxiv': ArXiv(),
            'pubmed': PubMed(),
            'crossref': CrossRef()
        }
        
        print("✓ LLM-Powered Agent initialized")
    
    def search(
        self,
        query: str,
        min_papers: int = 30,
        max_papers: int = 60,
        min_relevance: int = 30,
        verbose: bool = True
    ) -> Dict:
        """
        Intelligent research search powered by LLMs.
        
        Args:
            query: Research query
            min_papers: Minimum papers to find
            max_papers: Maximum papers to return
            min_relevance: Minimum LLM relevance score
            verbose: Print progress
        
        Returns:
            Complete research package with scored papers
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🔍 LLM-Powered Research Search")
            print(f"{'='*80}")
            print(f"\nQuery: {query}\n")
        
        # STEP 1: LLM Domain Classification
        if verbose:
            print("🤖 Step 1/4: LLM Domain Classification...")
        
        classification = self.domain_classifier.classify(query)
        
        if verbose:
            print(f"   ✓ Domain: {classification['primary_domain']}")
            print(f"   ✓ Confidence: {classification['confidence']:.0%}")
            print(f"   ✓ Concepts: {', '.join(classification['key_concepts'][:5])}\n")
        
        # STEP 2: LLM Filter Generation
        if verbose:
            print("🤖 Step 2/4: LLM Filter Generation...")
        
        filters = self.filter_generator.generate_filters(query, classification)
        
        if verbose:
            print(f"   ✓ Include: {', '.join(filters['include_keywords'][:4])}...")
            print(f"   ✓ Variations: {len(filters.get('search_variations', []))} query variants")
            print(f"   ✓ Sources: {', '.join(filters['preferred_sources'])}\n")
        
        # STEP 3: Multi-Source Search
        if verbose:
            print("📚 Step 3/4: Searching databases...")
        
        all_papers = []
        
        # Build search queries
        search_queries = [query]
        search_queries.extend(filters.get('search_variations', [])[:3])
        search_queries.extend([' '.join(classification['key_concepts'][:3])])
        
        # Search each preferred source
        for source_name in filters['preferred_sources']:
            if source_name not in self.sources:
                continue
            
            source = self.sources[source_name]
            
            for search_query in search_queries[:2]:  # Limit queries per source
                try:
                    papers = source.search(search_query, limit=25)
                    all_papers.extend([p.to_dict() for p in papers])
                    time.sleep(0.3)  # Rate limiting
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ {source_name} error: {e}")
        
        if verbose:
            print(f"   ✓ Raw papers found: {len(all_papers)}")
        
        # Deduplicate
        unique_papers = self._deduplicate_dicts(all_papers)
        
        if verbose:
            print(f"   ✓ After dedup: {len(unique_papers)}\n")
        
        # STEP 4: LLM Relevance Scoring
        if verbose:
            print("🤖 Step 4/4: LLM Relevance Scoring...")
        
        # Limit papers to score (for API cost)
        papers_to_score = unique_papers[:max_papers]
        
        scored_papers = self.relevance_scorer.score_papers(
            query,
            papers_to_score,
            classification['primary_domain']
        )
        
        # Filter by relevance
        high_quality = [
            p for p in scored_papers
            if p.get('llm_relevance_score', 0) >= min_relevance
        ]
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Complete in {elapsed:.1f}s")
            print(f"   High-quality papers: {len(high_quality)}")
        
        return {
            'query': query,
            'papers': high_quality[:max_papers],
            'statistics': {
                'raw_count': len(all_papers),
                'unique_count': len(unique_papers),
                'scored_count': len(scored_papers),
                'high_quality_count': len(high_quality),
                'avg_relevance': sum(p.get('llm_relevance_score', 0) for p in high_quality) / len(high_quality) if high_quality else 0,
                'search_time_seconds': round(elapsed, 2)
            },
            'classification': classification,
            'filters': filters,
            'timestamp': datetime.now().isoformat()
        }
    
    def _deduplicate_dicts(self, papers: List[Dict]) -> List[Dict]:
        """Deduplicate paper dicts by DOI or title"""
        seen = set()
        unique = []
        
        for paper in papers:
            identifier = paper.get('doi') or paper.get('title', '').lower().strip()[:100]
            
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique.append(paper)
        
        return unique
    
    def format_for_llm(self, result: Dict) -> str:
        """Format results for LLM consumption"""
        papers = result['papers']
        stats = result['statistics']
        
        if not papers:
            return """⚠️ No relevant papers found.
Please generate analysis based on general knowledge, clearly marking claims as "Unverified"."""

        lines = [
            f"## Verified Academic Sources ({len(papers)} papers)",
            f"### Quality: Avg Relevance {stats['avg_relevance']:.0f}/100",
            ""
        ]
        
        for i, paper in enumerate(papers[:30], 1):
            authors = paper.get('authors', [])
            author_str = authors[0] if authors else "Unknown"
            if len(authors) > 1:
                author_str += " et al."
            
            year = paper.get('year', 'n.d.')
            title = paper.get('title', 'Untitled')
            score = paper.get('llm_relevance_score', 0)
            doi = paper.get('doi', '')
            
            line = f"[{i}] {author_str} ({year}). {title}. [Relevance: {score}%]"
            if doi:
                line += f" DOI: {doi}"
            
            lines.append(line)
        
        return "\n".join(lines)


def main():
    """Test the LLM-powered agent"""
    agent = LLMPoweredAgent(primary_llm="gemini", fallback_llm="deepseek")
    
    result = agent.search(
        "Mechanisms of cryptocurrency liquidation cascades",
        min_papers=20,
        max_papers=30
    )
    
    print(f"\n{'='*80}")
    print("📊 FINAL RESULTS")
    print('='*80)
    print(f"\nHigh-Quality Papers: {len(result['papers'])}")
    print(f"Avg Relevance: {result['statistics']['avg_relevance']:.0f}/100")
    
    print("\nTop 5:")
    for i, paper in enumerate(result['papers'][:5], 1):
        print(f"\n{i}. {paper['title'][:70]}...")
        print(f"   Score: {paper['llm_relevance_score']}/100")
        print(f"   {paper.get('llm_reasoning', '')[:60]}...")


if __name__ == "__main__":
    main()
