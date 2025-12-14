"""
Academic Research Agent - Main Orchestrator

This is the main entry point that coordinates:
1. Query analysis
2. Search strategy generation
3. Multi-source paper retrieval
4. Quality filtering and deduplication
5. Result formatting
"""

from typing import List, Dict, Optional
import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from core.query_analyzer import QueryAnalyzer
from core.search_strategy import SearchStrategy
from sources.semantic_scholar import SemanticScholar
from sources.arxiv import ArXiv
from sources.pubmed import PubMed
from sources.crossref import CrossRef
from sources.base_source import Paper
from validation.quality_scorer import QualityScorer
from validation.deduplicator import Deduplicator
from config.settings import settings


class ResearchAgent:
    """
    Domain-agnostic academic research agent.
    Searches multiple sources, validates, and returns quality papers.
    """
    
    def __init__(self):
        # Initialize components
        self.analyzer = QueryAnalyzer()
        self.quality_scorer = QualityScorer()
        self.deduplicator = Deduplicator()
        
        # Initialize data sources
        self.sources = {
            'semantic_scholar': SemanticScholar(),
            'arxiv': ArXiv(),
            'pubmed': PubMed(),
            'crossref': CrossRef()
        }
        
        print("✓ Research Agent initialized")
    
    def search(
        self,
        query: str,
        limit: int = 50,
        min_quality: float = None,
        sources: List[str] = None
    ) -> Dict:
        """
        Execute a full research search.
        
        Args:
            query: Research query string
            limit: Maximum papers to return
            min_quality: Minimum quality score (0-1)
            sources: List of sources to use (default: based on field detection)
            
        Returns:
            {
                'query': str,
                'analysis': dict (query analysis result),
                'strategy': dict (search strategy used),
                'papers': List[dict] (found papers),
                'statistics': dict (search statistics),
                'timestamp': str
            }
        """
        print(f"\n{'='*80}")
        print(f"🔍 Research Query: {query}")
        print('='*80)
        
        # Step 1: Analyze query
        print("\n📊 Step 1: Analyzing query...")
        analysis = self.analyzer.analyze(query)
        print(f"   Fields detected: {analysis['fields']}")
        print(f"   Research types: {analysis['research_types']}")
        print(f"   Key concepts: {analysis['key_concepts'][:5]}")
        
        # Step 2: Generate search strategy
        print("\n📋 Step 2: Generating search strategy...")
        strategy = SearchStrategy(analysis)
        search_plan = strategy.generate()
        all_queries = strategy.get_all_queries()
        print(f"   Generated {len(all_queries)} query variations")
        
        # Determine which sources to use
        if sources is None:
            sources = search_plan['field_specific_sources']
        print(f"   Sources: {sources}")
        
        # Step 3: Execute searches
        print("\n🌐 Step 3: Searching academic databases...")
        all_papers = []
        
        import time
        
        for source_name in sources:
            if source_name not in self.sources:
                print(f"   ⚠️  Unknown source: {source_name}")
                continue
            
            source = self.sources[source_name]
            
            # Search with multiple query variations (limit to avoid rate limits)
            queries_to_run = all_queries[:2]  # Only 2 queries per source
            for query_variant in queries_to_run:
                papers = source.search(query_variant, limit=25)
                all_papers.extend(papers)
                time.sleep(0.5)  # Small delay to avoid rate limiting
        
        print(f"\n   Total raw papers: {len(all_papers)}")
        
        # Step 4: Deduplicate
        print("\n🔄 Step 4: Deduplicating...")
        unique_papers = self.deduplicator.deduplicate(all_papers)
        
        # Step 5: Quality filter
        print("\n⭐ Step 5: Quality filtering...")
        if min_quality:
            self.quality_scorer.min_score = min_quality
        
        quality_papers = self.quality_scorer.filter_by_quality(unique_papers)
        
        # Step 6: Limit results
        final_papers = quality_papers[:limit]
        
        # Compile statistics
        stats = self._compile_statistics(all_papers, unique_papers, final_papers)
        
        print(f"\n✅ Final result: {len(final_papers)} papers")
        
        return {
            'query': query,
            'analysis': analysis,
            'strategy': {
                'fields': analysis['fields'],
                'sources_used': sources,
                'queries_generated': len(all_queries)
            },
            'papers': [p.to_dict() for p in final_papers],
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _compile_statistics(
        self,
        raw: List[Paper],
        unique: List[Paper],
        final: List[Paper]
    ) -> Dict:
        """Compile search statistics"""
        return {
            'raw_count': len(raw),
            'unique_count': len(unique),
            'final_count': len(final),
            'with_doi': sum(1 for p in final if p.doi),
            'avg_citations': round(
                sum(p.citation_count or 0 for p in final) / len(final)
            ) if final else 0,
            'avg_quality_score': round(
                sum(p.quality_score for p in final) / len(final), 2
            ) if final else 0,
            'year_range': {
                'min': min((p.year for p in final if p.year), default=0),
                'max': max((p.year for p in final if p.year), default=0)
            },
            'sources': list(set(p.source for p in final))
        }
    
    def format_for_llm(self, result: Dict) -> str:
        """
        Format search results for LLM consumption.
        
        Args:
            result: Output from search()
            
        Returns:
            Formatted string for LLM prompt
        """
        papers = result['papers']
        stats = result['statistics']
        
        if not papers:
            return """⚠️ WARNING: No academic papers found for this query.
The search system attempted multiple query variations but could not locate relevant sources.
Please generate analysis based on general knowledge, but clearly mark any claims as "Unverified - No Source"."""
        
        lines = [
            f"## Verified Academic Sources ({len(papers)} papers)",
            f"### Statistics: {stats['with_doi']}/{len(papers)} have DOI | " 
            f"Avg Citations: {stats['avg_citations']} | "
            f"Avg Quality: {stats['avg_quality_score']:.0%}",
            ""
        ]
        
        for i, paper in enumerate(papers, 1):
            # Format authors
            authors = paper.get('authors', [])
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            elif authors:
                author_str = ", ".join(authors)
            else:
                author_str = "Unknown"
            
            # Format citation
            year = paper.get('year', 'n.d.')
            title = paper.get('title', 'Untitled')
            venue = paper.get('venue', '')
            doi = paper.get('doi', '')
            citations = paper.get('citation_count', 0)
            quality = paper.get('quality_score', 0)
            
            line = f"[{i}] {author_str} ({year}). {title}."
            if venue:
                line += f" {venue}."
            if doi:
                line += f" https://doi.org/{doi}"
            line += f" (Citations: {citations}) [Quality: {quality:.0%}]"
            
            # Add abstract excerpt
            abstract = paper.get('abstract', '')
            if abstract:
                line += f"\n    Abstract: {abstract[:200]}..."
            
            lines.append(line)
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Example usage of the Research Agent"""
    agent = ResearchAgent()
    
    # Test with different research queries
    test_queries = [
        "Mechanisms of cryptocurrency liquidation cascades",
        # "CRISPR gene editing efficiency",
        # "Climate change impact on coral reefs"
    ]
    
    for query in test_queries:
        result = agent.search(query, limit=10)
        
        print("\n" + "="*80)
        print("FORMATTED OUTPUT FOR LLM:")
        print("="*80)
        print(agent.format_for_llm(result))
        
        # Save results
        output_file = Path(f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
