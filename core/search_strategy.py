"""
Search Strategy Generator - Creates multi-tier search plans for any domain.
Generates query variations, synonyms, and recommends data sources.
"""

from typing import List, Dict, Set
from itertools import combinations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.field_mappings import SOURCE_RECOMMENDATIONS, RESEARCH_SYNONYMS


class SearchStrategy:
    """Generate comprehensive search strategy for any domain"""
    
    def __init__(self, analysis: Dict):
        """
        Initialize with query analysis results.
        
        Args:
            analysis: Output from QueryAnalyzer.analyze()
        """
        self.analysis = analysis
        self.original_query = analysis['original_query']
        self.key_concepts = analysis['key_concepts']
        self.fields = analysis['fields']
        self.research_types = analysis['research_types']
    
    def generate(self) -> Dict[str, List[str]]:
        """
        Generate multi-tier search strategy.
        
        Returns:
            {
                'tier1_exact': [exact queries],
                'tier2_synonyms': [synonym-expanded queries],
                'tier3_components': [component searches],
                'tier4_related': [related field searches],
                'field_specific_sources': [recommended databases]
            }
        """
        return {
            'tier1_exact': self._tier1_exact(),
            'tier2_synonyms': self._tier2_synonyms(),
            'tier3_components': self._tier3_components(),
            'tier4_related': self._tier4_related(),
            'field_specific_sources': self._recommend_sources()
        }
    
    def _tier1_exact(self) -> List[str]:
        """Tier 1: Exact and near-exact queries"""
        queries = [
            self.original_query,  # Exact original
        ]
        
        # Add research type variations
        for research_type in self.research_types:
            if research_type != 'general':
                queries.append(f"{research_type} {self.original_query}")
                queries.append(f"{self.original_query} {research_type}")
        
        return queries
    
    def _tier2_synonyms(self) -> List[str]:
        """Tier 2: Synonym-expanded searches"""
        queries = []
        query_lower = self.original_query.lower()
        
        # Replace words with synonyms
        for term, synonyms in RESEARCH_SYNONYMS.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # Limit to 2 synonyms per term
                    expanded = query_lower.replace(term, syn)
                    if expanded != query_lower:
                        queries.append(expanded)
        
        # Add field-specific broadening
        for field in self.fields:
            if field != 'general':
                top_concepts = ' '.join(self.key_concepts[:2]) if self.key_concepts else self.original_query
                queries.append(f"{top_concepts} {field.replace('_', ' ')}")
        
        return queries[:10]  # Limit
    
    def _tier3_components(self) -> List[str]:
        """Tier 3: Component searches (pairwise concept combinations)"""
        queries = []
        top_concepts = self.key_concepts[:6]
        
        # Generate pairwise combinations
        if len(top_concepts) >= 2:
            for concept1, concept2 in combinations(top_concepts, 2):
                queries.append(f"{concept1} {concept2}")
        
        return queries[:10]  # Limit
    
    def _tier4_related(self) -> List[str]:
        """Tier 4: Related field and cross-domain searches"""
        queries = []
        
        # Universal academic search terms
        universal_terms = [
            'systematic review',
            'meta-analysis',
            'empirical study',
            'theoretical framework',
            'quantitative analysis'
        ]
        
        # Combine with top concept
        if self.key_concepts:
            top_concept = self.key_concepts[0]
            for term in universal_terms[:3]:
                queries.append(f"{top_concept} {term}")
        
        # Add cross-domain traditional terms
        if 'economics' in self.fields or 'computer_science' in self.fields:
            cross_domain = [
                'market contagion systemic risk',
                'financial cascade analysis',
                'network effects propagation'
            ]
            queries.extend(cross_domain)
        
        return queries[:8]
    
    def _recommend_sources(self) -> List[str]:
        """Recommend which data sources to prioritize based on field"""
        sources = set()
        
        for field in self.fields:
            field_sources = SOURCE_RECOMMENDATIONS.get(field, SOURCE_RECOMMENDATIONS['general'])
            sources.update(field_sources)
        
        # Always include universal sources
        sources.update(['semantic_scholar', 'crossref'])
        
        return list(sources)
    
    def get_all_queries(self) -> List[str]:
        """Get all queries flattened into a single list (deduplicated)"""
        strategy = self.generate()
        all_queries = []
        
        for tier in ['tier1_exact', 'tier2_synonyms', 'tier3_components', 'tier4_related']:
            all_queries.extend(strategy[tier])
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        
        return unique


def test_strategy():
    """Test the search strategy generator"""
    from core.query_analyzer import QueryAnalyzer
    
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "Mechanisms of cryptocurrency liquidation cascades",
        "CRISPR gene editing efficiency in mammalian cells",
        "Neural network interpretability methods",
        "Climate change impact on coral reef ecosystems"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        analysis = analyzer.analyze(query)
        strategy = SearchStrategy(analysis)
        search_plan = strategy.generate()
        
        print(f"\n🎯 TIER 1 - Exact ({len(search_plan['tier1_exact'])} queries):")
        for q in search_plan['tier1_exact'][:3]:
            print(f"   • {q}")
        
        print(f"\n🎯 TIER 2 - Synonyms ({len(search_plan['tier2_synonyms'])} queries):")
        for q in search_plan['tier2_synonyms'][:3]:
            print(f"   • {q}")
        
        print(f"\n🎯 TIER 3 - Components ({len(search_plan['tier3_components'])} queries):")
        for q in search_plan['tier3_components'][:3]:
            print(f"   • {q}")
        
        print(f"\n🎯 TIER 4 - Related ({len(search_plan['tier4_related'])} queries):")
        for q in search_plan['tier4_related'][:3]:
            print(f"   • {q}")
        
        print(f"\n📚 Recommended Sources:")
        print(f"   {', '.join(search_plan['field_specific_sources'])}")
        
        print(f"\n📊 Total unique queries: {len(strategy.get_all_queries())}")


if __name__ == "__main__":
    test_strategy()
