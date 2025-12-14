"""
Query Analyzer - Domain-agnostic research query analysis using NLP.
Extracts key concepts, detects academic field, and identifies research type.
"""

import spacy
from typing import List, Dict, Set, Optional
import re
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.field_mappings import FIELD_INDICATORS, RESEARCH_TYPES


class QueryAnalyzer:
    """Analyze research query to extract concepts and determine field"""
    
    def __init__(self):
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        self.field_indicators = FIELD_INDICATORS
        self.research_types = RESEARCH_TYPES
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze research query and extract structured information.
        
        Args:
            query: The research query string
            
        Returns:
            {
                'original_query': str,
                'key_concepts': List[str],
                'entities': List[str],
                'fields': List[str],
                'research_types': List[str],
                'temporal_scope': Optional[str],
                'is_comparative': bool,
                'comparative_entities': List[str]
            }
        """
        doc = self.nlp(query)
        
        return {
            'original_query': query,
            'key_concepts': self._extract_key_concepts(doc),
            'entities': self._extract_entities(doc),
            'fields': self._detect_fields(query),
            'research_types': self._detect_research_types(query),
            'temporal_scope': self._extract_temporal_scope(query),
            'is_comparative': self._is_comparative(query),
            'comparative_entities': self._extract_comparative_entities(query)
        }
    
    def _extract_key_concepts(self, doc) -> List[str]:
        """Extract key noun phrases and important terms"""
        concepts = []
        
        # Extract noun chunks (multi-word phrases)
        for chunk in doc.noun_chunks:
            # Filter out very short or stopword-only chunks
            text = chunk.text.lower().strip()
            if len(text.split()) >= 2 or len(text) > 4:
                concepts.append(text)
        
        # Extract important single words (nouns, verbs, adjectives)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
                if token.text.lower() not in [c for c in concepts]:
                    concepts.append(token.text.lower())
        
        # Remove duplicates, preserve order
        return list(dict.fromkeys(concepts))
    
    def _extract_entities(self, doc) -> List[str]:
        """Extract named entities (organizations, products, etc.)"""
        entities = [ent.text for ent in doc.ents]
        return list(set(entities))
    
    def _detect_fields(self, query: str) -> List[str]:
        """Detect academic field(s) based on keywords"""
        query_lower = query.lower()
        detected_fields = []
        field_scores = {}
        
        for field, indicators in self.field_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in query_lower)
            if score > 0:
                field_scores[field] = score
        
        # Sort by score and return top fields
        sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
        detected_fields = [f[0] for f in sorted_fields[:3]]  # Top 3 fields
        
        return detected_fields if detected_fields else ['general']
    
    def _detect_research_types(self, query: str) -> List[str]:
        """Detect type of research being requested"""
        query_lower = query.lower()
        detected_types = []
        
        for research_type, indicators in self.research_types.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_types.append(research_type)
        
        return detected_types if detected_types else ['general']
    
    def _extract_temporal_scope(self, query: str) -> Optional[str]:
        """Extract temporal scope (e.g., 'recent', 'last 5 years', '2020-2024')"""
        patterns = [
            r'(?:last|past)\s+(\d+)\s+years?',
            r'(\d{4})\s*[-–]\s*(\d{4})',
            r'since\s+(\d{4})',
            r'recent',
            r'latest',
            r'current'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _is_comparative(self, query: str) -> bool:
        """Check if query is asking for comparison"""
        comparative_words = [
            'compare', 'comparison', 'versus', 'vs', 'vs.',
            'difference between', 'similarities', 'contrast',
            'better than', 'worse than', 'compared to'
        ]
        query_lower = query.lower()
        return any(word in query_lower for word in comparative_words)
    
    def _extract_comparative_entities(self, query: str) -> List[str]:
        """Extract entities being compared (e.g., 'A vs B')"""
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+(?:vs\.?|versus)\s+(\w+(?:\s+\w+)?)',
            r'(?:compare|comparison of)\s+(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)',
            r'(\w+(?:\s+\w+)?)\s+(?:versus|vs\.?|or)\s+(\w+(?:\s+\w+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1), match.group(2)]
        
        return []


def test_analyzer():
    """Test the query analyzer with diverse research queries"""
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "Mechanisms of cryptocurrency liquidation cascades",
        "Recent advances in neural machine translation",
        "Comparison of BERT vs GPT for text classification",
        "Systematic review of cancer immunotherapy clinical trials since 2020",
        "Deep learning methods for protein structure prediction",
        "Climate change impact on coral reef ecosystems",
        "Effectiveness of cognitive behavioral therapy for anxiety disorders"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        result = analyzer.analyze(query)
        
        print(f"\n📌 Key Concepts: {result['key_concepts'][:5]}")
        print(f"🏷️  Entities: {result['entities']}")
        print(f"🔬 Fields: {result['fields']}")
        print(f"📄 Research Types: {result['research_types']}")
        print(f"📅 Temporal Scope: {result['temporal_scope']}")
        print(f"⚖️  Comparative: {result['is_comparative']}")
        if result['comparative_entities']:
            print(f"   Comparing: {result['comparative_entities']}")


if __name__ == "__main__":
    test_analyzer()
