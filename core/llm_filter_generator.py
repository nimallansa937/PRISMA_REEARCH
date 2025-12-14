"""
LLM-Powered Filter Generator - Uses AI to generate smart search filters.
"""

import json
from typing import Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMClient


class LLMFilterGenerator:
    """
    Use LLM to generate smart, context-aware search filters.
    Much better than hardcoded rules.
    """
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient(primary="gemini", fallback="deepseek")
        
        self.response_schema = {
            "include_keywords": "array of strings - important terms that papers MUST contain",
            "exclude_keywords": "array of strings - terms that indicate IRRELEVANT papers",
            "search_variations": "array of strings - alternative phrasings of the query",
            "min_relevance_score": "integer - 0-100 threshold for paper relevance",
            "preferred_sources": "array of strings - best databases (semantic_scholar, arxiv, pubmed, crossref)",
            "reasoning": "string - explanation of filter choices"
        }
    
    def generate_filters(self, query: str, domain_classification: Dict) -> Dict:
        """
        Generate adaptive filters using LLM intelligence.
        
        Args:
            query: Original research query
            domain_classification: Output from LLMDomainClassifier
        
        Returns:
            Filter configuration for search optimization
        """
        system_prompt = """You are an expert research librarian specializing in literature search optimization.

Your task: Generate precise filters to find relevant academic papers while excluding noise.

Guidelines for filters:

INCLUDE keywords should be:
- Core concepts from the research query
- Domain-specific terminology
- Alternative phrasings and synonyms
- Related methodologies

EXCLUDE keywords should be:
- Terms from completely different domains that might appear in false positives
- Common words that appear in unrelated contexts
- Be careful not to exclude terms that might appear in interdisciplinary papers

SEARCH VARIATIONS should be:
- Different ways to phrase the same query
- Synonyms and related terms combined
- More specific and more general versions

Available data sources:
- semantic_scholar: All fields, comprehensive, best for general searches
- arxiv: CS, physics, math, biology preprints - good for cutting-edge research
- pubmed: Medicine, biology, life sciences - essential for biomedical
- crossref: DOI verification, metadata - good for citation data

Min relevance score recommendations:
- 40-50 for narrow, specific topics
- 30-40 for interdisciplinary topics  
- 20-30 for broad explorations

Respond ONLY with valid JSON."""

        user_prompt = f"""Generate search filters for this research query:

Query: "{query}"

Domain Classification:
{json.dumps(domain_classification, indent=2)}

Consider:
1. What are the CORE concepts that MUST appear in relevant papers?
2. What terms would indicate IRRELEVANT papers from other fields?
3. What are alternative ways to phrase this query?
4. Which data sources are best for this domain?

Generate optimal filters."""

        try:
            response = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=self.response_schema,
                temperature=0.2
            )
            
            # Ensure we have lists
            if not isinstance(response.get('include_keywords'), list):
                response['include_keywords'] = domain_classification.get('key_concepts', [])
            if not isinstance(response.get('exclude_keywords'), list):
                response['exclude_keywords'] = []
            if not isinstance(response.get('search_variations'), list):
                response['search_variations'] = [query]
            if not isinstance(response.get('preferred_sources'), list):
                response['preferred_sources'] = ['semantic_scholar', 'arxiv']
            
            print(f"✓ LLM generated {len(response['include_keywords'])} include, {len(response['exclude_keywords'])} exclude filters")
            return response
            
        except Exception as e:
            print(f"❌ Filter generation failed: {e}")
            # Safe fallback
            return {
                'include_keywords': domain_classification.get('key_concepts', []),
                'exclude_keywords': [],
                'search_variations': [query],
                'min_relevance_score': 30,
                'preferred_sources': ['semantic_scholar', 'arxiv'],
                'reasoning': f'LLM failed, using fallback. Error: {str(e)}'
            }


def test_filter_generator():
    """Test the filter generator"""
    from core.llm_domain_classifier import LLMDomainClassifier
    
    classifier = LLMDomainClassifier()
    filter_gen = LLMFilterGenerator()
    
    query = "Mechanisms of cryptocurrency liquidation cascades"
    
    print(f"\nQuery: {query}\n")
    
    # Classify
    classification = classifier.classify(query)
    
    # Generate filters
    filters = filter_gen.generate_filters(query, classification)
    
    print(f"\n🛡️ Generated Filters:")
    print(json.dumps(filters, indent=2))


if __name__ == "__main__":
    test_filter_generator()
