"""
LLM-Powered Domain Classifier - Uses AI to detect academic field from query.
Much smarter than rule-based approach.
"""

import json
from typing import Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMClient


class LLMDomainClassifier:
    """
    Use LLM to detect academic domain from research query.
    Much smarter than keyword matching.
    """
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient(primary="gemini", fallback="deepseek")
        
        # Schema for LLM response
        self.response_schema = {
            "primary_domain": "string - one of: computer_science, medicine, biology, physics, chemistry, economics, psychology, sociology, engineering, environmental_science, mathematics, general",
            "confidence": "float - 0.0 to 1.0",
            "secondary_domains": "array of strings - other relevant domains",
            "key_concepts": "array of strings - 3-7 main concepts from query",
            "research_types": "array of strings - e.g., review, empirical, theoretical, methodology",
            "is_multidisciplinary": "boolean",
            "reasoning": "string - brief explanation of classification"
        }
    
    def classify(self, query: str) -> Dict:
        """
        Classify research query using LLM.
        
        Args:
            query: User's research question
        
        Returns:
            Classification with domain, concepts, and confidence
        """
        system_prompt = """You are an expert academic librarian and research classifier.
Your task is to analyze research queries and determine:
1. The primary academic field/domain
2. Key concepts and terminology
3. Type of research being requested
4. Whether it's multidisciplinary

Be precise but flexible. Consider:
- Keywords and terminology specific to fields
- Methodological approaches mentioned
- Context and phrasing
- Interdisciplinary connections

Available domains:
- computer_science: AI, algorithms, software, networks, databases, blockchain, crypto, DeFi
- medicine: clinical, patients, diseases, treatments, drugs, therapy, healthcare
- biology: genes, proteins, cells, organisms, evolution, ecology
- physics: particles, energy, quantum, relativity, mechanics
- chemistry: molecules, reactions, compounds, synthesis
- economics: markets, finance, trade, monetary, fiscal policy, cryptocurrency markets
- psychology: behavior, cognition, mental health, therapy
- sociology: society, culture, demographics, institutions
- engineering: design, systems, optimization, manufacturing
- environmental_science: climate, ecology, sustainability, conservation
- mathematics: proofs, theorems, equations, analysis
- general: if truly interdisciplinary or doesn't fit above

Respond ONLY with valid JSON matching the schema provided."""

        user_prompt = f"""Classify this research query:

"{query}"

Provide a detailed classification with primary domain, key concepts, and confidence."""

        try:
            response = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=self.response_schema,
                temperature=0.1
            )
            
            # Validate response
            required_keys = ['primary_domain', 'confidence', 'key_concepts']
            if not all(k in response for k in required_keys):
                raise ValueError(f"LLM response missing required keys")
            
            # Ensure lists
            if not isinstance(response.get('key_concepts'), list):
                response['key_concepts'] = [response.get('key_concepts', query)]
            if not isinstance(response.get('secondary_domains'), list):
                response['secondary_domains'] = []
            if not isinstance(response.get('research_types'), list):
                response['research_types'] = ['general']
            
            print(f"✓ LLM classified as: {response['primary_domain']} ({response['confidence']:.0%})")
            return response
            
        except Exception as e:
            print(f"❌ LLM classification failed: {e}")
            # Fallback to safe default
            return {
                'primary_domain': 'general',
                'confidence': 0.3,
                'secondary_domains': [],
                'key_concepts': query.lower().split()[:5],
                'research_types': ['general'],
                'is_multidisciplinary': True,
                'reasoning': f'LLM classification failed, using fallback. Error: {str(e)}'
            }


def test_llm_classifier():
    """Test the LLM classifier"""
    classifier = LLMDomainClassifier()
    
    test_queries = [
        "Mechanisms of cryptocurrency liquidation cascades in DeFi",
        "CRISPR-Cas9 gene editing efficiency in mammalian cells",
        "Deep learning models for protein structure prediction"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = classifier.classify(query)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_llm_classifier()
