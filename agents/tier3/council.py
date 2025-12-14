"""
Tier 3 Council Agents - Strategic LLMs for high-level decisions.
Uses DeepSeek as primary (better reasoning), Gemini as fallback.

Agents:
- ResearchStrategist: Query decomposition, stopping decisions
- PatternSynthesizer: Cross-paper pattern recognition
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import Tier3Agent
from agents.models import CrossCuttingPattern


class ResearchStrategist(Tier3Agent):
    """
    High-level strategic decisions.
    Uses DeepSeek (primary) for complex reasoning.
    
    Responsibilities:
    - Query decomposition into sub-questions
    - Dimension matrix generation
    - Stopping decisions
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchStrategist",
            description="Strategic research planning and decisions"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Main execution - decompose query"""
        query = input_data.get('query', '')
        return self.decompose_query(query)
    
    def decompose_query(self, query: str) -> Dict:
        """Break query into sub-questions and dimensions"""
        
        system_prompt = """You are a research methodology strategist.
Decompose research queries into structured sub-questions and dimensions."""

        user_prompt = f"""Decompose this research query:

"{query}"

## Output JSON
{{
  "sub_questions": [
    {{"question": "Specific sub-question", "priority": "critical|high|medium"}}
  ],
  "dimensions": {{
    "methodological": ["method1", "method2"],
    "temporal": ["2020-2022", "2023-2025"],
    "geographic": ["region1", "region2"],
    "outcome": ["outcome1", "outcome2"]
  }},
  "baseline_query": "keyword-dense search query (5-8 keywords)"
}}"""

        schema = {
            "sub_questions": "array",
            "dimensions": "object",
            "baseline_query": "string"
        }
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            return response
        except Exception as e:
            print(f"⚠️ Query decomposition failed: {e}")
            return {
                "sub_questions": [{"question": query, "priority": "critical"}],
                "dimensions": {},
                "baseline_query": query
            }
    
    def make_stopping_decision(
        self,
        coverage_score: float,
        gaps_remaining: int,
        high_severity_gaps: int,
        round_number: int,
        papers_found: int
    ) -> Tuple[bool, str]:
        """Decide whether to continue refinement"""
        
        # Rule-based stopping (no LLM needed for simple decisions)
        if round_number >= 5:
            return False, "Maximum rounds (5) reached"
        
        if coverage_score >= 85:
            return False, f"Excellent coverage ({coverage_score:.0f}%)"
        
        if papers_found >= 150:
            return False, f"Sufficient papers found ({papers_found})"
        
        if high_severity_gaps == 0 and coverage_score >= 70:
            return False, "No critical gaps, good coverage"
        
        # Continue if high-severity gaps remain
        return True, f"{high_severity_gaps} critical gaps require attention"


class PatternSynthesizer(Tier3Agent):
    """
    Cross-paper pattern recognition and synthesis.
    Uses DeepSeek (primary) for complex reasoning.
    """
    
    def __init__(self):
        super().__init__(
            name="PatternSynthesizer",
            description="Synthesizes cross-cutting patterns from papers"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Main execution - synthesize patterns"""
        papers = input_data.get('papers', [])
        dimensions = input_data.get('dimensions', {})
        return self.synthesize(papers, dimensions)
    
    def synthesize(
        self,
        papers: List[Dict],
        dimensions: Dict
    ) -> Dict:
        """Identify cross-cutting patterns across papers"""
        
        if len(papers) < 5:
            return {"patterns": [], "insights": "Insufficient papers for synthesis"}
        
        # Prepare paper summaries
        papers_text = self._create_papers_summary(papers[:30])
        
        system_prompt = """You are a research synthesis specialist.
Identify emergent cross-cutting patterns that no single paper states explicitly."""

        user_prompt = f"""Analyze these {len(papers)} papers for cross-cutting patterns:

## Papers
{papers_text}

## Dimensions Analyzed
{json.dumps(dimensions, indent=2)}

## Output JSON
{{
  "cross_cutting_patterns": [
    {{
      "pattern_name": "Descriptive name",
      "description": "What this pattern reveals",
      "supporting_papers": [1, 5, 12],
      "emergent_insight": "What we learn from synthesis (not stated in any single paper)",
      "actionable_implication": "Practical takeaway",
      "confidence": 0.0-1.0
    }}
  ],
  "methodological_trends": ["trend1", "trend2"],
  "future_directions": ["direction1", "direction2"],
  "key_debates": ["debate1", "debate2"]
}}"""

        schema = {
            "cross_cutting_patterns": "array",
            "methodological_trends": "array",
            "future_directions": "array"
        }
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema, temperature=0.3)
            
            # Parse patterns
            patterns = []
            for p in response.get('cross_cutting_patterns', []):
                patterns.append(CrossCuttingPattern(
                    pattern_name=p.get('pattern_name', 'Unknown'),
                    description=p.get('description', ''),
                    supporting_paper_indices=p.get('supporting_papers', []),
                    actionable_insight=p.get('actionable_implication', ''),
                    confidence=float(p.get('confidence', 0.5))
                ))
            
            return {
                'patterns': [self._pattern_to_dict(p) for p in patterns],
                'methodological_trends': response.get('methodological_trends', []),
                'future_directions': response.get('future_directions', []),
                'key_debates': response.get('key_debates', [])
            }
            
        except Exception as e:
            print(f"⚠️ Pattern synthesis failed: {e}")
            return {"patterns": [], "error": str(e)}
    
    def _create_papers_summary(self, papers: List[Dict]) -> str:
        """Create concise summary for LLM"""
        lines = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'Unknown')[:80]
            year = paper.get('year', 'N/A')
            reasoning = paper.get('relevance_reasoning', '')[:50]
            lines.append(f"[{i+1}] {title} ({year})\n    Key: {reasoning}")
        return "\n\n".join(lines)
    
    def _pattern_to_dict(self, pattern: CrossCuttingPattern) -> Dict:
        return {
            'name': pattern.pattern_name,
            'description': pattern.description,
            'supporting_papers': pattern.supporting_paper_indices,
            'insight': pattern.actionable_insight,
            'confidence': pattern.confidence
        }
