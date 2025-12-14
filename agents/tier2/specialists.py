"""
Tier 2 Specialist Agents - Domain LLMs for judgment-heavy tasks.
Uses Gemini as primary, DeepSeek as fallback.

Agents:
- GapDetectionAgent: Explores coverage landscape
- QueryRefinementAgent: Validates query quality
- RelevanceFilterAgent: Integrates papers into corpus
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import Tier2Agent
from agents.models import (
    ResearchGap, RefinementQuery, CoverageMatrix,
    GapType, GapSeverity
)


class GapDetectionAgent(Tier2Agent):
    """
    Analyzes literature coverage to identify systematic gaps.
    Uses Gemini (primary) for analysis.
    """
    
    def __init__(self, min_coverage_threshold: float = 0.70):
        super().__init__(
            name="GapDetector",
            description="Identifies gaps in literature coverage"
        )
        self.min_coverage_threshold = min_coverage_threshold
        self.analysis_history: List[Dict] = []
    
    def execute(self, input_data: Dict) -> Dict:
        """Main execution method"""
        papers = input_data.get('papers', [])
        dimensions = input_data.get('dimensions', {})
        round_number = input_data.get('round', 0)
        
        gaps, coverage = self.analyze_coverage(papers, dimensions, round_number)
        
        return {
            'gaps': [self._gap_to_dict(g) for g in gaps],
            'coverage_matrix': coverage.to_dict(),
            'should_continue': self._should_continue(gaps, coverage, round_number)
        }
    
    def analyze_coverage(
        self,
        papers: List[Dict],
        dimensions: Dict,
        round_number: int = 0
    ) -> Tuple[List[ResearchGap], CoverageMatrix]:
        """Perform gap analysis on current paper collection"""
        
        papers_summary = self._create_papers_summary(papers)
        
        system_prompt = """You are a systematic literature review specialist.
Analyze papers to identify coverage gaps. Be rigorous and evidence-based."""

        user_prompt = f"""Analyze these {len(papers)} papers for coverage gaps.

## Papers
{papers_summary}

## Target Dimensions
{json.dumps(dimensions, indent=2)}

## Output JSON
{{
  "identified_gaps": [
    {{
      "gap_type": "methodological|temporal|geographic|outcome",
      "description": "What's missing",
      "severity": "high|medium|low",
      "suggested_query": "Query to fill this gap"
    }}
  ],
  "coverage_matrix": {{
    "methodologies": {{"method1": count, "method2": count}},
    "years": {{"2020-2022": count, "2023-2025": count}}
  }},
  "coverage_score": 0-100
}}"""

        schema = {
            "identified_gaps": "array",
            "coverage_matrix": "object",
            "coverage_score": "number"
        }
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            
            gaps = self._parse_gaps(response.get('identified_gaps', []))
            coverage = CoverageMatrix(
                dimensions=response.get('coverage_matrix', {}),
                total_papers=len(papers),
                coverage_score=float(response.get('coverage_score', 50))
            )
            
            self.analysis_history.append({
                'round': round_number,
                'papers': len(papers),
                'gaps': len(gaps),
                'coverage': coverage.coverage_score
            })
            
            return gaps, coverage
            
        except Exception as e:
            print(f"⚠️ Gap analysis failed: {e}")
            return [], CoverageMatrix({}, len(papers), 0.0)
    
    def _create_papers_summary(self, papers: List[Dict], max_papers: int = 25) -> str:
        """Create concise summary for LLM"""
        lines = []
        for i, paper in enumerate(papers[:max_papers]):
            title = paper.get('title', 'Unknown')[:100]
            year = paper.get('year', 'N/A')
            lines.append(f"[{i+1}] {title} ({year})")
        
        if len(papers) > max_papers:
            lines.append(f"... +{len(papers) - max_papers} more papers")
        
        return "\n".join(lines)
    
    def _parse_gaps(self, gaps_data: List[Dict]) -> List[ResearchGap]:
        """Parse LLM output into ResearchGap objects"""
        gaps = []
        for g in gaps_data:
            try:
                gaps.append(ResearchGap(
                    gap_type=GapType(g['gap_type']),
                    description=g['description'],
                    severity=GapSeverity(g['severity']),
                    suggested_query=g['suggested_query'],
                    affected_dimensions=g.get('affected_dimensions', [])
                ))
            except (KeyError, ValueError):
                continue
        
        # Sort by severity
        order = {GapSeverity.HIGH: 0, GapSeverity.MEDIUM: 1, GapSeverity.LOW: 2}
        gaps.sort(key=lambda g: order[g.severity])
        return gaps
    
    def _gap_to_dict(self, gap: ResearchGap) -> Dict:
        return {
            'gap_type': gap.gap_type.value,
            'description': gap.description,
            'severity': gap.severity.value,
            'suggested_query': gap.suggested_query
        }
    
    def _should_continue(
        self,
        gaps: List[ResearchGap],
        coverage: CoverageMatrix,
        round_number: int,
        max_rounds: int = 4
    ) -> Tuple[bool, str]:
        """Stopping criteria"""
        
        if round_number >= max_rounds:
            return False, f"Max rounds ({max_rounds}) reached"
        
        if coverage.coverage_score >= 85:
            return False, f"Excellent coverage ({coverage.coverage_score:.0f}%)"
        
        high_gaps = [g for g in gaps if g.severity == GapSeverity.HIGH]
        if not high_gaps and coverage.coverage_score >= 70:
            return False, "No critical gaps remain"
        
        return True, f"{len(high_gaps)} high-severity gaps found"


class QueryRefinementAgent(Tier2Agent):
    """
    Generates optimized search queries to fill gaps.
    Uses Gemini (primary) for query construction.
    """
    
    def __init__(self):
        super().__init__(
            name="QueryRefiner",
            description="Generates optimized search queries"
        )
        self.query_history: List[Dict] = []
    
    def execute(self, input_data: Dict) -> Dict:
        """Main execution method"""
        gaps = input_data.get('gaps', [])
        round_number = input_data.get('round', 0)
        
        queries = self.generate_queries(gaps, round_number)
        
        return {
            'queries': [self._query_to_dict(q) for q in queries],
            'round': round_number
        }
    
    def generate_queries(
        self,
        gaps: List[Dict],
        round_number: int,
        max_queries: int = 3
    ) -> List[RefinementQuery]:
        """Generate queries to fill gaps"""
        
        if not gaps:
            return []
        
        system_prompt = """You are a search query optimization expert.
Generate keyword-dense academic database queries."""

        user_prompt = f"""Generate search queries to fill these gaps:

## Gaps
{json.dumps(gaps[:5], indent=2)}

## Rules
- Use 5-8 keywords per query
- Include methodological terms
- Add year ranges for temporal gaps
- Target: arXiv, Semantic Scholar, CrossRef

## Output JSON
{{
  "queries": [
    {{
      "query": "keyword1 keyword2 keyword3 2023 2024",
      "target_gap": "Which gap this addresses",
      "expected_yield": "10-20 papers",
      "database_priority": ["arxiv", "semantic_scholar"]
    }}
  ]
}}"""

        schema = {"queries": "array"}
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            
            queries = []
            for q in response.get('queries', [])[:max_queries]:
                queries.append(RefinementQuery(
                    query=q['query'],
                    target_gap_description=q.get('target_gap', ''),
                    expected_yield=q.get('expected_yield', 'Unknown'),
                    database_priority=q.get('database_priority', [])
                ))
            
            self.query_history.append({
                'round': round_number,
                'queries_generated': len(queries)
            })
            
            return queries
            
        except Exception as e:
            print(f"⚠️ Query refinement failed: {e}")
            return []
    
    def _query_to_dict(self, query: RefinementQuery) -> Dict:
        return {
            'query': query.query,
            'target_gap': query.target_gap_description,
            'expected_yield': query.expected_yield,
            'databases': query.database_priority
        }


class RelevanceFilterAgent(Tier2Agent):
    """
    Evaluates paper relevance to filter false positives.
    Uses Gemini (primary) for semantic understanding.
    """
    
    def __init__(self, relevance_threshold: float = 7.0):
        super().__init__(
            name="RelevanceFilter",
            description="Filters papers by relevance"
        )
        self.relevance_threshold = relevance_threshold
        self.evaluation_count = 0
    
    def execute(self, input_data: Dict) -> Dict:
        """Main execution method"""
        papers = input_data.get('papers', [])
        query_context = input_data.get('context', {})
        
        filtered = self.batch_evaluate(papers, query_context)
        
        return {
            'filtered_papers': filtered,
            'retention_rate': len(filtered) / len(papers) if papers else 0
        }
    
    def batch_evaluate(
        self,
        papers: List[Dict],
        query_context: Dict,
        batch_size: int = 10
    ) -> List[Dict]:
        """Evaluate and filter papers by relevance"""
        
        if not papers:
            return []
        
        query_intent = query_context.get('intent', 'Research query')
        
        all_filtered = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            filtered = self._evaluate_batch(batch, query_intent)
            all_filtered.extend(filtered)
        
        # Sort by relevance score
        all_filtered.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
        
        return all_filtered
    
    def _evaluate_batch(self, papers: List[Dict], query_intent: str) -> List[Dict]:
        """Evaluate a batch of papers"""
        
        papers_text = "\n".join([
            f"[{i+1}] {p.get('title', 'Unknown')}\n    Abstract: {p.get('abstract', 'N/A')[:200]}..."
            for i, p in enumerate(papers)
        ])
        
        system_prompt = """You are an academic paper relevance evaluator.
Score papers 0-10 for relevance to the research query."""

        user_prompt = f"""Research Intent: {query_intent}

## Papers to Evaluate
{papers_text}

## Output JSON
{{
  "evaluations": [
    {{"paper_index": 1, "score": 8, "reasoning": "Directly addresses..."}}
  ]
}}

Score 0-3 = off-topic, 4-6 = tangential, 7-10 = highly relevant"""

        schema = {"evaluations": "array"}
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            
            evaluations = {e['paper_index']: e for e in response.get('evaluations', [])}
            
            filtered = []
            for i, paper in enumerate(papers):
                eval_data = evaluations.get(i + 1, {'score': 5})
                score = float(eval_data.get('score', 5))
                
                if score >= self.relevance_threshold:
                    paper['relevance_score'] = score
                    paper['relevance_reasoning'] = eval_data.get('reasoning', '')
                    filtered.append(paper)
            
            self.evaluation_count += len(papers)
            return filtered
            
        except Exception as e:
            print(f"⚠️ Relevance evaluation failed: {e}")
            # Return all papers with default score on failure
            for paper in papers:
                paper['relevance_score'] = 5.0
            return papers
