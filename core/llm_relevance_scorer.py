"""
LLM-Powered Relevance Scorer - Uses AI to score paper relevance.
"""

import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMClient


class LLMRelevanceScorer:
    """
    Use LLM to score paper relevance with deep understanding.
    Considers nuance, context, and semantic similarity.
    """
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient(primary="gemini", fallback="deepseek")
        
        self.response_schema = {
            "relevance_score": "integer - 0 to 100",
            "reasoning": "string - brief explanation",
            "key_matches": "array of strings - specific relevant aspects",
            "concerns": "string - any issues or limitations"
        }
    
    def score_papers(
        self, 
        query: str, 
        papers: List[Dict], 
        domain: str,
        max_workers: int = 3,
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Score multiple papers using LLM.
        Uses batching to reduce API calls.
        
        Args:
            query: Research query
            papers: List of paper metadata
            domain: Academic domain
            max_workers: Parallel threads
            batch_size: Papers per LLM call
        
        Returns:
            Papers with added 'llm_relevance_score' field
        """
        if not papers:
            return []
        
        print(f"🤖 LLM scoring {len(papers)} papers...")
        
        # Score in batches for efficiency
        scored_papers = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            try:
                batch_scored = self._score_batch(query, batch, domain)
                scored_papers.extend(batch_scored)
                print(f"   Scored batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"⚠️  Batch scoring failed: {e}")
                # Add with fallback scores
                for paper in batch:
                    paper['llm_relevance_score'] = self._fallback_score(query, paper)
                    paper['llm_reasoning'] = 'Batch scoring failed, used fallback'
                    scored_papers.append(paper)
        
        # Sort by LLM score
        return sorted(scored_papers, key=lambda x: x.get('llm_relevance_score', 0), reverse=True)
    
    def _score_batch(self, query: str, papers: List[Dict], domain: str) -> List[Dict]:
        """Score a batch of papers in one LLM call"""
        
        # Prepare papers summary
        papers_text = ""
        for i, paper in enumerate(papers):
            papers_text += f"""
Paper {i+1}:
- Title: {paper.get('title', 'Unknown')[:150]}
- Abstract: {paper.get('abstract', 'No abstract')[:300]}...
- Year: {paper.get('year', 'Unknown')}
- Citations: {paper.get('citation_count', 0)}
"""

        system_prompt = f"""You are an expert academic paper reviewer in {domain}.

Score how relevant each paper is to the research query.

Scoring (0-100):
- 90-100: Directly addresses query, highly relevant
- 70-89: Closely related, covers key aspects
- 50-69: Moderately relevant
- 30-49: Tangentially related
- 10-29: Minimally relevant
- 0-9: Irrelevant

Respond with JSON array of scores for each paper."""

        user_prompt = f"""Research Query: "{query}"

Papers to Evaluate:
{papers_text}

For each paper, provide:
{{"paper_index": 1, "score": 85, "reasoning": "..."}}

Return as JSON array."""

        batch_schema = {
            "scores": "array of objects with paper_index, score, reasoning"
        }

        try:
            response = self.llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=batch_schema,
                temperature=0.1
            )
            
            # Parse scores - handle both dict and list responses
            scores_list = []
            if isinstance(response, list):
                scores_list = response
            elif isinstance(response, dict):
                scores_list = response.get('scores', [])
                # Sometimes scores might be nested differently
                if not scores_list and 'results' in response:
                    scores_list = response.get('results', [])
            
            for score_info in scores_list:
                if not isinstance(score_info, dict):
                    continue
                idx = score_info.get('paper_index', 1) - 1
                if 0 <= idx < len(papers):
                    papers[idx]['llm_relevance_score'] = score_info.get('score', 30)
                    papers[idx]['llm_reasoning'] = score_info.get('reasoning', '')
            
            # Ensure all papers have scores
            for paper in papers:
                if 'llm_relevance_score' not in paper:
                    paper['llm_relevance_score'] = self._fallback_score(query, paper)
                    paper['llm_reasoning'] = 'Not scored in batch, used fallback'
            
            return papers
            
        except Exception as e:
            print(f"⚠️  Batch scoring error: {e}")
            # Use fallback for all
            for paper in papers:
                paper['llm_relevance_score'] = self._fallback_score(query, paper)
                paper['llm_reasoning'] = f'Scoring failed: {str(e)}'
            return papers
    
    def _fallback_score(self, query: str, paper: Dict) -> int:
        """Simple keyword-based fallback scoring"""
        query_words = set(query.lower().split())
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        
        text_words = set(title.split() + abstract.split())
        overlap = len(query_words & text_words)
        
        # Simple heuristic
        score = min(overlap * 12, 50)
        
        # Boost for citations
        citations = paper.get('citation_count', 0)
        if citations > 100:
            score += 20
        elif citations > 50:
            score += 15
        elif citations > 10:
            score += 10
        
        return min(score, 100)


def test_relevance_scorer():
    """Test the relevance scorer"""
    scorer = LLMRelevanceScorer()
    
    query = "Mechanisms of cryptocurrency liquidation cascades"
    
    papers = [
        {
            'title': 'Cascading Failures in Cryptocurrency Markets',
            'abstract': 'This paper examines liquidation cascades in DeFi protocols...',
            'year': 2023,
            'citation_count': 45
        },
        {
            'title': 'Machine Learning for Weather Prediction',
            'abstract': 'We present a neural network for forecasting temperature...',
            'year': 2024,
            'citation_count': 120
        }
    ]
    
    scored = scorer.score_papers(query, papers, domain='economics')
    
    print("\n📊 Scored Papers:")
    for paper in scored:
        print(f"\nTitle: {paper['title']}")
        print(f"Score: {paper['llm_relevance_score']}/100")
        print(f"Reasoning: {paper.get('llm_reasoning', 'N/A')}")


if __name__ == "__main__":
    test_relevance_scorer()
