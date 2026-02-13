"""
Tier 3 Synthesis Agents - Deep multi-paper analysis.
Performs PhD-level synthesis across entire corpus.

Agents:
- ContradictionAnalyzer: Finds conflicting findings
- TemporalEvolutionAnalyzer: Tracks research trends
- CausalChainExtractor: Builds A→B→C chains
- ConsensusQuantifier: Measures agreement strength
- PredictiveInsightsGenerator: Forecasts future research
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import Tier2Agent


class ContradictionAnalyzer(Tier2Agent):
    """
    Identifies contradictions across papers and resolves them.
    Uses Gemini for semantic understanding of conflicts.
    """
    
    def __init__(self):
        super().__init__(
            name="ContradictionAnalyzer",
            description="Finds and resolves contradictions"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Find contradictions across papers"""
        papers = input_data.get('papers', [])
        research_topic = input_data.get('topic', '')
        
        contradictions = self.find_contradictions(papers, research_topic)
        
        return {
            'contradictions': contradictions,
            'count': len(contradictions)
        }
    
    def find_contradictions(self, papers: List[Dict], topic: str) -> List[Dict]:
        """Identify papers with conflicting findings"""
        
        if len(papers) < 5:
            return []
        
        # Create papers summary (top 20 papers)
        papers_text = ""
        for i, p in enumerate(papers[:20]):
            title = p.get('title', 'Unknown')[:80]
            year = p.get('year', 'N/A')
            finding = p.get('abstract', '')[:200]
            score = p.get('llm_relevance_score', 'N/A')
            
            papers_text += f"\n[{i+1}] {title} ({year}, Score: {score})\n"
            papers_text += f"    Finding: {finding}...\n"
        
        system_prompt = """You are a systematic review specialist.
Identify contradictions where papers disagree on key findings."""

        user_prompt = f"""Topic: {topic}

## Papers
{papers_text}

## Task
Find papers with **conflicting findings** on the same topic.

## Output JSON
{{
  "contradictions": [
    {{
      "topic": "What they disagree about",
      "position_a": {{
        "claim": "What papers say",
        "paper_ids": [1, 3, 5],
        "evidence_strength": "HIGH|MEDIUM|LOW"
      }},
      "position_b": {{
        "claim": "What other papers say",
        "paper_ids": [2, 4],
        "evidence_strength": "HIGH|MEDIUM|LOW"
      }},
      "resolution": "Which position has stronger evidence?",
      "consensus_strength": "STRONG|MEDIUM|WEAK"
    }}
  ]
}}

Only include REAL contradictions (not just different topics)."""

        schema = {"contradictions": "array"}
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            return response.get('contradictions', [])
        except Exception as e:
            print(f"⚠️  Contradiction analysis failed: {e}")
            return []


class TemporalEvolutionAnalyzer(Tier2Agent):
    """
    Analyzes how research themes have evolved over time.
    """
    
    def __init__(self):
        super().__init__(
            name="TemporalEvolutionAnalyzer",
            description="Tracks research trends over time"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Analyze temporal evolution"""
        papers = input_data.get('papers', [])
        
        evolution = self.analyze_evolution(papers)
        
        return evolution
    
    def analyze_evolution(self, papers: List[Dict]) -> Dict:
        """Track how themes evolved"""
        
        if len(papers) < 10:
            return {"error": "Need ≥10 papers for temporal analysis"}
        
        # Group papers by year
        by_year = defaultdict(list)
        for p in papers:
            year = p.get('year')
            if year and isinstance(year, int):
                by_year[year].append(p)
        
        # Sort years
        years = sorted(by_year.keys())
        if len(years) < 3:
            return {"error": "Need ≥3 different years"}
        
        # Create year summary
        year_summary = ""
        for year in years:
            count = len(by_year[year])
            titles = [p.get('title', '')[:60] for p in by_year[year][:3]]
            year_summary += f"\n{year} ({count} papers):\n"
            for title in titles:
                year_summary += f"  - {title}\n"
        
        system_prompt = """You are a research trend analyst.
Identify emerging and declining themes across time periods."""

        user_prompt = f"""## Papers by Year
{year_summary}

## Task
Identify trends: what's EMERGING (recent), what's DECLINING (older)?

## Output JSON
{{
  "emerging_themes": [
    {{
      "theme": "Theme name",
      "first_appeared": 2023,
      "paper_count": 8,
      "growth_rate": "+400%"
    }}
  ],
  "declining_themes": [
    {{
      "theme": "Theme name",
      "peak_year": 2020,
      "paper_count": 2,
      "decline_rate": "-75%"
    }}
  ],
  "stable_themes": ["theme1", "theme2"],
  "interpretation": "What this tells us about field evolution"
}}"""

        schema = {
            "emerging_themes": "array",
            "declining_themes": "array",
            "stable_themes": "array",
            "interpretation": "string"
        }
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            return response
        except Exception as e:
            print(f"⚠️  Temporal analysis failed: {e}")
            return {"error": str(e)}


class CausalChainExtractor(Tier2Agent):
    """
    Builds causal chains (A → B → C) from paper findings.
    """
    
    def __init__(self):
        super().__init__(
            name="CausalChainExtractor",
            description="Extracts causal relationships"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Extract causal chains"""
        papers = input_data.get('papers', [])
        topic = input_data.get('topic', '')
        
        chains = self.extract_chains(papers, topic)
        
        return {
            'causal_chains': chains,
            'count': len(chains)
        }
    
    def extract_chains(self, papers: List[Dict], topic: str) -> List[Dict]:
        """Build multi-step causal chains"""
        
        if len(papers) < 5:
            return []
        
        # Summarize findings
        findings_text = ""
        for i, p in enumerate(papers[:15]):
            title = p.get('title', '')[:70]
            finding = p.get('abstract', '')[:150]
            findings_text += f"\n[{i+1}] {title}\n    → {finding}...\n"
        
        system_prompt = """You are a causal relationship expert.
Build multi-step causal chains from research findings."""

        user_prompt = f"""Topic: {topic}

## Paper Findings
{findings_text}

## Task
Extract causal chains: A LEADS TO B LEADS TO C

## Output JSON
{{
  "causal_chains": [
    {{
      "chain": [
        {{"step": "A", "description": "High leverage", "paper_ids": [1, 3]}},
        {{"step": "B", "description": "Increased liquidations", "paper_ids": [2, 5]}},
        {{"step": "C", "description": "Market panic", "paper_ids": [4]}}
      ],
      "evidence_strength": "STRONG|MEDIUM|WEAK",
      "loe_range": "2-4"
    }}
  ]
}}

Only include chains with ≥2 steps supported by papers."""

        schema = {"causal_chains": "array"}
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            return response.get('causal_chains', [])
        except Exception as e:
            print(f"⚠️  Causal chain extraction failed: {e}")
            return []


class ConsensusQuantifier(Tier2Agent):
    """
    Builds evidence pyramids showing consensus strength.
    """
    
    def __init__(self):
        super().__init__(
            name="ConsensusQuantifier",
            description="Quantifies research consensus"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Quantify consensus on key themes"""
        papers = input_data.get('papers', [])
        
        consensus = self.quantify_consensus(papers)
        
        return consensus
    
    def quantify_consensus(self, papers: List[Dict]) -> Dict:
        """Build evidence pyramids"""
        
        if len(papers) < 5:
            return {"error": "Need ≥5 papers"}
        
        # Group by theme (simplified - use LLM to cluster)
        themes_text = ""
        for i, p in enumerate(papers[:20]):
            title = p.get('title', '')[:70]
            score = p.get('llm_relevance_score', 'N/A')
            themes_text += f"[{i+1}] {title} (Score: {score})\n"
        
        system_prompt = """You are a consensus analyst.
Quantify how strongly papers agree on key themes."""

        user_prompt = f"""## Papers
{themes_text}

## Task
For each major theme, build an evidence pyramid:
- How many papers support it?
- What's their quality (score)?
- How strong is consensus?

## Output JSON
{{
  "consensus_results": [
    {{
      "theme": "Theme name",
      "paper_count": 12,
      "avg_relevance": 85,
      "quality_distribution": {{
        "high_quality": 5,
        "medium_quality": 4,
        "low_quality": 3
      }},
      "consensus_strength": "VERY_STRONG|STRONG|MEDIUM|WEAK",
      "actionable": true
    }}
  ]
}}"""

        schema = {"consensus_results": "array"}
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            return response
        except Exception as e:
            print(f"⚠️  Consensus analysis failed: {e}")
            return {"error": str(e)}


class PredictiveInsightsGenerator(Tier2Agent):
    """
    Generates predictions about future research directions.
    """
    
    def __init__(self):
        super().__init__(
            name="PredictiveInsightsGenerator",
            description="Forecasts research trends"
        )
    
    def execute(self, input_data: Dict) -> Dict:
        """Generate predictions"""
        temporal = input_data.get('temporal_evolution', {})
        gaps = input_data.get('gaps', [])
        papers = input_data.get('papers', [])
        
        predictions = self.generate_predictions(temporal, gaps, papers)
        
        return {
            'predictions': predictions,
            'count': len(predictions)
        }
    
    def generate_predictions(self, temporal: Dict, gaps: List, papers: List[Dict]) -> List[Dict]:
        """Generate research predictions"""
        
        if not temporal or 'emerging_themes' not in temporal:
            return []
        
        emerging = temporal.get('emerging_themes', [])
        declining = temporal.get('declining_themes', [])
        
        context = f"""
Emerging themes: {json.dumps(emerging, indent=2)}
Declining themes: {json.dumps(declining, indent=2)}
Research gaps: {len(gaps)} identified
Total papers: {len(papers)}
"""
        
        system_prompt = """You are a research forecasting expert.
Predict future research directions based on current trends."""

        user_prompt = f"""## Current Research Landscape
{context}

## Task
Generate testable predictions about future research (1-3 years).

## Output JSON
{{
  "predictions": [
    {{
      "prediction": "What will happen",
      "basis": "Why we predict this",
      "confidence": "HIGH|MEDIUM|LOW",
      "timeframe": "1-2 years",
      "testable_metric": "How to verify (e.g., paper counts)"
    }}
  ]
}}

Generate 3-5 predictions."""

        schema = {"predictions": "array"}
        
        try:
            response = self._call_llm(system_prompt, user_prompt, schema)
            return response.get('predictions', [])
        except Exception as e:
            print(f"⚠️  Prediction generation failed: {e}")
            return []
