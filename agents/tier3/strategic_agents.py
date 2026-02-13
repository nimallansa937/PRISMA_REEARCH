"""
Tier 3 Strategic Agents - High-level decision-making.
Uses DeepSeek as primary (better reasoning), Gemini as fallback.

Agents:
- AdaptiveStoppingAgent: Intelligent search termination decisions
- SynthesisCoordinatorAgent: Orchestrates multi-phase synthesis
- ReportComposerAgent: Intelligent report structure and generation
- CitationCrawlStrategyAgent: Optimizes citation network exploration
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import Tier3Agent


class AdaptiveStoppingAgent(Tier3Agent):
    """
    Makes intelligent decisions about when to stop searching.
    Uses DeepSeek (primary) for complex multi-factor reasoning.

    Considers:
      - Coverage score and gap severity
      - Diminishing returns rate
      - Target paper count
      - Round budget
      - Source saturation
    """

    def __init__(self, max_rounds: int = 10):
        super().__init__(
            name="AdaptiveStopper",
            description="Intelligent search termination decisions"
        )
        self.max_rounds = max_rounds
        self.decision_history: List[Dict] = []

    def execute(self, input_data: Dict) -> Dict:
        """Decide whether to continue searching."""
        should_continue, reason, strategy = self.decide(
            coverage_score=input_data.get('coverage_score', 0),
            gaps=input_data.get('gaps', []),
            round_number=input_data.get('round_number', 0),
            papers_found=input_data.get('papers_found', 0),
            target_papers=input_data.get('target_papers', 500),
            discovery_rates=input_data.get('discovery_rates', []),
            source_saturation=input_data.get('source_saturation', {}),
        )

        return {
            'should_continue': should_continue,
            'reason': reason,
            'strategy': strategy,
        }

    def decide(self, coverage_score: float, gaps: List[Dict],
               round_number: int, papers_found: int, target_papers: int,
               discovery_rates: List[float],
               source_saturation: Dict) -> Tuple[bool, str, Dict]:
        """Multi-factor stopping decision."""

        # Hard stops (no LLM needed)
        if round_number >= self.max_rounds:
            return False, f"Max rounds ({self.max_rounds}) reached", {}

        if papers_found >= target_papers * 1.5:
            return False, f"Exceeded target ({papers_found}/{target_papers})", {}

        # Soft factors for LLM decision
        high_gaps = [g for g in gaps if g.get('severity') == 'high']
        medium_gaps = [g for g in gaps if g.get('severity') == 'medium']

        # Diminishing returns
        avg_recent_rate = 0
        if len(discovery_rates) >= 2:
            avg_recent_rate = sum(discovery_rates[-3:]) / min(len(discovery_rates), 3)

        # Source saturation
        saturated_sources = sum(
            1 for v in source_saturation.values() if v >= 0.9
        )

        context = {
            'coverage_score': coverage_score,
            'high_gaps': len(high_gaps),
            'medium_gaps': len(medium_gaps),
            'round_number': round_number,
            'papers_found': papers_found,
            'target_papers': target_papers,
            'avg_discovery_rate': avg_recent_rate,
            'saturated_sources': saturated_sources,
            'total_sources': len(source_saturation),
        }

        # Rule-based fast path
        if coverage_score >= 85 and not high_gaps:
            decision = (False, f"Excellent coverage ({coverage_score:.0f}%), no critical gaps", {})
            self._log_decision(context, decision)
            return decision

        if coverage_score >= 70 and not high_gaps and avg_recent_rate < 0.05:
            decision = (False, "Good coverage, diminishing returns", {})
            self._log_decision(context, decision)
            return decision

        if papers_found >= target_papers and coverage_score >= 60:
            decision = (False, f"Target reached ({papers_found} papers)", {})
            self._log_decision(context, decision)
            return decision

        # LLM-assisted decision for complex cases
        try:
            strategy = self._get_llm_strategy(context, gaps)
            should_continue = strategy.get('continue', True)
            reason = strategy.get('reason', 'LLM strategy')

            decision = (should_continue, reason, strategy)
            self._log_decision(context, decision)
            return decision

        except Exception as e:
            # Fallback: continue if high gaps exist
            should_continue = len(high_gaps) > 0
            reason = f"Fallback: {'high gaps remain' if should_continue else 'no critical gaps'}"
            return should_continue, reason, {}

    def _get_llm_strategy(self, context: Dict, gaps: List[Dict]) -> Dict:
        """Use LLM for nuanced stopping decision."""
        system_prompt = """You are a systematic review strategist.
Decide whether additional search rounds are worthwhile given the current state."""

        user_prompt = f"""## Current Search State
{json.dumps(context, indent=2)}

## Remaining Gaps
{json.dumps(gaps[:5], indent=2)}

## Decision Required
Should we continue searching? Consider:
1. Is coverage adequate for a rigorous review?
2. Are remaining gaps addressable with more searching?
3. Are we getting diminishing returns?
4. Is the cost of another round justified?

## Output JSON
{{
  "continue": true|false,
  "reason": "Clear justification",
  "priority_gaps": ["gap1 to focus on"],
  "suggested_sources": ["source1", "source2"],
  "estimated_additional_yield": "10-30 papers"
}}"""

        schema = {
            "continue": "boolean",
            "reason": "string"
        }

        return self._call_llm(system_prompt, user_prompt, schema)

    def _log_decision(self, context: Dict, decision: tuple):
        """Log the stopping decision."""
        self.decision_history.append({
            'context': context,
            'should_continue': decision[0],
            'reason': decision[1],
        })


class SynthesisCoordinatorAgent(Tier3Agent):
    """
    Orchestrates multi-phase synthesis across the paper corpus.
    Uses DeepSeek (primary) for strategic coordination.

    Responsibilities:
      - Decide synthesis strategy based on corpus characteristics
      - Coordinate map-reduce phases
      - Determine which deep analysis agents to activate
      - Merge results from multiple synthesis agents
    """

    def __init__(self):
        super().__init__(
            name="SynthesisCoordinator",
            description="Orchestrates multi-phase synthesis"
        )

    def execute(self, input_data: Dict) -> Dict:
        """Plan and coordinate synthesis."""
        papers = input_data.get('papers', [])
        clusters = input_data.get('clusters', [])
        query = input_data.get('query', '')

        plan = self.plan_synthesis(papers, clusters, query)
        return plan

    def plan_synthesis(self, papers: List[Dict], clusters: List[Dict],
                       query: str) -> Dict:
        """Create a synthesis execution plan."""
        corpus_stats = self._analyze_corpus(papers, clusters)

        system_prompt = """You are a synthesis strategist.
Design the optimal synthesis approach for a systematic review corpus.
Consider corpus size, diversity, and the research question."""

        user_prompt = f"""## Research Query
{query}

## Corpus Statistics
{json.dumps(corpus_stats, indent=2)}

## Available Synthesis Agents
1. MapReduceSynthesizer - handles large corpora in chunks
2. ContradictionAnalyzer - finds conflicting findings
3. TemporalEvolutionAnalyzer - tracks research trends over time
4. CausalChainExtractor - builds A→B→C causal chains
5. ConsensusQuantifier - measures agreement strength
6. PredictiveInsightsGenerator - forecasts future directions

## Task
Create a synthesis execution plan:
- Which agents to activate and in what order
- How to split the corpus for map-reduce (chunk size)
- Which agents to skip if corpus is too small
- How to merge results

## Output JSON
{{
  "strategy": "comprehensive|focused|minimal",
  "map_reduce_config": {{
    "chunk_size": 15,
    "overlap": 2,
    "max_chunks": 50
  }},
  "agents_to_activate": [
    {{
      "agent": "ContradictionAnalyzer",
      "priority": 1,
      "reason": "Large corpus with diverse findings",
      "input_subset": "top_50_by_relevance"
    }}
  ],
  "agents_to_skip": [
    {{
      "agent": "TemporalEvolutionAnalyzer",
      "reason": "Narrow date range (<3 years)"
    }}
  ],
  "merge_strategy": "How to combine results from all agents"
}}"""

        schema = {
            "strategy": "string",
            "map_reduce_config": "object",
            "agents_to_activate": "array"
        }

        try:
            plan = self._call_llm(system_prompt, user_prompt, schema)
            plan['corpus_stats'] = corpus_stats
            return plan

        except Exception as e:
            print(f"  Synthesis planning failed: {e} - using default plan")
            return self._default_plan(corpus_stats)

    def _analyze_corpus(self, papers: List[Dict], clusters: List[Dict]) -> Dict:
        """Analyze corpus characteristics for planning."""
        years = [p.get('year', 0) for p in papers if p.get('year')]
        citations = [p.get('citation_count', 0) or 0 for p in papers]
        sources = set(p.get('source', '') for p in papers)
        has_abstracts = sum(1 for p in papers if len(p.get('abstract', '') or '') > 50)

        return {
            'total_papers': len(papers),
            'year_range': f"{min(years)}-{max(years)}" if years else 'N/A',
            'year_span': max(years) - min(years) if years else 0,
            'avg_citations': sum(citations) / max(len(citations), 1),
            'max_citations': max(citations) if citations else 0,
            'source_count': len(sources),
            'sources': list(sources),
            'abstract_coverage': has_abstracts / max(len(papers), 1),
            'cluster_count': len(clusters),
            'avg_cluster_size': len(papers) / max(len(clusters), 1) if clusters else 0,
        }

    def _default_plan(self, stats: Dict) -> Dict:
        """Default synthesis plan when LLM fails."""
        n = stats['total_papers']
        return {
            'strategy': 'comprehensive' if n > 100 else 'focused' if n > 30 else 'minimal',
            'map_reduce_config': {
                'chunk_size': 15 if n > 50 else 10,
                'overlap': 2,
                'max_chunks': 50
            },
            'agents_to_activate': [
                {'agent': 'ContradictionAnalyzer', 'priority': 1,
                 'reason': 'Standard', 'input_subset': 'all'},
                {'agent': 'TemporalEvolutionAnalyzer', 'priority': 2,
                 'reason': 'Standard', 'input_subset': 'all'},
                {'agent': 'ConsensusQuantifier', 'priority': 3,
                 'reason': 'Standard', 'input_subset': 'all'},
            ],
            'agents_to_skip': [],
            'merge_strategy': 'Sequential combination',
            'corpus_stats': stats,
        }


class ReportComposerAgent(Tier3Agent):
    """
    Intelligent report structure generation.
    Uses DeepSeek (primary) for narrative coherence.

    Responsibilities:
      - Determine optimal report structure based on findings
      - Generate executive summary
      - Write section transitions
      - Create actionable recommendations
    """

    def __init__(self):
        super().__init__(
            name="ReportComposer",
            description="Intelligent report generation"
        )

    def execute(self, input_data: Dict) -> Dict:
        """Compose the final report."""
        synthesis = input_data.get('synthesis', {})
        papers = input_data.get('papers', [])
        query = input_data.get('query', '')
        clusters = input_data.get('clusters', [])
        deep_analysis = input_data.get('deep_analysis', {})
        prisma_flow = input_data.get('prisma_flow', {})

        report = self.compose_report(
            query, papers, synthesis, clusters, deep_analysis, prisma_flow
        )
        return report

    def compose_report(self, query: str, papers: List[Dict],
                       synthesis: Dict, clusters: List[Dict],
                       deep_analysis: Dict, prisma_flow: Dict) -> Dict:
        """Generate an intelligently structured report."""

        # Gather all synthesis results
        final = synthesis.get('final_synthesis', {})
        reduced = synthesis.get('reduced_synthesis', {})
        contradictions = deep_analysis.get('contradictions', [])
        temporal = deep_analysis.get('temporal_evolution', {})
        causal = deep_analysis.get('causal_chains', [])
        consensus = deep_analysis.get('consensus', {})
        predictions = deep_analysis.get('predictions', [])

        system_prompt = """You are an expert academic report writer.
Synthesize all analysis results into a coherent narrative.
Write in academic style but make it actionable and insightful."""

        user_prompt = f"""## Research Query
{query}

## Key Statistics
- Papers analyzed: {len(papers)}
- Topic clusters: {len(clusters)}
- PRISMA included: {prisma_flow.get('included', 0)}
- Sources: {prisma_flow.get('by_source', {})}

## Synthesis Results (Summary)
- Key findings: {len(final.get('key_findings', []))}
- Major themes: {len(reduced.get('major_themes', []))}
- Contradictions: {len(contradictions)}
- Causal chains: {len(causal)}
- Predictions: {len(predictions)}

## Top Findings
{json.dumps(final.get('key_findings', [])[:5], indent=2)}

## Major Themes
{json.dumps(reduced.get('major_themes', [])[:5], indent=2)}

## Task
Generate the report structure and key narrative elements.

## Output JSON
{{
  "title": "Descriptive report title",
  "executive_summary": "3-5 sentence executive summary",
  "key_takeaways": ["takeaway1", "takeaway2", "takeaway3"],
  "section_order": ["section1", "section2"],
  "recommendations": [
    {{
      "recommendation": "Actionable recommendation",
      "evidence_strength": "STRONG|MEDIUM|WEAK",
      "priority": "HIGH|MEDIUM|LOW"
    }}
  ],
  "limitations": ["limitation1"],
  "confidence_assessment": "Overall confidence in findings"
}}"""

        schema = {
            "title": "string",
            "executive_summary": "string",
            "key_takeaways": "array",
            "recommendations": "array"
        }

        try:
            report_structure = self._call_llm(system_prompt, user_prompt, schema,
                                             temperature=0.3)
            return report_structure

        except Exception as e:
            print(f"  Report composition failed: {e}")
            return {
                'title': f"Systematic Review: {query}",
                'executive_summary': f"Analysis of {len(papers)} papers.",
                'key_takeaways': [],
                'recommendations': [],
                'confidence_assessment': 'Unable to assess'
            }


class CitationCrawlStrategyAgent(Tier3Agent):
    """
    Optimizes citation network exploration strategy.
    Uses DeepSeek (primary) for strategic planning.

    Responsibilities:
      - Select optimal seed papers for snowball sampling
      - Determine crawl depth and breadth
      - Prioritize which citation directions to follow
      - Estimate expected yield
    """

    def __init__(self):
        super().__init__(
            name="CitationStrategist",
            description="Optimizes citation crawling strategy"
        )

    def execute(self, input_data: Dict) -> Dict:
        """Plan citation crawl strategy."""
        papers = input_data.get('papers', [])
        gaps = input_data.get('gaps', [])
        target_additional = input_data.get('target_additional', 200)

        strategy = self.plan_crawl(papers, gaps, target_additional)
        return strategy

    def plan_crawl(self, papers: List[Dict], gaps: List[Dict],
                   target_additional: int) -> Dict:
        """Plan optimal citation crawl strategy."""
        if len(papers) < 5:
            return {
                'seed_papers': [],
                'strategy': 'skip',
                'reason': 'Too few papers for citation crawling'
            }

        # Rank papers for seed selection
        ranked = self._rank_seed_candidates(papers)

        # Build context for LLM
        top_seeds_text = ""
        for i, p in enumerate(ranked[:15]):
            title = p.get('title', '')[:80]
            cites = p.get('citation_count', 0)
            year = p.get('year', 'N/A')
            top_seeds_text += f"[{i+1}] {title} ({year}, {cites} citations)\n"

        gaps_text = json.dumps(gaps[:5], indent=2) if gaps else "No specific gaps"

        system_prompt = """You are a citation network analysis expert.
Plan the optimal snowball sampling strategy."""

        user_prompt = f"""## Top Seed Candidates
{top_seeds_text}

## Remaining Gaps
{gaps_text}

## Target: +{target_additional} additional papers

## Task
Design a citation crawl strategy.

## Output JSON
{{
  "seed_indices": [1, 3, 5, 7],
  "crawl_depth": 1,
  "refs_per_paper": 30,
  "cites_per_paper": 20,
  "strategy_notes": "Focus on highly-cited papers to find seminal works",
  "priority_direction": "backward|forward|both",
  "estimated_yield": 150,
  "gap_targeted_seeds": [
    {{
      "seed_index": 1,
      "target_gap": "methodological gap in DeFi analysis"
    }}
  ]
}}"""

        schema = {
            "seed_indices": "array",
            "crawl_depth": "number",
            "priority_direction": "string"
        }

        try:
            strategy = self._call_llm(system_prompt, user_prompt, schema)

            # Map indices back to actual papers
            seed_indices = strategy.get('seed_indices', list(range(1, 21)))
            strategy['seed_papers'] = [
                ranked[i - 1] for i in seed_indices
                if 1 <= i <= len(ranked)
            ]

            return strategy

        except Exception as e:
            print(f"  Citation strategy planning failed: {e}")
            return self._default_strategy(ranked, target_additional)

    def _rank_seed_candidates(self, papers: List[Dict]) -> List[Dict]:
        """Rank papers by seed quality (high citations, recent, has DOI)."""
        scored = []
        for p in papers:
            score = 0
            cites = p.get('citation_count', 0) or 0
            score += min(cites / 10, 10)  # Cap at 10 points
            year = p.get('year', 0) or 0
            if year >= 2023:
                score += 3
            elif year >= 2020:
                score += 1
            if p.get('doi'):
                score += 2
            if len(p.get('abstract', '') or '') > 100:
                score += 1
            p['_seed_score'] = score
            scored.append(p)

        scored.sort(key=lambda x: x.get('_seed_score', 0), reverse=True)
        return scored

    def _default_strategy(self, ranked: List[Dict], target: int) -> Dict:
        """Default strategy when LLM fails."""
        top_n = min(20, len(ranked))
        return {
            'seed_papers': ranked[:top_n],
            'seed_indices': list(range(1, top_n + 1)),
            'crawl_depth': 1,
            'refs_per_paper': 30,
            'cites_per_paper': 20,
            'strategy_notes': 'Default: top papers by citation count',
            'priority_direction': 'both',
            'estimated_yield': min(target, top_n * 25),
        }
