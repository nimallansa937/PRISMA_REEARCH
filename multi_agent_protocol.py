"""
Multi-Agent Research Protocol Orchestrator.

Coordinates all 20 agents across three tiers:
- Tier 1 (3 agents): DatabaseQuery, Deduplication, PRISMACompliance
- Tier 2 (6 agents): GapDetector, QueryRefiner, RelevanceFilter,
                      Screener, QualityAssessor, ClusterThemer
- Tier 3 (11 agents): ResearchStrategist, PatternSynthesizer,
                       ContradictionAnalyzer, TemporalEvolutionAnalyzer,
                       CausalChainExtractor, ConsensusQuantifier,
                       PredictiveInsightsGenerator, AdaptiveStopper,
                       SynthesisCoordinator, ReportComposer, CitationStrategist
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Tier 1 - Scripted executors
from agents.tier1.database_agent import DatabaseQueryAgent
from agents.tier1.deduplication_agent import DeduplicationAgent
from agents.tier1.prisma_agent import PRISMAComplianceAgent

# Tier 2 - Specialist analysts (Gemini primary)
from agents.tier2.specialists import GapDetectionAgent, QueryRefinementAgent, RelevanceFilterAgent
from agents.tier2.screening_agents import ScreeningAgent, QualityTierAgent, ClusterThemingAgent

# Tier 3 - Strategic council (DeepSeek primary)
from agents.tier3.council import ResearchStrategist, PatternSynthesizer
from agents.tier3.synthesis_agents import (
    ContradictionAnalyzer,
    TemporalEvolutionAnalyzer,
    CausalChainExtractor,
    ConsensusQuantifier,
    PredictiveInsightsGenerator
)
from agents.tier3.strategic_agents import (
    AdaptiveStoppingAgent,
    SynthesisCoordinatorAgent,
    ReportComposerAgent,
    CitationCrawlStrategyAgent
)


class MultiAgentProtocol:
    """
    Orchestrates the complete multi-agent research workflow.

    Workflow:
    1. Tier 3: Query decomposition (DeepSeek)
    2. Tier 1: Baseline search (scripted)
    3. Loop:
       a. Tier 2: Gap detection (Gemini)
       b. Tier 3: Adaptive stopping decision (DeepSeek)
       c. Tier 2: Query refinement (Gemini)
       d. Tier 1: Execute refinement queries
       e. Tier 1: Deduplication
       f. Tier 2: Relevance filtering (Gemini)
    4. Tier 2: Screening + Quality assessment
    5. Tier 3: Synthesis coordination + Pattern synthesis (DeepSeek)
    6. Tier 3: Report composition
    """

    def __init__(self, max_refinement_rounds: int = 4, verbose: bool = True):
        print("Initializing Multi-Agent Research Protocol (20 agents)...")

        self.max_rounds = max_refinement_rounds
        self.verbose = verbose

        # Tier 1: Scripted Executors (no LLM)
        self.database_agent = DatabaseQueryAgent()
        self.dedup_agent = DeduplicationAgent()
        self.prisma_agent = PRISMAComplianceAgent()

        # Tier 2: Specialists (Gemini primary)
        self.gap_detector = GapDetectionAgent()
        self.query_refiner = QueryRefinementAgent()
        self.relevance_filter = RelevanceFilterAgent()
        self.screener = ScreeningAgent()
        self.quality_assessor = QualityTierAgent()
        self.cluster_themer = ClusterThemingAgent()

        # Tier 3: Council (DeepSeek primary)
        self.strategist = ResearchStrategist()
        self.synthesizer = PatternSynthesizer()
        self.adaptive_stopper = AdaptiveStoppingAgent(max_rounds=max_refinement_rounds)
        self.synthesis_coordinator = SynthesisCoordinatorAgent()
        self.report_composer = ReportComposerAgent()
        self.citation_strategist = CitationCrawlStrategyAgent()

        # Deep synthesis agents (Tier 2/3)
        self.contradiction_analyzer = ContradictionAnalyzer()
        self.temporal_analyzer = TemporalEvolutionAnalyzer()
        self.causal_extractor = CausalChainExtractor()
        self.consensus_quantifier = ConsensusQuantifier()
        self.prediction_generator = PredictiveInsightsGenerator()

        self._all_agents = [
            self.database_agent, self.dedup_agent, self.prisma_agent,
            self.gap_detector, self.query_refiner, self.relevance_filter,
            self.screener, self.quality_assessor, self.cluster_themer,
            self.strategist, self.synthesizer, self.adaptive_stopper,
            self.synthesis_coordinator, self.report_composer, self.citation_strategist,
            self.contradiction_analyzer, self.temporal_analyzer,
            self.causal_extractor, self.consensus_quantifier, self.prediction_generator,
        ]

        print(f"  Tier 1: {sum(1 for a in self._all_agents if a.tier.value == 1)} scripted agents")
        print(f"  Tier 2: {sum(1 for a in self._all_agents if a.tier.value == 2)} specialist agents (Gemini)")
        print(f"  Tier 3: {sum(1 for a in self._all_agents if a.tier.value == 3)} strategic agents (DeepSeek)")
        print(f"  Total:  {len(self._all_agents)} agents ready\n")

    def execute(self, query: str, domain: str = None) -> Dict:
        """Execute complete research protocol"""

        start_time = time.time()

        if self.verbose:
            print("=" * 80)
            print("  MULTI-AGENT RESEARCH PROTOCOL v3.0")
            print("=" * 80)
            print(f"\n  Query: {query}")
            print(f"  Domain: {domain or 'auto-detect'}\n")

        # ========== PHASE 1: STRATEGIC DECOMPOSITION (Tier 3) ==========
        if self.verbose:
            print("  PHASE 1: Query Decomposition (Tier 3 - DeepSeek)")

        decomposition = self.strategist.decompose_query(query)
        dimensions = decomposition.get('dimensions', {})
        baseline_query = decomposition.get('baseline_query', query)

        if self.verbose:
            print(f"    Sub-questions: {len(decomposition.get('sub_questions', []))}")
            print(f"    Baseline query: {baseline_query[:60]}...\n")

        # ========== PHASE 2: BASELINE SEARCH (Tier 1) ==========
        if self.verbose:
            print("  PHASE 2: Baseline Search (Tier 1 - Scripted)")

        baseline_result = self.database_agent.execute({
            'query': baseline_query,
            'databases': ['semantic_scholar', 'arxiv'],
            'limit': 30
        })
        all_papers = baseline_result['papers']

        if self.verbose:
            print(f"    Found {len(all_papers)} papers\n")

        # ========== PHASE 3: ITERATIVE REFINEMENT (Tier 2 + 3) ==========
        if self.verbose:
            print("  PHASE 3: Iterative Refinement")

        final_coverage = None
        discovery_rates = []
        round_num = 0

        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n    --- Round {round_num}/{self.max_rounds} ---")

            # Tier 2: Gap Detection
            gap_result = self.gap_detector.execute({
                'papers': all_papers,
                'dimensions': dimensions,
                'round': round_num
            })

            gaps = gap_result['gaps']
            coverage = gap_result['coverage_matrix']
            final_coverage = coverage
            coverage_score = coverage.get('coverage_score', 0)

            if self.verbose:
                print(f"    Gaps: {len(gaps)} | Coverage: {coverage_score:.0f}%")

            # Tier 3: Adaptive Stopping Decision
            stop_result = self.adaptive_stopper.execute({
                'coverage_score': coverage_score,
                'gaps': gaps,
                'round_number': round_num,
                'papers_found': len(all_papers),
                'target_papers': 200,
                'discovery_rates': discovery_rates,
                'source_saturation': {},
            })

            if not stop_result['should_continue']:
                if self.verbose:
                    print(f"    Stopping: {stop_result['reason']}")
                break

            # Tier 2: Query Refinement
            refine_result = self.query_refiner.execute({
                'gaps': gaps,
                'round': round_num
            })

            queries = refine_result['queries']
            if not queries:
                if self.verbose:
                    print("    No refinement queries needed")
                break

            if self.verbose:
                print(f"    Generated {len(queries)} refinement queries")

            # Tier 1: Execute Queries
            prev_count = len(all_papers)
            new_papers = self.database_agent.search_with_queries(
                queries=[{
                    'query': q['query'],
                    'databases': q.get('databases', ['semantic_scholar'])
                } for q in queries],
                limit_per_query=15
            )

            if self.verbose:
                print(f"    Found {len(new_papers)} new papers")

            # Tier 2: Relevance Filtering
            filter_result = self.relevance_filter.execute({
                'papers': new_papers,
                'context': {'intent': query}
            })

            filtered = filter_result['filtered_papers']
            all_papers.extend(filtered)

            # Tier 1: Deduplication
            dedup_result = self.dedup_agent.execute({'papers': all_papers})
            all_papers = dedup_result['papers']

            discovery_rate = (len(all_papers) - prev_count) / max(prev_count, 1)
            discovery_rates.append(discovery_rate)

            if self.verbose:
                print(f"    After filter+dedup: {len(all_papers)} unique papers")

            time.sleep(1)

        # ========== PHASE 4: SCREENING + QUALITY (Tier 2) ==========
        if self.verbose:
            print(f"\n  PHASE 4: Screening & Quality Assessment (Tier 2)")

        screen_result = self.screener.execute({
            'papers': all_papers,
            'query': query,
        })
        screened_papers = screen_result['included']

        quality_result = self.quality_assessor.execute({
            'papers': screened_papers,
            'domain': domain or ''
        })

        if self.verbose:
            print(f"    Screened: {len(all_papers)} -> {len(screened_papers)}")
            print(f"    Quality tiers: {quality_result['tier_distribution']}")

        # ========== PHASE 5: PATTERN SYNTHESIS (Tier 3) ==========
        if self.verbose:
            print(f"\n  PHASE 5: Pattern Synthesis (Tier 3 - DeepSeek)")

        synthesis = self.synthesizer.execute({
            'papers': screened_papers,
            'dimensions': dimensions
        })

        patterns = synthesis.get('patterns', [])

        if self.verbose:
            print(f"    Patterns identified: {len(patterns)}")

        elapsed = time.time() - start_time

        # ========== FINAL REPORT ==========
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("  RESEARCH COMPLETE")
            print(f"{'=' * 80}")
            print(f"\n  Total Papers: {len(screened_papers)}")
            print(f"  Coverage: {final_coverage.get('coverage_score', 0) if final_coverage else 0:.0f}%")
            print(f"  Patterns Found: {len(patterns)}")
            print(f"  Quality: {quality_result['tier_distribution']}")
            print(f"  Time: {elapsed:.1f}s")

        return {
            'query': query,
            'papers': screened_papers,
            'patterns': patterns,
            'decomposition': decomposition,
            'coverage': final_coverage,
            'quality_tiers': quality_result['tier_distribution'],
            'statistics': {
                'total_papers': len(screened_papers),
                'patterns_found': len(patterns),
                'refinement_rounds': round_num,
                'elapsed_seconds': elapsed,
                'agents_used': len(self._all_agents),
            }
        }

    def run_deep_synthesis(
        self,
        papers: List[Dict],
        topic: str,
        gaps: List[Dict] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Tier 3 Deep Synthesis - PhD-level analysis.

        Now coordinated by SynthesisCoordinatorAgent.
        """
        if verbose:
            print("\n" + "=" * 80)
            print("  TIER 3: DEEP SYNTHESIS ANALYSIS")
            print("=" * 80 + "\n")

        # Get synthesis plan from coordinator
        plan = self.synthesis_coordinator.execute({
            'papers': papers,
            'clusters': [],
            'query': topic
        })

        agents_to_activate = [
            a.get('agent', '') for a in plan.get('agents_to_activate', [])
        ]
        agents_to_skip = [
            a.get('agent', '') for a in plan.get('agents_to_skip', [])
        ]

        if verbose:
            print(f"  Strategy: {plan.get('strategy', 'comprehensive')}")
            print(f"  Agents to activate: {len(agents_to_activate)}")
            print(f"  Agents to skip: {len(agents_to_skip)}\n")

        results = {}

        # 1. Contradiction Analysis
        if 'ContradictionAnalyzer' not in agents_to_skip:
            if verbose:
                print("  1/5: Analyzing contradictions...")
            results['contradictions'] = self.contradiction_analyzer.execute({
                'papers': papers, 'topic': topic
            })
            if verbose:
                print(f"    Found {results['contradictions'].get('count', 0)} contradictions\n")

        # 2. Temporal Evolution
        if 'TemporalEvolutionAnalyzer' not in agents_to_skip:
            if verbose:
                print("  2/5: Tracking temporal evolution...")
            results['temporal_evolution'] = self.temporal_analyzer.execute({
                'papers': papers
            })
            if verbose:
                te = results['temporal_evolution']
                if 'error' not in te:
                    print(f"    {len(te.get('emerging_themes', []))} emerging, "
                          f"{len(te.get('declining_themes', []))} declining\n")
                else:
                    print(f"    {te['error']}\n")

        # 3. Causal Chains
        if 'CausalChainExtractor' not in agents_to_skip:
            if verbose:
                print("  3/5: Extracting causal chains...")
            results['causal_chains'] = self.causal_extractor.execute({
                'papers': papers, 'topic': topic
            })
            if verbose:
                print(f"    Extracted {results['causal_chains'].get('count', 0)} chains\n")

        # 4. Consensus Analysis
        if 'ConsensusQuantifier' not in agents_to_skip:
            if verbose:
                print("  4/5: Quantifying consensus...")
            results['consensus'] = self.consensus_quantifier.execute({
                'papers': papers
            })
            if verbose:
                c = results['consensus']
                if 'error' not in c:
                    print(f"    Analyzed {len(c.get('consensus_results', []))} themes\n")
                else:
                    print(f"    {c['error']}\n")

        # 5. Predictive Insights
        if 'PredictiveInsightsGenerator' not in agents_to_skip:
            if verbose:
                print("  5/5: Generating predictions...")
            results['predictions'] = self.prediction_generator.execute({
                'temporal_evolution': results.get('temporal_evolution', {}),
                'gaps': gaps or [],
                'papers': papers
            })
            if verbose:
                print(f"    Generated {results['predictions'].get('count', 0)} predictions\n")

        if verbose:
            print("=" * 80)
            print("  Deep synthesis complete!")
            print("=" * 80 + "\n")

        return results

    def get_agent_status(self) -> List[Dict]:
        """Get status of all agents."""
        return [a.get_status() for a in self._all_agents]

    def export_results(self, results: Dict, filename: str = None) -> str:
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"research_results_{timestamp}.json"

        export_data = {
            'query': results['query'],
            'papers': results['papers'],
            'patterns': results.get('patterns', []),
            'statistics': results['statistics'],
            'quality_tiers': results.get('quality_tiers', {}),
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"  Exported to {filename}")
        return filename


def main():
    """Test the multi-agent protocol"""

    protocol = MultiAgentProtocol(
        max_refinement_rounds=3,
        verbose=True
    )

    results = protocol.execute(
        query="Mechanisms of cryptocurrency liquidation cascades in DeFi protocols",
        domain="economics"
    )

    # Print top papers
    print(f"\n  Top 5 Papers:")
    for i, paper in enumerate(results['papers'][:5], 1):
        print(f"\n  {i}. {paper.get('title', 'Unknown')[:70]}")
        print(f"     Year: {paper.get('year', 'N/A')}")
        print(f"     Score: {paper.get('relevance_score', 'N/A')}")

    # Print patterns
    if results.get('patterns'):
        print(f"\n  Key Patterns:")
        for pattern in results['patterns'][:3]:
            print(f"\n  - {pattern.get('name', 'Unknown')}")
            print(f"    {pattern.get('insight', '')[:80]}...")

    # Export
    protocol.export_results(results)


if __name__ == "__main__":
    main()
