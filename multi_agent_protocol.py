"""
Multi-Agent Research Protocol Orchestrator.

Coordinates all three tiers:
- Tier 1: DatabaseQueryAgent (scripted)
- Tier 2: Gap Detector, Query Refiner, Relevance Filter (Gemini)
- Tier 3: Research Strategist, Pattern Synthesizer (DeepSeek)
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from agents.tier1.database_agent import DatabaseQueryAgent
from agents.tier2.specialists import GapDetectionAgent, QueryRefinementAgent, RelevanceFilterAgent
from agents.tier3.council import ResearchStrategist, PatternSynthesizer


class MultiAgentProtocol:
    """
    Orchestrates the complete multi-agent research workflow.
    
    Workflow:
    1. Tier 3: Query decomposition (DeepSeek)
    2. Tier 1: Baseline search (scripted)
    3. Loop:
       a. Tier 2: Gap detection (Gemini)
       b. Tier 3: Stopping decision
       c. Tier 2: Query refinement (Gemini)
       d. Tier 1: Execute refinement queries
       e. Tier 2: Relevance filtering (Gemini)
    4. Tier 3: Pattern synthesis (DeepSeek)
    """
    
    def __init__(self, max_refinement_rounds: int = 4, verbose: bool = True):
        print("🚀 Initializing Multi-Agent Research Protocol...")
        
        self.max_rounds = max_refinement_rounds
        self.verbose = verbose
        
        # Tier 1: Scripted Executors
        self.database_agent = DatabaseQueryAgent()
        
        # Tier 2: Specialists (Gemini primary)
        self.gap_detector = GapDetectionAgent()
        self.query_refiner = QueryRefinementAgent()
        self.relevance_filter = RelevanceFilterAgent()
        
        # Tier 3: Council (DeepSeek primary)
        self.strategist = ResearchStrategist()
        self.synthesizer = PatternSynthesizer()
        
        print("✓ All agents initialized\n")
    
    def execute(self, query: str, domain: str = None) -> Dict:
        """Execute complete research protocol"""
        
        start_time = time.time()
        
        if self.verbose:
            print("="*80)
            print("🔬 MULTI-AGENT RESEARCH PROTOCOL")
            print("="*80)
            print(f"\n📝 Query: {query}")
            print(f"🎯 Domain: {domain or 'auto-detect'}\n")
        
        # ========== PHASE 1: STRATEGIC DECOMPOSITION ==========
        if self.verbose:
            print("📌 PHASE 1: Query Decomposition (Tier 3 - DeepSeek)")
        
        decomposition = self.strategist.decompose_query(query)
        dimensions = decomposition.get('dimensions', {})
        baseline_query = decomposition.get('baseline_query', query)
        
        if self.verbose:
            print(f"   ✓ Sub-questions: {len(decomposition.get('sub_questions', []))}")
            print(f"   ✓ Baseline query: {baseline_query[:60]}...")
            print()
        
        # ========== PHASE 2: BASELINE SEARCH ==========
        if self.verbose:
            print("📌 PHASE 2: Baseline Search (Tier 1 - Scripted)")
        
        baseline_result = self.database_agent.execute({
            'query': baseline_query,
            'databases': ['semantic_scholar', 'arxiv'],
            'limit': 30
        })
        all_papers = baseline_result['papers']
        
        if self.verbose:
            print(f"   ✓ Found {len(all_papers)} papers")
            print()
        
        # ========== PHASE 3: ITERATIVE REFINEMENT ==========
        if self.verbose:
            print("📌 PHASE 3: Iterative Refinement (Tier 2 + Tier 3)")
        
        final_coverage = None
        
        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n   --- Round {round_num}/{self.max_rounds} ---")
            
            # Tier 2: Gap Detection
            gap_result = self.gap_detector.execute({
                'papers': all_papers,
                'dimensions': dimensions,
                'round': round_num
            })
            
            gaps = gap_result['gaps']
            coverage = gap_result['coverage_matrix']
            final_coverage = coverage
            
            if self.verbose:
                print(f"   • Gaps found: {len(gaps)}")
                print(f"   • Coverage: {coverage.get('coverage_score', 0):.0f}%")
            
            # Tier 3: Stopping Decision
            should_continue, reason = self.strategist.make_stopping_decision(
                coverage_score=coverage.get('coverage_score', 50),
                gaps_remaining=len(gaps),
                high_severity_gaps=len([g for g in gaps if g.get('severity') == 'high']),
                round_number=round_num,
                papers_found=len(all_papers)
            )
            
            if not should_continue:
                if self.verbose:
                    print(f"   ✓ Stopping: {reason}")
                break
            
            # Tier 2: Query Refinement
            refine_result = self.query_refiner.execute({
                'gaps': gaps,
                'round': round_num
            })
            
            queries = refine_result['queries']
            
            if not queries:
                if self.verbose:
                    print("   ✓ No refinement queries needed")
                break
            
            if self.verbose:
                print(f"   • Generated {len(queries)} refinement queries")
            
            # Tier 1: Execute Queries
            new_papers = self.database_agent.search_with_queries(
                queries=[{'query': q['query'], 'databases': q.get('databases', ['semantic_scholar'])} for q in queries],
                limit_per_query=15
            )
            
            if self.verbose:
                print(f"   • Found {len(new_papers)} new papers")
            
            # Tier 2: Relevance Filtering
            filter_result = self.relevance_filter.execute({
                'papers': new_papers,
                'context': {'intent': query}
            })
            
            filtered = filter_result['filtered_papers']
            
            if self.verbose:
                print(f"   • After filtering: {len(filtered)} relevant")
            
            all_papers.extend(filtered)
            
            # Rate limiting
            time.sleep(1)
        
        # ========== PHASE 4: PATTERN SYNTHESIS ==========
        if self.verbose:
            print(f"\n📌 PHASE 4: Pattern Synthesis (Tier 3 - DeepSeek)")
        
        synthesis = self.synthesizer.execute({
            'papers': all_papers,
            'dimensions': dimensions
        })
        
        patterns = synthesis.get('patterns', [])
        
        if self.verbose:
            print(f"   ✓ Patterns identified: {len(patterns)}")
        
        elapsed = time.time() - start_time
        
        # ========== FINAL REPORT ==========
        if self.verbose:
            print(f"\n{'='*80}")
            print("📊 RESEARCH COMPLETE")
            print(f"{'='*80}")
            print(f"\n✅ Total Papers: {len(all_papers)}")
            print(f"✅ Coverage: {final_coverage.get('coverage_score', 0) if final_coverage else 0:.0f}%")
            print(f"✅ Patterns Found: {len(patterns)}")
            print(f"✅ Time: {elapsed:.1f}s")
        
        return {
            'query': query,
            'papers': all_papers,
            'patterns': patterns,
            'decomposition': decomposition,
            'coverage': final_coverage,
            'statistics': {
                'total_papers': len(all_papers),
                'patterns_found': len(patterns),
                'refinement_rounds': round_num if 'round_num' in dir() else 0,
                'elapsed_seconds': elapsed
            }
        }
    
    def export_results(self, results: Dict, filename: str = None) -> str:
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"research_results_{timestamp}.json"
        
        # Convert to serializable format
        export_data = {
            'query': results['query'],
            'papers': results['papers'],
            'patterns': results['patterns'],
            'statistics': results['statistics']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported to {filename}")
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
    print(f"\n📚 Top 5 Papers:")
    for i, paper in enumerate(results['papers'][:5], 1):
        print(f"\n{i}. {paper.get('title', 'Unknown')[:70]}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   Score: {paper.get('relevance_score', 'N/A')}")
    
    # Print patterns
    if results['patterns']:
        print(f"\n🔍 Key Patterns:")
        for pattern in results['patterns'][:3]:
            print(f"\n• {pattern.get('name', 'Unknown')}")
            print(f"  {pattern.get('insight', '')[:80]}...")
    
    # Export
    protocol.export_results(results)


if __name__ == "__main__":
    main()
