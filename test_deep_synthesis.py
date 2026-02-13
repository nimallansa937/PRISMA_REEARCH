"""
Test script for LLM-Powered Research Agent with Deep Synthesis
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from llm_agent import LLMPoweredAgent
from multi_agent_protocol import MultiAgentProtocol
from synthesis.report_generator import generate_synthesis_report


def main():
    """Test agent with deep synthesis"""
    
    print("\n" + "="*80)
    print("🚀 TESTING RESEARCH AGENT PRO - DEEP SYNTHESIS MODE")
    print("="*80 + "\n")
    
    # Initialize agent
    agent = LLMPoweredAgent(primary_llm="gemini", fallback_llm="deepseek")
    
    # Test query
    query = "Mechanisms of cryptocurrency liquidation cascades"
    
    # STEP 1: Search for papers
    print("STEP 1: Searching for papers...")
    results = agent.search(
        query=query,
        min_papers=20,
        max_papers=40,
        min_relevance=30
    )
    
    papers = results['papers']
    
    print(f"\n✅ Found {len(papers)} high-quality papers")
    print(f"   Avg Relevance: {results['statistics']['avg_relevance']:.0f}/100\n")
    
    # STEP 2: Run deep synthesis
    print("STEP 2: Running deep synthesis...")
    
    protocol = MultiAgentProtocol()
    
    synthesis = protocol.run_deep_synthesis(
        papers=papers,
        topic=query,
        gaps=[],  # Can add gap analysis results here
        verbose=True
    )
    
    # STEP 3: Generate report
    print("\nSTEP 3: Generating enhanced report...\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"deep_synthesis_report_{timestamp}.md"
    
    report = generate_synthesis_report(
        base_results=results,
        synthesis=synthesis,
        save_path=report_path
    )
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SYNTHESIS SUMMARY")
    print("="*80)
    print(f"\n🔍 Contradictions: {synthesis['statistics']['contradictions_found']}")
    print(f"📈 Emerging Themes: {synthesis['statistics']['emerging_themes']}")
    print(f"⛓️  Causal Chains: {synthesis['statistics']['causal_chains_found']}")
    print(f"🤝 Consensus Themes: {synthesis['statistics']['consensus_themes']}")
    print(f"🔮 Predictions: {synthesis['statistics']['predictions_generated']}")
    
    print(f"\n✅ Full report saved: {report_path}")
    
    # Save JSON for analysis
    json_path = f"synthesis_data_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'base_results': results,
            'synthesis': synthesis
        }, f, indent=2)
    
    print(f"✅ JSON data saved: {json_path}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
