"""
Complete Test Script for LLM-Powered Research Agent
Tests the full pipeline: classify -> filter -> search -> score -> export
"""
import sys
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from llm_agent import LLMPoweredAgent


def print_results(results: dict, top_n: int = 10):
    """Pretty print search results"""
    papers = results['papers']
    stats = results['statistics']
    classification = results['classification']
    
    print(f"\n{'='*80}")
    print(f"📊 RESEARCH RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"🎯 Domain: {classification['primary_domain']} ({classification['confidence']:.0%} confidence)")
    print(f"⏱️  Search time: {stats['search_time_seconds']}s")
    print(f"📄 Papers: {stats['raw_count']} found → {stats['high_quality_count']} high-quality")
    print(f"⭐ Avg Relevance: {stats['avg_relevance']:.0f}/100\n")
    
    print(f"{'='*80}")
    print(f"📚 Top {min(top_n, len(papers))} Papers (by LLM Relevance)")
    print(f"{'='*80}\n")
    
    for i, paper in enumerate(papers[:top_n], 1):
        print(f"{i}. {paper.get('title', 'Untitled')}")
        print(f"   📅 Year: {paper.get('year', 'N/A')}")
        print(f"   📖 Citations: {paper.get('citation_count', 0)}")
        print(f"   🏆 LLM Score: {paper.get('llm_relevance_score', 0)}/100")
        reasoning = paper.get('llm_reasoning', 'N/A')
        if reasoning:
            print(f"   💡 Why: {reasoning[:80]}...")
        print(f"   🔗 DOI: {paper.get('doi', 'N/A')}\n")


def export_to_csv(papers: list, filename: str):
    """Export papers to CSV file"""
    filepath = Path(filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'title', 'year', 'citation_count', 'llm_relevance_score', 
            'doi', 'source', 'llm_reasoning'
        ])
        writer.writeheader()
        
        for paper in papers:
            writer.writerow({
                'title': paper.get('title', ''),
                'year': paper.get('year', ''),
                'citation_count': paper.get('citation_count', 0),
                'llm_relevance_score': paper.get('llm_relevance_score', 0),
                'doi': paper.get('doi', ''),
                'source': paper.get('source', ''),
                'llm_reasoning': paper.get('llm_reasoning', '')
            })
    
    print(f"✅ Exported {len(papers)} papers to {filepath}")


def main():
    """Run complete agent test with your crypto query"""
    
    # Initialize agent
    agent = LLMPoweredAgent(
        primary_llm="gemini",
        fallback_llm="deepseek"
    )
    
    # Your research query
    query = "Mechanisms of cryptocurrency liquidation cascades in DeFi protocols"
    
    print(f"\n🔎 Research Query: {query}\n")
    
    # Run intelligent search
    results = agent.search(
        query=query,
        min_papers=20,
        max_papers=40,
        min_relevance=30,
        verbose=True
    )
    
    # Print results
    print_results(results, top_n=10)
    
    # Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"research_results_{timestamp}.csv"
    export_to_csv(results['papers'], csv_filename)
    
    return results


if __name__ == "__main__":
    results = main()
