"""
Enhanced Report Generator with Deep Synthesis
"""

from typing import Dict, List
from datetime import datetime


def format_contradictions(contradictions: Dict) -> str:
    """Format contradiction analysis"""
    if contradictions.get('count', 0) == 0:
        return "No significant contradictions found.\n"
    
    lines = []
    for i, c in enumerate(contradictions.get('contradictions', []), 1):
        topic = c.get('topic', 'Unknown')
        pos_a = c.get('position_a', {})
        pos_b = c.get('position_b', {})
        resolution = c.get('resolution', '')
        consensus = c.get('consensus_strength', 'UNKNOWN')
        
        lines.append(f"\n### Contradiction {i}: {topic}")
        lines.append(f"\n**Position A:** {pos_a.get('claim', 'N/A')}")
        lines.append(f"- Papers: {', '.join(map(str, pos_a.get('paper_ids', [])))}")
        lines.append(f"- Evidence Strength: {pos_a.get('evidence_strength', 'N/A')}")
        
        lines.append(f"\n**Position B:** {pos_b.get('claim', 'N/A')}")
        lines.append(f"- Papers: {', '.join(map(str, pos_b.get('paper_ids', [])))}")
        lines.append(f"- Evidence Strength: {pos_b.get('evidence_strength', 'N/A')}")
        
        lines.append(f"\n**Resolution:** {resolution}")
        lines.append(f"**Consensus Strength:** {consensus}\n")
        lines.append("-" * 80)
    
    return "\n".join(lines)


def format_temporal_evolution(temporal: Dict) -> str:
    """Format temporal trend analysis"""
    if 'error' in temporal:
        return f"Temporal analysis unavailable: {temporal['error']}\n"
    
    lines = []
    
    # Emerging themes
    emerging = temporal.get('emerging_themes', [])
    if emerging:
        lines.append("\n### 📈 Emerging Themes (2023-2025)")
        for theme in emerging:
            name = theme.get('theme', 'Unknown')
            growth = theme.get('growth_rate', 'N/A')
            count = theme.get('paper_count', 0)
            lines.append(f"- **{name}** ({growth}, {count} papers)")
    
    # Declining themes
    declining = temporal.get('declining_themes', [])
    if declining:
        lines.append("\n### 📉 Declining Themes")
        for theme in declining:
            name = theme.get('theme', 'Unknown')
            decline = theme.get('decline_rate', 'N/A')
            count = theme.get('paper_count', 0)
            lines.append(f"- **{name}** ({decline}, {count} papers)")
    
    # Stable themes
    stable = temporal.get('stable_themes', [])
    if stable:
        lines.append("\n### 📊 Stable Themes")
        lines.append(f"{', '.join(stable)}")
    
    # Interpretation
    interpretation = temporal.get('interpretation', '')
    if interpretation:
        lines.append(f"\n### 💡 Interpretation")
        lines.append(interpretation)
    
    return "\n".join(lines) + "\n"


def format_causal_chains(causal: Dict) -> str:
    """Format causal relationship chains"""
    if causal.get('count', 0) == 0:
        return "No causal chains identified.\n"
    
    lines = []
    for i, chain_data in enumerate(causal.get('causal_chains', []), 1):
        chain = chain_data.get('chain', [])
        evidence = chain_data.get('evidence_strength', 'UNKNOWN')
        loe = chain_data.get('loe_range', 'N/A')
        
        lines.append(f"\n### Chain {i}")
        
        for j, step in enumerate(chain):
            step_letter = chr(65 + j)  # A, B, C...
            desc = step.get('description', 'Unknown')
            paper_ids = ', '.join(map(str, step.get('paper_ids', [])))
            
            lines.append(f"\n**{step_letter}.** {desc}")
            lines.append(f"   (Papers: {paper_ids})")
            
            if j < len(chain) - 1:
                lines.append("   ↓ LEADS TO")
        
        lines.append(f"\n**Evidence Strength:** {evidence}")
        lines.append(f"**LOE Range:** {loe}\n")
        lines.append("-" * 80)
    
    return "\n".join(lines)


def format_consensus(consensus: Dict) -> str:
    """Format consensus analysis"""
    if 'error' in consensus:
        return f"Consensus analysis unavailable: {consensus['error']}\n"
    
    lines = []
    results = consensus.get('consensus_results', [])
    
    if not results:
        return "No consensus themes identified.\n"
    
    for i, theme_data in enumerate(results, 1):
        theme = theme_data.get('theme', 'Unknown')
        count = theme_data.get('paper_count', 0)
        avg_rel = theme_data.get('avg_relevance', 0)
        strength = theme_data.get('consensus_strength', 'UNKNOWN')
        actionable = theme_data.get('actionable', False)
        quality = theme_data.get('quality_distribution', {})
        
        lines.append(f"\n### {i}. {theme}")
        lines.append(f"- **Papers:** {count}")
        lines.append(f"- **Avg Relevance:** {avg_rel}/100")
        lines.append(f"- **Quality Distribution:**")
        lines.append(f"  - High: {quality.get('high_quality', 0)}")
        lines.append(f"  - Medium: {quality.get('medium_quality', 0)}")
        lines.append(f"  - Low: {quality.get('low_quality', 0)}")
        lines.append(f"- **Consensus Strength:** {strength}")
        lines.append(f"- **Actionable:** {'✅ Yes' if actionable else '❌ No'}\n")
    
    return "\n".join(lines)


def format_predictions(predictions: Dict) -> str:
    """Format predictive insights"""
    if predictions.get('count', 0) == 0:
        return "No predictions generated.\n"
    
    lines = []
    for i, pred in enumerate(predictions.get('predictions', []), 1):
        prediction = pred.get('prediction', 'Unknown')
        basis = pred.get('basis', 'N/A')
        confidence = pred.get('confidence', 'UNKNOWN')
        timeframe = pred.get('timeframe', 'N/A')
        testable = pred.get('testable_metric', 'N/A')
        
        lines.append(f"\n### Prediction {i}")
        lines.append(f"**{prediction}**")
        lines.append(f"\n- **Basis:** {basis}")
        lines.append(f"- **Confidence:** {confidence}")
        lines.append(f"- **Timeframe:** {timeframe}")
        lines.append(f"- **How to Test:** {testable}\n")
    
    return "\n".join(lines)


def generate_synthesis_report(
    base_results: Dict,
    synthesis: Dict,
    save_path: str = None
) -> str:
    """
    Generate complete research report with deep synthesis.
    
    Args:
        base_results: Standard research results (papers, gaps, etc.)
        synthesis: Deep synthesis results
        save_path: Optional path to save report
    
    Returns:
        Formatted markdown report
    """
    papers = base_results.get('papers', [])
    stats = base_results.get('statistics', {})
    
    report = f"""# DEEP RESEARCH REPORT

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Research Query:** {base_results.get('query', 'N/A')}

---

## 📊 Executive Summary

- **Papers Found:** {len(papers)}
- **Average Relevance:** {stats.get('avg_relevance', 0):.0f}/100
- **Search Time:** {stats.get('search_time_seconds', 0):.1f}s
- **Contradictions Found:** {synthesis['statistics'].get('contradictions_found', 0)}
- **Causal Chains Identified:** {synthesis['statistics'].get('causal_chains_found', 0)}
- **Emerging Themes:** {synthesis['statistics'].get('emerging_themes', 0)}

---

## 📚 Top Papers

"""
    
    # Add top 10 papers
    for i, paper in enumerate(papers[:10], 1):
        title = paper.get('title', 'Unknown')
        authors = paper.get('authors', [])
        author_str = authors[0] if authors else 'Unknown'
        if len(authors) > 1:
            author_str += ' et al.'
        year = paper.get('year', 'N/A')
        score = paper.get('llm_relevance_score', 0)
        
        report += f"{i}. **{title}**\n"
        report += f"   - Authors: {author_str} ({year})\n"
        report += f"   - Relevance: {score}/100\n\n"
    
    report += f"""
---

# 🔬 DEEP SYNTHESIS ANALYSIS

## 1. Contradiction Analysis

{format_contradictions(synthesis['contradictions'])}

---

## 2. Temporal Evolution

{format_temporal_evolution(synthesis['temporal_evolution'])}

---

## 3. Causal Chains

{format_causal_chains(synthesis['causal_chains'])}

---

## 4. Consensus Analysis

{format_consensus(synthesis['consensus'])}

---

## 5. Predictive Insights

{format_predictions(synthesis['predictions'])}

---

## 📝 Conclusion

This report represents a deep synthesis of {len(papers)} academic papers using multi-agent AI analysis.

**Key Findings:**
- {synthesis['statistics']['contradictions_found']} contradictions identified and resolved
- {synthesis['statistics']['causal_chains_found']} causal relationships mapped
- {synthesis['statistics']['emerging_themes']} emerging research themes detected
- {synthesis['statistics']['predictions_generated']} testable predictions generated

---

*Generated by ResearchAgent Pro - Deep Synthesis Mode*
"""
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ Report saved to: {save_path}")
    
    return report
