"""
Map-Reduce Synthesis Engine - Upgrade #3.
Enables analysis of 1000+ papers by chunking, summarizing, and reducing.

Architecture:
  Phase 1 (MAP):    Split papers into chunks of 20-30
                    → LLM summarizes each chunk independently
  Phase 2 (REDUCE): Combine chunk summaries
                    → LLM finds cross-chunk patterns
  Phase 3 (FINAL):  Take reduced synthesis + top papers
                    → Generate final deep report
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_client import LLMClient


@dataclass
class ChunkSummary:
    """Summary of a chunk of papers."""
    chunk_id: int
    paper_count: int
    key_findings: List[str]
    methods_used: List[str]
    themes: List[str]
    contradictions: List[str]
    year_range: str
    top_papers: List[Dict]


@dataclass
class ReducedSynthesis:
    """Result of reducing multiple chunk summaries."""
    total_papers: int
    major_themes: List[Dict]
    cross_chunk_patterns: List[Dict]
    methodological_landscape: Dict
    contradictions: List[Dict]
    temporal_trends: Dict
    consensus_areas: List[Dict]
    research_gaps: List[str]


class MapReduceSynthesizer:
    """
    Processes 1000+ papers through map-reduce pattern.

    Key insight: Instead of sending all papers to LLM at once (context limit),
    we summarize chunks independently then synthesize the summaries.
    """

    def __init__(self, chunk_size: int = 25, llm: LLMClient = None, rag_engine=None):
        self.chunk_size = chunk_size
        self.llm = llm or LLMClient(primary="gemini", fallback="deepseek")
        self.rag_engine = rag_engine  # Optional RAG engine for evidence retrieval

    def synthesize(self, papers: List[Dict], query: str = "",
                   progress_callback=None) -> Dict:
        """Full map-reduce synthesis pipeline."""

        total = len(papers)
        num_chunks = math.ceil(total / self.chunk_size)

        # Phase 1: MAP - Summarize each chunk
        chunk_summaries = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total)
            chunk = papers[start:end]

            summary = self._map_chunk(chunk, i, query)
            chunk_summaries.append(summary)

            if progress_callback:
                progress_callback('map', i + 1, num_chunks, summary)

        # Phase 2: REDUCE - Combine summaries
        if len(chunk_summaries) <= 3:
            # Small enough to reduce in one pass
            reduced = self._reduce_summaries(chunk_summaries, query)
        else:
            # Multi-level reduction for very large sets
            reduced = self._hierarchical_reduce(chunk_summaries, query, progress_callback)

        if progress_callback:
            progress_callback('reduce_complete', 1, 1, reduced)

        # Phase 3: FINAL - Deep synthesis with top papers
        top_papers = self._select_top_papers(papers, chunk_summaries)
        final = self._final_synthesis(reduced, top_papers, query)

        if progress_callback:
            progress_callback('final_complete', 1, 1, final)

        return {
            'chunk_summaries': [self._summary_to_dict(s) for s in chunk_summaries],
            'reduced_synthesis': self._reduced_to_dict(reduced),
            'final_synthesis': final,
            'statistics': {
                'total_papers': total,
                'chunks_processed': num_chunks,
                'chunk_size': self.chunk_size,
                'themes_found': len(reduced.major_themes),
                'contradictions_found': len(reduced.contradictions),
                'patterns_found': len(reduced.cross_chunk_patterns)
            }
        }

    def _map_chunk(self, papers: List[Dict], chunk_id: int, query: str) -> ChunkSummary:
        """MAP phase: Summarize a single chunk of papers, with optional RAG evidence."""

        papers_text = ""
        for i, p in enumerate(papers):
            title = (p.get('title') or 'Unknown')[:120]
            year = p.get('year', 'N/A')
            abstract = (p.get('abstract') or '')[:500]
            cites = p.get('citation_count', 0)
            source = p.get('source', 'unknown')
            doi = p.get('doi', '')
            venue = p.get('venue', '')
            authors = ', '.join((p.get('authors') or ['Unknown'])[:3])
            papers_text += (f"\n[{i+1}] {title} ({year}, {cites} cites, {source})"
                           f"\n    Authors: {authors}"
                           f"\n    Venue: {venue}" + (f" | DOI: {doi}" if doi else "") +
                           f"\n    Abstract: {abstract}\n")

        # RAG enhancement: retrieve full-text evidence for this chunk's topics
        rag_evidence = ""
        if self.rag_engine:
            try:
                # Build chunk-specific query from top paper titles
                chunk_titles = ' '.join(
                    (p.get('title') or '')[:50] for p in papers[:5]
                )
                chunk_query = f"{query} {chunk_titles}"
                chunks = self.rag_engine.retrieve(chunk_query, top_k=4)
                if chunks:
                    rag_evidence = "\n## Full-Text Evidence (from RAG retrieval)\n"
                    for rc in chunks:
                        rag_evidence += rc.format_evidence(max_words=300) + "\n"
            except Exception:
                pass  # RAG enhancement is optional

        system_prompt = """You are a research synthesis specialist performing systematic literature review.
Extract key findings, methods, and themes from this batch of papers.
Be COMPREHENSIVE and SPECIFIC - cite paper numbers, mention concrete data/statistics from abstracts.
Do NOT give vague summaries like "several papers studied X". Instead say "Paper [3] found that X increased by 40%"."""

        user_prompt = f"""Research Topic: {query}

## Papers (Chunk {chunk_id + 1}, {len(papers)} papers)
{papers_text}
{rag_evidence}
## Task
Analyze this batch thoroughly. For each finding, reference the specific paper(s).

Extract:
1. Key findings - specific claims with evidence (e.g., "Paper [2] demonstrated 95% accuracy using method X")
2. Research methods used (quantitative, qualitative, mixed, survey, experiment, meta-analysis, etc.)
3. Major themes/topics covered
4. Any contradictions between papers (with paper numbers)
5. Most important/highly-cited papers and why

## Output JSON
{{
  "key_findings": ["Paper [1] found that X leads to Y with p<0.01", "Papers [3,5] both confirm Z"],
  "methods_used": ["Randomized controlled trial (Papers 1,4)", "Survey (Paper 2)"],
  "themes": ["Theme 1: description", "Theme 2: description"],
  "contradictions": ["Paper [2] reports X increases Y while Paper [5] shows opposite effect"],
  "year_range": "2020-2024",
  "top_papers": [
    {{"index": 1, "title": "...", "reason": "Highest cited, foundational framework"}}
  ]
}}"""

        schema = {
            "key_findings": "array",
            "methods_used": "array",
            "themes": "array",
            "contradictions": "array",
            "year_range": "string",
            "top_papers": "array"
        }

        try:
            response = self.llm.generate_structured(system_prompt, user_prompt, schema)

            return ChunkSummary(
                chunk_id=chunk_id,
                paper_count=len(papers),
                key_findings=response.get('key_findings', []),
                methods_used=response.get('methods_used', []),
                themes=response.get('themes', []),
                contradictions=response.get('contradictions', []),
                year_range=response.get('year_range', ''),
                top_papers=response.get('top_papers', [])
            )
        except Exception as e:
            print(f"  [MapReduce] Chunk {chunk_id} map failed: {e}")
            return ChunkSummary(
                chunk_id=chunk_id, paper_count=len(papers),
                key_findings=[], methods_used=[], themes=[],
                contradictions=[], year_range='', top_papers=[]
            )

    def _reduce_summaries(self, summaries: List[ChunkSummary], query: str) -> ReducedSynthesis:
        """REDUCE phase: Combine chunk summaries into unified synthesis."""

        total_papers = sum(s.paper_count for s in summaries)

        # Compile all data from chunks
        all_findings = []
        all_methods = []
        all_themes = []
        all_contradictions = []

        for s in summaries:
            all_findings.extend(s.key_findings)
            all_methods.extend(s.methods_used)
            all_themes.extend(s.themes)
            all_contradictions.extend(s.contradictions)

        # Build summary text for LLM
        chunks_text = ""
        for s in summaries:
            chunks_text += f"\n### Chunk {s.chunk_id + 1} ({s.paper_count} papers, {s.year_range})\n"
            chunks_text += f"Findings: {'; '.join(s.key_findings[:5])}\n"
            chunks_text += f"Methods: {', '.join(s.methods_used[:5])}\n"
            chunks_text += f"Themes: {', '.join(s.themes[:5])}\n"
            if s.contradictions:
                chunks_text += f"Contradictions: {'; '.join(s.contradictions[:3])}\n"

        system_prompt = """You are performing meta-synthesis across multiple research summaries.
Identify cross-cutting patterns, resolve contradictions, and build a unified picture.
Focus on EMERGENT insights that aren't obvious from any single chunk."""

        user_prompt = f"""Research Topic: {query}
Total Papers Analyzed: {total_papers} across {len(summaries)} chunks

## Chunk Summaries
{chunks_text}

## All Unique Themes Found
{json.dumps(list(set(all_themes))[:30])}

## All Contradictions Noted
{json.dumps(all_contradictions[:20])}

## Task
Synthesize into a unified analysis. Find:
1. Major themes (with paper count estimates)
2. Cross-chunk patterns (insights only visible when combining chunks)
3. Methodological landscape
4. Resolved contradictions
5. Temporal trends
6. Areas of strong consensus
7. Research gaps

## Output JSON
{{
  "major_themes": [
    {{"theme": "name", "prevalence": "high/medium/low", "paper_count_est": 50, "description": "..."}}
  ],
  "cross_chunk_patterns": [
    {{"pattern": "name", "insight": "What combining chunks reveals", "confidence": 0.8}}
  ],
  "methodological_landscape": {{
    "dominant_methods": ["method1"],
    "emerging_methods": ["method2"],
    "methodology_gaps": ["gap1"]
  }},
  "contradictions": [
    {{"topic": "...", "positions": ["A says X", "B says Y"], "resolution": "...", "strength": "STRONG/MEDIUM/WEAK"}}
  ],
  "temporal_trends": {{
    "emerging": ["trend1"],
    "declining": ["trend2"],
    "stable": ["trend3"]
  }},
  "consensus_areas": [
    {{"topic": "...", "strength": "VERY_STRONG/STRONG/MEDIUM", "paper_count_est": 30}}
  ],
  "research_gaps": ["gap1", "gap2"]
}}"""

        schema = {
            "major_themes": "array",
            "cross_chunk_patterns": "array",
            "methodological_landscape": "object",
            "contradictions": "array",
            "temporal_trends": "object",
            "consensus_areas": "array",
            "research_gaps": "array"
        }

        try:
            response = self.llm.generate_structured(system_prompt, user_prompt, schema, temperature=0.2)

            return ReducedSynthesis(
                total_papers=total_papers,
                major_themes=response.get('major_themes', []),
                cross_chunk_patterns=response.get('cross_chunk_patterns', []),
                methodological_landscape=response.get('methodological_landscape', {}),
                contradictions=response.get('contradictions', []),
                temporal_trends=response.get('temporal_trends', {}),
                consensus_areas=response.get('consensus_areas', []),
                research_gaps=response.get('research_gaps', [])
            )
        except Exception as e:
            print(f"  [MapReduce] Reduce failed: {e}")
            return ReducedSynthesis(
                total_papers=total_papers,
                major_themes=[], cross_chunk_patterns=[],
                methodological_landscape={}, contradictions=[],
                temporal_trends={}, consensus_areas=[], research_gaps=[]
            )

    def _hierarchical_reduce(self, summaries: List[ChunkSummary], query: str,
                              progress_callback=None) -> ReducedSynthesis:
        """Multi-level reduction for very large paper sets (50+ chunks)."""

        # Group summaries into super-chunks of 8
        super_chunk_size = 8
        level = 0
        current = summaries

        while len(current) > 3:
            level += 1
            next_level = []

            for i in range(0, len(current), super_chunk_size):
                group = current[i:i + super_chunk_size]
                reduced = self._reduce_summaries(group, query)

                # Convert reduced back to ChunkSummary for next level
                combined = ChunkSummary(
                    chunk_id=i // super_chunk_size,
                    paper_count=reduced.total_papers,
                    key_findings=[t.get('description', '') for t in reduced.major_themes[:5]],
                    methods_used=reduced.methodological_landscape.get('dominant_methods', []),
                    themes=[t.get('theme', '') for t in reduced.major_themes],
                    contradictions=[c.get('topic', '') for c in reduced.contradictions],
                    year_range='',
                    top_papers=[]
                )
                next_level.append(combined)

                if progress_callback:
                    progress_callback('reduce', len(next_level),
                                    math.ceil(len(current) / super_chunk_size),
                                    {'level': level})

            current = next_level

        return self._reduce_summaries(current, query)

    def _select_top_papers(self, papers: List[Dict],
                           summaries: List[ChunkSummary], top_n: int = 50) -> List[Dict]:
        """Select most important papers across all chunks."""
        # Score papers by: citation count + whether they were marked as top in chunks
        top_indices = set()
        for s in summaries:
            for tp in s.top_papers:
                idx = tp.get('index', 0) + (s.chunk_id * self.chunk_size) - 1
                if 0 <= idx < len(papers):
                    top_indices.add(idx)

        # Sort by citation count, prioritizing chunk-identified top papers
        scored = []
        for i, p in enumerate(papers):
            score = p.get('citation_count', 0) or 0
            if i in top_indices:
                score += 10000  # Boost chunk-identified top papers
            scored.append((score, i, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, _, p in scored[:top_n]]

    def _final_synthesis(self, reduced: ReducedSynthesis,
                         top_papers: List[Dict], query: str) -> Dict:
        """FINAL phase: Deep synthesis combining reduced analysis with top papers + RAG evidence."""

        papers_text = ""
        for i, p in enumerate(top_papers[:30]):
            title = (p.get('title') or 'Unknown')[:100]
            year = p.get('year', 'N/A')
            abstract = (p.get('abstract') or '')[:400]
            cites = p.get('citation_count', 0)
            authors = ', '.join((p.get('authors') or ['Unknown'])[:3])
            papers_text += f"[{i+1}] {title} ({year}, {cites} cites)\n    Authors: {authors}\n    {abstract}\n\n"

        synthesis_text = json.dumps(self._reduced_to_dict(reduced), indent=2)[:4000]

        # RAG enhancement: retrieve deep evidence for final synthesis
        rag_evidence_text = ""
        if self.rag_engine:
            try:
                final_chunks = self.rag_engine.retrieve(query, top_k=25)
                if final_chunks:
                    rag_evidence_text = "\n## Full-Text Evidence (RAG Retrieved)\n\n"
                    for rc in final_chunks:
                        rag_evidence_text += rc.format_evidence(max_words=400) + "\n"
            except Exception:
                pass

        system_prompt = """You are writing a comprehensive systematic review synthesis.
Produce a DETAILED executive-level research report with SPECIFIC findings, statistics, and evidence.
Do NOT give generic summaries. Each finding must reference concrete evidence from papers.
Each debate must name the opposing positions with supporting paper evidence.
Future directions must be specific and actionable, not generic "more research needed"."""

        user_prompt = f"""Research Topic: {query}

## Meta-Analysis Results ({reduced.total_papers} papers)
{synthesis_text}

## Top 30 Most Important Papers
{papers_text}
{rag_evidence_text}
## Task
Write a comprehensive final synthesis. Be SPECIFIC and DETAILED:

1. Executive summary (5-7 sentences covering scope, key results, implications)
2. State of the field (detailed paragraph, not just "growing area")
3. Key findings - each with evidence strength rating AND specific paper references
4. Unresolved debates - name the sides and the evidence for each
5. Future research directions - specific, actionable gaps
6. Practical implications - concrete applications for practitioners

Generate AT LEAST 5 key findings, 3 debates, 5 future directions, and 4 implications.

## Output JSON
{{
  "executive_summary": "5-7 sentence overview with specific scope and findings",
  "state_of_field": "Detailed state assessment paragraph",
  "key_findings": [
    {{"finding": "Specific finding with evidence detail", "evidence_strength": "STRONG/MEDIUM/WEAK", "paper_count": 50}}
  ],
  "unresolved_debates": [
    {{"debate": "Specific question", "sides": ["Position A with evidence", "Position B with evidence"], "current_evidence": "Summary of where evidence leans"}}
  ],
  "future_directions": ["Specific actionable direction 1", "direction 2"],
  "practical_implications": ["Concrete implication for practitioners", "implication 2"],
  "confidence_assessment": "HIGH/MEDIUM/LOW with explanation"
}}"""

        schema = {
            "executive_summary": "string",
            "state_of_field": "string",
            "key_findings": "array",
            "unresolved_debates": "array",
            "future_directions": "array",
            "practical_implications": "array",
            "confidence_assessment": "string"
        }

        try:
            return self.llm.generate_structured(system_prompt, user_prompt, schema, temperature=0.3)
        except Exception as e:
            print(f"  [MapReduce] Final synthesis failed: {e}")
            return {"executive_summary": f"Synthesis of {reduced.total_papers} papers completed with errors."}

    def _summary_to_dict(self, s: ChunkSummary) -> Dict:
        return {
            'chunk_id': s.chunk_id,
            'paper_count': s.paper_count,
            'key_findings': s.key_findings,
            'methods_used': s.methods_used,
            'themes': s.themes,
            'contradictions': s.contradictions,
            'year_range': s.year_range
        }

    def _reduced_to_dict(self, r: ReducedSynthesis) -> Dict:
        return {
            'total_papers': r.total_papers,
            'major_themes': r.major_themes,
            'cross_chunk_patterns': r.cross_chunk_patterns,
            'methodological_landscape': r.methodological_landscape,
            'contradictions': r.contradictions,
            'temporal_trends': r.temporal_trends,
            'consensus_areas': r.consensus_areas,
            'research_gaps': r.research_gaps
        }
