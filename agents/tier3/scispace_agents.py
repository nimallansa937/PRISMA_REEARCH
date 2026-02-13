"""
SciSpace-Replicating Agents - Chat-with-Paper + Deep Review.

These agents replicate key SciSpace capabilities:
1. PaperChatAgent (Tier 3) - Interactive Q&A over individual papers using full text
2. DeepReviewAgent (Tier 3) - Multi-pass iterative literature review with deepening context

Both use DeepSeek primary (Tier 3 strategic reasoning) with Gemini fallback.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import Tier3Agent


class PaperChatAgent(Tier3Agent):
    """
    Chat with a single paper - ask questions about its content.

    Replicates SciSpace's "Chat with PDF" feature:
    - Takes a paper's full text (or abstract if no full text)
    - Answers arbitrary questions about the paper
    - Can summarize sections, explain methodology, identify limitations
    - Generates follow-up questions to deepen understanding
    """

    def __init__(self):
        super().__init__(
            name="PaperChatAgent",
            description="Interactive Q&A over individual papers"
        )

    def execute(self, input_data: Dict) -> Dict:
        """
        Answer a question about a specific paper.

        Input:
            paper: Dict with title, abstract, full_text (optional), sections (optional)
            question: str - The question to answer
            chat_history: List[Dict] - Previous Q&A pairs (optional)
        """
        paper = input_data.get('paper', {})
        question = input_data.get('question', '')
        chat_history = input_data.get('chat_history', [])

        if not question:
            return {'error': 'No question provided'}

        answer = self.ask_paper(paper, question, chat_history)

        return answer

    def ask_paper(self, paper: Dict, question: str,
                  chat_history: List[Dict] = None) -> Dict:
        """Ask a question about a paper using its content."""

        title = paper.get('title', 'Unknown')
        abstract = paper.get('abstract', '')
        full_text = paper.get('full_text', '')
        sections = paper.get('sections', [])

        # Build context from paper content
        context = f"# {title}\n\n"

        if full_text:
            # Use chunked full text (cap at ~3000 words for LLM context)
            chunks = paper.get('chunks', [])
            if chunks:
                # Find most relevant chunks for the question
                relevant_text = self._select_relevant_chunks(chunks, question)
                context += f"## Paper Content (relevant sections)\n{relevant_text}\n\n"
            else:
                context += f"## Paper Content\n{full_text[:6000]}\n\n"
        elif sections:
            for sec in sections[:8]:
                context += f"### {sec.get('heading', 'Section')}\n{sec.get('content', '')[:1000]}\n\n"
        elif abstract:
            context += f"## Abstract\n{abstract}\n\n"
        else:
            return {'error': 'No paper content available', 'answer': ''}

        # Add metadata
        authors = ', '.join(paper.get('authors', [])[:5])
        year = paper.get('year', 'N/A')
        context += f"\n**Authors**: {authors}\n**Year**: {year}\n"

        # Build chat history context
        history_text = ""
        if chat_history:
            for h in chat_history[-5:]:  # Last 5 exchanges
                history_text += f"\nQ: {h.get('question', '')}\nA: {h.get('answer', '')}\n"

        system_prompt = """You are an expert research paper analyst.
Answer questions about the paper using ONLY information from the provided content.
If the answer is not in the paper, say so explicitly.
Be precise and cite specific parts of the paper when possible.
Suggest follow-up questions the user might want to ask."""

        user_prompt = f"""{context}

{f"## Previous Discussion{history_text}" if history_text else ""}

## Question
{question}

## Instructions
1. Answer the question using the paper content above
2. Be specific - reference sections, figures, or methods when relevant
3. If the answer isn't clearly in the paper, state what IS available
4. Suggest 2-3 follow-up questions

## Output JSON
{{
  "answer": "Your detailed answer here",
  "confidence": "HIGH|MEDIUM|LOW",
  "evidence_sections": ["Section names where answer was found"],
  "follow_up_questions": [
    "Suggested question 1",
    "Suggested question 2"
  ]
}}"""

        schema = {
            "answer": "string",
            "confidence": "string",
            "evidence_sections": "array",
            "follow_up_questions": "array"
        }

        try:
            response = self._call_llm(system_prompt, user_prompt, schema, temperature=0.2)
            response['paper_title'] = title
            return response
        except Exception as e:
            return {
                'answer': f'Error analyzing paper: {e}',
                'confidence': 'LOW',
                'evidence_sections': [],
                'follow_up_questions': []
            }

    def summarize_paper(self, paper: Dict) -> Dict:
        """Generate a structured summary of a paper."""
        return self.ask_paper(
            paper,
            "Provide a comprehensive summary of this paper including: "
            "1) Main research question/hypothesis, "
            "2) Key methodology, "
            "3) Main findings/results, "
            "4) Limitations, "
            "5) Key contributions to the field."
        )

    def explain_methodology(self, paper: Dict) -> Dict:
        """Explain the paper's methodology in simple terms."""
        return self.ask_paper(
            paper,
            "Explain the methodology of this paper in detail. "
            "What approach did they use? What data? What analysis methods? "
            "Are there any methodological limitations?"
        )

    def identify_limitations(self, paper: Dict) -> Dict:
        """Identify limitations and potential issues."""
        return self.ask_paper(
            paper,
            "What are the limitations of this study? "
            "Consider: sample size, methodology, generalizability, "
            "potential biases, and what the authors themselves acknowledge."
        )

    def _select_relevant_chunks(self, chunks: List[str], question: str) -> str:
        """Select chunks most relevant to the question (simple keyword matching)."""
        question_words = set(question.lower().split())
        scored_chunks = []

        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            scored_chunks.append((overlap, chunk))

        # Sort by relevance and take top 3
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        relevant = [chunk for _, chunk in scored_chunks[:3]]

        return '\n\n---\n\n'.join(relevant)


class DeepReviewAgent(Tier3Agent):
    """
    Multi-pass iterative literature review.

    Replicates SciSpace's Deep Review feature:
    - Pass 1: Quick scan - identify key themes from all papers
    - Pass 2: Deep dive - analyze each theme with relevant papers
    - Pass 3: Cross-cutting synthesis - find connections between themes
    - Pass 4: Evidence assessment - rate strength of each finding
    - Pass 5: Knowledge gap mapping - what's missing?

    Each pass builds on the previous, creating progressively deeper understanding.
    """

    def __init__(self):
        super().__init__(
            name="DeepReviewAgent",
            description="Multi-pass iterative deep literature review"
        )

    def execute(self, input_data: Dict) -> Dict:
        """
        Run full deep review pipeline.

        Input:
            papers: List[Dict] - Papers to review
            topic: str - Research topic
            depth: int - Number of passes (1-5, default 3)
        """
        papers = input_data.get('papers', [])
        topic = input_data.get('topic', '')
        depth = min(input_data.get('depth', 3), 5)

        if len(papers) < 3:
            return {'error': 'Need at least 3 papers for deep review'}

        return self.deep_review(papers, topic, depth)

    def deep_review(self, papers: List[Dict], topic: str, depth: int = 3) -> Dict:
        """
        Multi-pass deep review pipeline.
        Each pass builds on the previous one.
        """
        results = {
            'topic': topic,
            'paper_count': len(papers),
            'depth': depth,
            'passes': []
        }

        # Pass 1: Quick scan - identify themes
        print(f"    Deep Review Pass 1/{depth}: Theme identification...")
        themes = self._pass_1_theme_scan(papers, topic)
        results['passes'].append({
            'pass': 1,
            'name': 'Theme Identification',
            'result': themes
        })
        results['themes'] = themes.get('themes', [])

        if depth < 2:
            return results

        # Pass 2: Deep dive per theme
        print(f"    Deep Review Pass 2/{depth}: Theme deep dive...")
        deep_themes = self._pass_2_deep_dive(papers, topic, themes)
        results['passes'].append({
            'pass': 2,
            'name': 'Theme Deep Dive',
            'result': deep_themes
        })
        results['deep_themes'] = deep_themes.get('theme_analyses', [])

        if depth < 3:
            return results

        # Pass 3: Cross-cutting synthesis
        print(f"    Deep Review Pass 3/{depth}: Cross-cutting synthesis...")
        cross_cutting = self._pass_3_cross_synthesis(papers, topic, deep_themes)
        results['passes'].append({
            'pass': 3,
            'name': 'Cross-Cutting Synthesis',
            'result': cross_cutting
        })
        results['cross_cutting'] = cross_cutting

        if depth < 4:
            return results

        # Pass 4: Evidence strength assessment
        print(f"    Deep Review Pass 4/{depth}: Evidence assessment...")
        evidence = self._pass_4_evidence_assessment(papers, topic, deep_themes, cross_cutting)
        results['passes'].append({
            'pass': 4,
            'name': 'Evidence Assessment',
            'result': evidence
        })
        results['evidence_map'] = evidence

        if depth < 5:
            return results

        # Pass 5: Knowledge gap mapping
        print(f"    Deep Review Pass 5/{depth}: Gap mapping...")
        gaps = self._pass_5_gap_mapping(papers, topic, deep_themes, cross_cutting, evidence)
        results['passes'].append({
            'pass': 5,
            'name': 'Knowledge Gap Mapping',
            'result': gaps
        })
        results['knowledge_gaps'] = gaps

        return results

    def _pass_1_theme_scan(self, papers: List[Dict], topic: str) -> Dict:
        """Pass 1: Quick scan to identify major themes."""

        papers_text = ""
        for i, p in enumerate(papers[:30]):
            title = p.get('title', '')[:80]
            abstract = p.get('abstract', '')[:150]
            year = p.get('year', 'N/A')
            papers_text += f"[{i+1}] ({year}) {title}\n    {abstract}...\n"

        system_prompt = """You are a systematic review expert performing a literature scan.
Identify the major research themes across these papers."""

        user_prompt = f"""Topic: {topic}

## Papers ({len(papers)} total, showing first 30)
{papers_text}

## Task
Identify 4-8 major research themes across these papers.

## Output JSON
{{
  "themes": [
    {{
      "id": 1,
      "name": "Theme name",
      "description": "What this theme covers",
      "paper_count": 12,
      "key_papers": [1, 5, 12],
      "maturity": "EMERGING|GROWING|MATURE|DECLINING"
    }}
  ],
  "coverage_assessment": "How well do these papers cover the topic?"
}}"""

        schema = {"themes": "array", "coverage_assessment": "string"}

        try:
            return self._call_llm(system_prompt, user_prompt, schema)
        except Exception as e:
            return {"themes": [], "error": str(e)}

    def _pass_2_deep_dive(self, papers: List[Dict], topic: str,
                           themes: Dict) -> Dict:
        """Pass 2: Deep analysis of each theme with relevant papers."""

        theme_list = themes.get('themes', [])
        if not theme_list:
            return {"theme_analyses": []}

        # Build a summary of all themes with their key papers
        theme_summary = ""
        for theme in theme_list[:6]:
            theme_summary += f"\n### Theme: {theme.get('name', '')}\n"
            theme_summary += f"Description: {theme.get('description', '')}\n"
            key_ids = theme.get('key_papers', [])
            for pid in key_ids[:5]:
                if isinstance(pid, int) and pid <= len(papers):
                    p = papers[pid - 1]
                    theme_summary += f"  - [{pid}] {p.get('title', '')[:60]} ({p.get('year', '')})\n"
                    theme_summary += f"    Abstract: {p.get('abstract', '')[:200]}...\n"

        system_prompt = """You are performing a deep dive analysis of research themes.
For each theme, identify key findings, methods used, and open questions."""

        user_prompt = f"""Topic: {topic}
Total papers: {len(papers)}

## Themes and Key Papers
{theme_summary}

## Task
For each theme, provide deep analysis:
- Key findings across papers
- Dominant methodologies
- Points of agreement
- Open questions within the theme

## Output JSON
{{
  "theme_analyses": [
    {{
      "theme_name": "Name",
      "key_findings": ["Finding 1", "Finding 2"],
      "methodologies": ["Method 1", "Method 2"],
      "consensus_points": ["Point 1"],
      "open_questions": ["Question 1"],
      "evidence_strength": "STRONG|MEDIUM|WEAK"
    }}
  ]
}}"""

        schema = {"theme_analyses": "array"}

        try:
            return self._call_llm(system_prompt, user_prompt, schema)
        except Exception as e:
            return {"theme_analyses": [], "error": str(e)}

    def _pass_3_cross_synthesis(self, papers: List[Dict], topic: str,
                                 deep_themes: Dict) -> Dict:
        """Pass 3: Find connections between themes."""

        analyses = deep_themes.get('theme_analyses', [])
        if len(analyses) < 2:
            return {"connections": []}

        themes_context = ""
        for a in analyses:
            themes_context += f"\n### {a.get('theme_name', '')}\n"
            findings = a.get('key_findings', [])
            for f in findings[:3]:
                themes_context += f"  - {f}\n"

        system_prompt = """You are synthesizing across research themes.
Find connections, contradictions, and emergent insights that span multiple themes."""

        user_prompt = f"""Topic: {topic}

## Theme Analyses
{themes_context}

## Task
Identify cross-cutting patterns:
1. Connections between themes (how does Theme A relate to Theme B?)
2. Contradictions (where do themes disagree?)
3. Emergent insights (what becomes clear only when viewing themes together?)
4. Synthesis statement (the big picture)

## Output JSON
{{
  "connections": [
    {{
      "themes": ["Theme A", "Theme B"],
      "relationship": "How they connect",
      "type": "REINFORCING|CONTRADICTING|COMPLEMENTARY"
    }}
  ],
  "emergent_insights": [
    "Insight that emerges from viewing themes together"
  ],
  "synthesis_statement": "Overall big-picture understanding",
  "research_narrative": "The story these papers tell together"
}}"""

        schema = {
            "connections": "array",
            "emergent_insights": "array",
            "synthesis_statement": "string",
            "research_narrative": "string"
        }

        try:
            return self._call_llm(system_prompt, user_prompt, schema)
        except Exception as e:
            return {"connections": [], "error": str(e)}

    def _pass_4_evidence_assessment(self, papers: List[Dict], topic: str,
                                      deep_themes: Dict, cross_cutting: Dict) -> Dict:
        """Pass 4: Assess evidence strength for each key finding."""

        findings_text = ""
        for a in deep_themes.get('theme_analyses', [])[:6]:
            findings_text += f"\n### {a.get('theme_name', '')}\n"
            for f in a.get('key_findings', [])[:3]:
                findings_text += f"  - {f}\n"

        # Add cross-cutting insights
        for insight in cross_cutting.get('emergent_insights', [])[:5]:
            findings_text += f"\n  Cross-cutting: {insight}\n"

        system_prompt = """You are an evidence quality assessor.
Rate the strength of evidence for each key finding based on:
- Number of supporting papers
- Methodology quality
- Consistency across studies
- Potential biases"""

        user_prompt = f"""Topic: {topic}
Total papers: {len(papers)}

## Key Findings to Assess
{findings_text}

## Task
For each finding, assess evidence strength.

## Output JSON
{{
  "evidence_ratings": [
    {{
      "finding": "The finding",
      "strength": "VERY_STRONG|STRONG|MODERATE|WEAK|INSUFFICIENT",
      "supporting_evidence": "What supports it",
      "concerns": "Any methodological concerns",
      "confidence_level": 85
    }}
  ],
  "overall_evidence_quality": "Assessment of the field's evidence base"
}}"""

        schema = {"evidence_ratings": "array", "overall_evidence_quality": "string"}

        try:
            return self._call_llm(system_prompt, user_prompt, schema)
        except Exception as e:
            return {"evidence_ratings": [], "error": str(e)}

    def _pass_5_gap_mapping(self, papers: List[Dict], topic: str,
                              deep_themes: Dict, cross_cutting: Dict,
                              evidence: Dict) -> Dict:
        """Pass 5: Map knowledge gaps and future research directions."""

        # Compile context from all previous passes
        open_questions = []
        for a in deep_themes.get('theme_analyses', []):
            open_questions.extend(a.get('open_questions', []))

        weak_evidence = []
        for e in evidence.get('evidence_ratings', []):
            if e.get('strength') in ('WEAK', 'INSUFFICIENT'):
                weak_evidence.append(e.get('finding', ''))

        system_prompt = """You are a research gap analyst.
Based on a thorough multi-pass review, identify what's MISSING from the literature."""

        user_prompt = f"""Topic: {topic}
Papers analyzed: {len(papers)}

## Open Questions from Themes
{json.dumps(open_questions[:10], indent=2)}

## Weakly Supported Findings
{json.dumps(weak_evidence[:5], indent=2)}

## Synthesis Statement
{cross_cutting.get('synthesis_statement', 'N/A')}

## Task
Map the knowledge gaps:
1. Methodological gaps (what approaches are missing?)
2. Topical gaps (what sub-topics are under-researched?)
3. Population/context gaps (what settings haven't been studied?)
4. Temporal gaps (what time periods need more research?)
5. Recommended research agenda (prioritized list)

## Output JSON
{{
  "methodological_gaps": ["Gap 1", "Gap 2"],
  "topical_gaps": ["Gap 1", "Gap 2"],
  "context_gaps": ["Gap 1"],
  "temporal_gaps": ["Gap 1"],
  "research_agenda": [
    {{
      "priority": 1,
      "question": "What needs to be studied",
      "rationale": "Why it matters",
      "suggested_approach": "How to study it",
      "impact": "HIGH|MEDIUM|LOW"
    }}
  ],
  "overall_maturity": "NASCENT|DEVELOPING|ESTABLISHED|MATURE"
}}"""

        schema = {
            "methodological_gaps": "array",
            "topical_gaps": "array",
            "research_agenda": "array",
            "overall_maturity": "string"
        }

        try:
            return self._call_llm(system_prompt, user_prompt, schema)
        except Exception as e:
            return {"research_agenda": [], "error": str(e)}
