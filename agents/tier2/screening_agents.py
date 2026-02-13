"""
Tier 2 Screening & Quality Agents - Domain LLMs for judgment tasks.
Uses Gemini as primary, DeepSeek as fallback.

Agents:
- ScreeningAgent: LLM-based inclusion/exclusion screening
- QualityTierAgent: Evidence quality assessment with justification
- ClusterThemingAgent: Semantic cluster labeling and theme extraction
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import Tier2Agent


class ScreeningAgent(Tier2Agent):
    """
    LLM-based title/abstract screening for inclusion/exclusion.
    Uses Gemini (primary) for fast semantic understanding.

    Applies inclusion criteria:
      - Topic relevance
      - Study type match
      - Language filter
      - Date appropriateness
    """

    def __init__(self, min_inclusion_score: float = 6.0):
        super().__init__(
            name="Screener",
            description="LLM-based inclusion/exclusion screening"
        )
        self.min_inclusion_score = min_inclusion_score
        self.screening_log: List[Dict] = []

    def execute(self, input_data: Dict) -> Dict:
        """Screen papers for inclusion/exclusion."""
        papers = input_data.get('papers', [])
        query = input_data.get('query', '')
        criteria = input_data.get('inclusion_criteria', {})

        included, excluded = self.screen_batch(papers, query, criteria)

        return {
            'included': included,
            'excluded': excluded,
            'inclusion_rate': len(included) / max(len(papers), 1),
            'total_screened': len(papers)
        }

    def screen_batch(self, papers: List[Dict], query: str,
                     criteria: Dict = None, batch_size: int = 15) -> tuple:
        """Screen papers in batches."""
        all_included = []
        all_excluded = []

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            inc, exc = self._screen_one_batch(batch, query, criteria)
            all_included.extend(inc)
            all_excluded.extend(exc)

        return all_included, all_excluded

    def _screen_one_batch(self, papers: List[Dict], query: str,
                          criteria: Dict = None) -> tuple:
        """Screen a single batch of papers."""
        papers_text = ""
        for i, p in enumerate(papers):
            title = p.get('title', 'Unknown')[:120]
            abstract = (p.get('abstract') or '')[:250]
            year = p.get('year', 'N/A')
            papers_text += f"\n[{i+1}] {title} ({year})\n    Abstract: {abstract}...\n"

        criteria_text = json.dumps(criteria, indent=2) if criteria else "Standard academic relevance"

        system_prompt = """You are a systematic review screener.
Apply strict inclusion/exclusion criteria to each paper.
Be thorough but fair - include borderline papers for now."""

        user_prompt = f"""Research Query: {query}

## Inclusion Criteria
{criteria_text}

## Papers to Screen
{papers_text}

## Output JSON
{{
  "decisions": [
    {{
      "paper_index": 1,
      "decision": "include|exclude",
      "score": 0-10,
      "reason": "Brief justification",
      "exclusion_category": "off_topic|wrong_type|low_quality|no_abstract|other"
    }}
  ]
}}

Score guide: 0-3=exclude, 4-5=borderline (exclude), 6-7=relevant, 8-10=highly relevant"""

        schema = {"decisions": "array"}

        try:
            response = self._call_llm(system_prompt, user_prompt, schema)

            decisions = {d['paper_index']: d for d in response.get('decisions', [])}

            included = []
            excluded = []
            for i, paper in enumerate(papers):
                decision = decisions.get(i + 1, {'score': 5, 'decision': 'include'})
                score = float(decision.get('score', 5))
                paper['screening_score'] = score
                paper['screening_reason'] = decision.get('reason', '')

                if score >= self.min_inclusion_score or decision.get('decision') == 'include':
                    included.append(paper)
                else:
                    paper['exclusion_category'] = decision.get('exclusion_category', 'other')
                    excluded.append(paper)

            self.screening_log.append({
                'batch_size': len(papers),
                'included': len(included),
                'excluded': len(excluded)
            })

            return included, excluded

        except Exception as e:
            print(f"  ⚠️ LLM screening failed: {e} - applying heuristic screening")
            # Heuristic fallback: exclude papers with no abstract and no title
            included = []
            excluded = []
            query_terms = set(query.lower().split())
            for paper in papers:
                title = (paper.get('title') or '').lower()
                abstract = (paper.get('abstract') or '').lower()
                has_content = bool(title and len(title) > 10)
                # Check if any query term appears in title or abstract
                relevance = any(t in title or t in abstract for t in query_terms) if query_terms else True
                if has_content and (relevance or abstract):
                    paper['screening_score'] = 5.0
                    paper['screening_reason'] = 'heuristic pass (LLM unavailable)'
                    included.append(paper)
                else:
                    paper['screening_score'] = 2.0
                    paper['screening_reason'] = 'heuristic exclude: no content or no relevance'
                    paper['exclusion_category'] = 'no_abstract' if not abstract else 'off_topic'
                    excluded.append(paper)

            self.screening_log.append({
                'batch_size': len(papers),
                'included': len(included),
                'excluded': len(excluded),
                'method': 'heuristic_fallback'
            })
            return included, excluded


class QualityTierAgent(Tier2Agent):
    """
    LLM-assisted quality assessment with evidence-based justification.
    Uses Gemini (primary) for consistent evaluation.

    Tiers:
      A - High quality: top venue, high citations, robust methodology
      B - Medium quality: decent venue, some citations
      C - Acceptable: meets minimum threshold
    """

    def __init__(self):
        super().__init__(
            name="QualityAssessor",
            description="Evidence quality assessment and tier assignment"
        )

    def execute(self, input_data: Dict) -> Dict:
        """Assess quality of papers."""
        papers = input_data.get('papers', [])
        domain = input_data.get('domain', '')

        assessed = self.assess_batch(papers, domain)

        tier_counts = {'A': 0, 'B': 0, 'C': 0}
        for p in assessed:
            tier = p.get('quality_tier', 'C')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            'papers': assessed,
            'tier_distribution': tier_counts,
            'total_assessed': len(assessed)
        }

    def assess_batch(self, papers: List[Dict], domain: str = '',
                     batch_size: int = 15) -> List[Dict]:
        """Assess quality in batches."""
        all_assessed = []

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            assessed = self._assess_one_batch(batch, domain)
            all_assessed.extend(assessed)

        return all_assessed

    def _assess_one_batch(self, papers: List[Dict], domain: str) -> List[Dict]:
        """Assess a single batch."""
        papers_text = ""
        for i, p in enumerate(papers):
            title = p.get('title', 'Unknown')[:100]
            venue = p.get('venue', 'Unknown')
            year = p.get('year', 'N/A')
            cites = p.get('citation_count', 0)
            has_doi = 'Yes' if p.get('doi') else 'No'
            abstract_len = len(p.get('abstract', ''))

            papers_text += (
                f"\n[{i+1}] {title}\n"
                f"    Venue: {venue} | Year: {year} | Citations: {cites}\n"
                f"    DOI: {has_doi} | Abstract length: {abstract_len} chars\n"
            )

        system_prompt = """You are a research quality assessor.
Evaluate papers using evidence-based criteria. Be consistent."""

        user_prompt = f"""Domain: {domain or 'General academic'}

## Papers to Assess
{papers_text}

## Quality Criteria
- Tier A: Top venue OR 50+ citations OR clearly robust methodology
- Tier B: Decent venue, some citations, adequate methodology
- Tier C: Meets minimum threshold, limited evidence of quality

## Output JSON
{{
  "assessments": [
    {{
      "paper_index": 1,
      "tier": "A|B|C",
      "quality_score": 0-100,
      "strengths": ["strength1"],
      "weaknesses": ["weakness1"],
      "justification": "Brief quality justification"
    }}
  ]
}}"""

        schema = {"assessments": "array"}

        try:
            response = self._call_llm(system_prompt, user_prompt, schema)

            assessments = {a['paper_index']: a for a in response.get('assessments', [])}

            for i, paper in enumerate(papers):
                assessment = assessments.get(i + 1, {})
                paper['quality_tier'] = assessment.get('tier', 'C')
                paper['quality_score'] = assessment.get('quality_score', 50)
                paper['quality_justification'] = assessment.get('justification', '')
                paper['quality_strengths'] = assessment.get('strengths', [])
                paper['quality_weaknesses'] = assessment.get('weaknesses', [])

            return papers

        except Exception as e:
            print(f"  Quality assessment failed: {e} - assigning default tiers")
            for paper in papers:
                paper['quality_tier'] = 'B'
                paper['quality_score'] = 50
            return papers


class ClusterThemingAgent(Tier2Agent):
    """
    Generates semantic labels and theme descriptions for paper clusters.
    Uses Gemini (primary) for natural language understanding.
    """

    def __init__(self):
        super().__init__(
            name="ClusterThemer",
            description="Semantic cluster labeling and theme extraction"
        )

    def execute(self, input_data: Dict) -> Dict:
        """Label clusters with semantic themes."""
        clusters = input_data.get('clusters', [])
        query = input_data.get('query', '')

        labeled = self.label_clusters(clusters, query)

        return {
            'labeled_clusters': labeled,
            'total_clusters': len(labeled)
        }

    def label_clusters(self, clusters: List[Dict], query: str) -> List[Dict]:
        """Generate labels for all clusters."""
        if not clusters:
            return []

        # Build cluster descriptions from paper titles
        cluster_text = ""
        for i, cluster in enumerate(clusters):
            papers = cluster.get('papers', [])
            label = cluster.get('label', f'Cluster {i}')
            size = cluster.get('size', len(papers))

            # Collect top titles
            titles = []
            for p in papers[:8]:
                if isinstance(p, dict):
                    titles.append(p.get('title', '')[:80])
                elif isinstance(p, str):
                    titles.append(p[:80])

            cluster_text += f"\n## Cluster {i+1} (current label: '{label}', {size} papers)\n"
            cluster_text += "Papers:\n"
            for t in titles:
                cluster_text += f"  - {t}\n"

        system_prompt = """You are a research taxonomy expert.
Generate precise, descriptive labels for paper clusters.
Labels should be 3-8 words, specific to the research content."""

        user_prompt = f"""Research Query: {query}

## Clusters to Label
{cluster_text}

## Output JSON
{{
  "cluster_labels": [
    {{
      "cluster_index": 1,
      "label": "Descriptive 3-8 word label",
      "theme_description": "2-3 sentence description of what this cluster covers",
      "key_concepts": ["concept1", "concept2", "concept3"],
      "relationship_to_query": "How this cluster relates to the main research question"
    }}
  ]
}}

Make labels specific, not generic (e.g., 'DeFi Liquidation Cascade Mechanisms' not 'Finance Studies')."""

        schema = {"cluster_labels": "array"}

        try:
            response = self._call_llm(system_prompt, user_prompt, schema)

            labels = {l['cluster_index']: l for l in response.get('cluster_labels', [])}

            for i, cluster in enumerate(clusters):
                label_data = labels.get(i + 1, {})
                cluster['themed_label'] = label_data.get('label', cluster.get('label', f'Cluster {i+1}'))
                cluster['theme_description'] = label_data.get('theme_description', '')
                cluster['key_concepts'] = label_data.get('key_concepts', [])
                cluster['relationship_to_query'] = label_data.get('relationship_to_query', '')

            return clusters

        except Exception as e:
            print(f"  Cluster theming failed: {e} - keeping original labels")
            return clusters
