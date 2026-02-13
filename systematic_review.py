"""
Systematic Review Protocol - The upgraded orchestrator.
Combines all 10 upgrades for 1000+ paper systematic reviews.

Upgrades integrated:
  #1  Async parallel queries (core/async_engine.py)
  #2  Pagination/bulk harvesting (sources/async_sources.py)
  #3  Map-Reduce synthesis (synthesis/map_reduce.py)
  #4  Semantic clustering (synthesis/clustering.py)
  #5  Caching layer (core/cache_layer.py)
  #6  Adaptive search rounds (built into this orchestrator)
  #7  Citation network crawling (core/citation_crawler.py)
  #8  Full-text analysis (sources/async_sources.py - Unpaywall)
  #9  Progress streaming (core/progress_streamer.py)
  #10 PRISMA flow tracking (core/prisma_tracker.py)
"""

import asyncio
import aiohttp
import json
import time
import re
from typing import List, Dict, Optional, Callable
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sources.async_sources import (
    AsyncSemanticScholar, AsyncOpenAlex, AsyncArXiv,
    AsyncPubMed, AsyncCrossRef, AsyncSSRN, AsyncUnpaywall
)
from core.cache_layer import ResearchCache
from core.citation_crawler import CitationCrawler
from core.prisma_tracker import PRISMATracker, ExclusionReason
from core.progress_streamer import ProgressStreamer, ResearchPhase
from synthesis.map_reduce import MapReduceSynthesizer
from synthesis.clustering import SemanticClusterer
from validation.deduplicator import Deduplicator
from validation.quality_scorer import QualityScorer
from core.llm_client import LLMClient
from agents.tier3.council import ResearchStrategist
from agents.tier2.specialists import GapDetectionAgent, QueryRefinementAgent, RelevanceFilterAgent
from agents.tier1.deduplication_agent import DeduplicationAgent
from agents.tier1.prisma_agent import PRISMAComplianceAgent
from agents.tier2.screening_agents import ScreeningAgent, QualityTierAgent, ClusterThemingAgent
from agents.tier3.strategic_agents import (
    AdaptiveStoppingAgent, SynthesisCoordinatorAgent,
    ReportComposerAgent, CitationCrawlStrategyAgent
)
from synthesis.report_generator import generate_synthesis_report


class SystematicReviewProtocol:
    """
    Production-grade systematic review engine.
    Processes 1000+ papers with PRISMA methodology.

    Pipeline:
    1. Query decomposition (Tier 3 LLM)
    2. Parallel database search (6 sources, async, paginated)
    3. Adaptive refinement (smart stopping, not fixed 4 rounds)
    4. Citation network crawling (snowball sampling)
    5. Deduplication + PRISMA tracking
    6. Relevance screening (LLM-based)
    7. Quality assessment + tier assignment
    8. Semantic clustering
    9. Map-Reduce synthesis (handles 1000+ papers)
    10. Deep analysis + report generation
    """

    def __init__(
        self,
        target_papers: int = 1000,
        max_search_rounds: int = 10,
        enable_cache: bool = True,
        enable_citation_crawl: bool = True,
        enable_clustering: bool = True,
        verbose: bool = True,
        llm_provider: str = "ollama",
        ollama_model: str = None
    ):
        self.target_papers = target_papers
        self.max_search_rounds = max_search_rounds
        self.enable_cache = enable_cache
        self.enable_citation_crawl = enable_citation_crawl
        self.enable_clustering = enable_clustering
        self.verbose = verbose

        # Initialize all components
        self.progress = ProgressStreamer(verbose=verbose)
        self.prisma = PRISMATracker()
        self.cache = ResearchCache() if enable_cache else None
        self.deduplicator = Deduplicator()
        self.quality_scorer = QualityScorer()
        self.llm = LLMClient(
            primary=llm_provider,
            fallback="gemini" if llm_provider != "gemini" else "deepseek",
            ollama_model=ollama_model
        )

        # Agents - Tier 3 (Strategic)
        self.strategist = ResearchStrategist()
        self.adaptive_stopper = AdaptiveStoppingAgent(max_rounds=max_search_rounds)
        self.synthesis_coordinator = SynthesisCoordinatorAgent()
        self.report_composer = ReportComposerAgent()
        self.citation_strategist = CitationCrawlStrategyAgent()

        # Agents - Tier 2 (Specialist)
        self.gap_detector = GapDetectionAgent()
        self.query_refiner = QueryRefinementAgent()
        self.relevance_filter = RelevanceFilterAgent()
        self.screener = ScreeningAgent()
        self.quality_assessor = QualityTierAgent()
        self.cluster_themer = ClusterThemingAgent()

        # Agents - Tier 1 (Scripted)
        self.dedup_agent = DeduplicationAgent()
        self.prisma_agent = PRISMAComplianceAgent()

        # Async sources
        self.sources = {
            'semantic_scholar': AsyncSemanticScholar(),
            'openalex': AsyncOpenAlex(),
            'arxiv': AsyncArXiv(),
            'pubmed': AsyncPubMed(),
            'crossref': AsyncCrossRef(),
            'ssrn': AsyncSSRN()
        }
        self.unpaywall = AsyncUnpaywall()

        # Synthesis engines
        self.map_reduce = MapReduceSynthesizer(llm=self.llm)
        self.clusterer = None  # Lazy init (needs sklearn)

        print("=" * 80)
        print("  SYSTEMATIC REVIEW ENGINE v2.0")
        print(f"  Target: {target_papers} papers | Max rounds: {max_search_rounds}")
        print(f"  Cache: {'ON' if enable_cache else 'OFF'} | "
              f"Citation crawl: {'ON' if enable_citation_crawl else 'OFF'} | "
              f"Clustering: {'ON' if enable_clustering else 'OFF'}")
        print("=" * 80)

    def execute(self, query: str, domain: str = None) -> Dict:
        """Execute the full systematic review. Sync wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._execute_async(query, domain))
            else:
                return loop.run_until_complete(self._execute_async(query, domain))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._execute_async(query, domain))
            finally:
                loop.close()

    async def _execute_async(self, query: str, domain: str = None) -> Dict:
        """Main async execution pipeline."""
        start_time = time.monotonic()

        self.progress.update(ResearchPhase.INITIALIZATION, "Starting systematic review")

        # =====================================================
        # PHASE 1: Query Decomposition (Tier 3)
        # =====================================================
        self.progress.update(ResearchPhase.QUERY_DECOMPOSITION,
                            "Decomposing research query", 0.0)

        decomposition = self.strategist.decompose_query(query)
        dimensions = decomposition.get('dimensions', {})
        baseline_query = decomposition.get('baseline_query', query)
        sub_questions = decomposition.get('sub_questions', [])

        self.progress.update(ResearchPhase.QUERY_DECOMPOSITION,
                            f"Decomposed into {len(sub_questions)} sub-questions",
                            1.0)

        # Build search queries from decomposition
        search_queries = [baseline_query]
        for sq in sub_questions[:5]:
            q = sq.get('question', '')
            if q and len(q) > 10:
                search_queries.append(q)

        # =====================================================
        # PHASE 2: Parallel Baseline Search (All 6 sources)
        # =====================================================
        self.progress.update(ResearchPhase.BASELINE_SEARCH,
                            "Searching 6 databases in parallel", 0.0)

        all_papers = []

        async with aiohttp.ClientSession() as session:
            # Determine papers per source per query
            papers_per_source = min(200, self.target_papers // (len(self.sources) * len(search_queries)))

            for qi, sq in enumerate(search_queries):
                # Check cache first
                if self.cache:
                    cached = self.cache.get_api_response('all', sq, papers_per_source)
                    if cached:
                        all_papers.extend(cached)
                        self.progress.update(ResearchPhase.BASELINE_SEARCH,
                                            f"Cache hit for query {qi+1}",
                                            (qi + 1) / len(search_queries),
                                            len(all_papers))
                        continue

                # Parallel search across all sources
                tasks = []
                source_names = []
                for name, source in self.sources.items():
                    tasks.append(source.search(sq, papers_per_source, 0, session))
                    source_names.append(name)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for name, result in zip(source_names, results):
                    if isinstance(result, list) and result:
                        all_papers.extend(result)
                        self.prisma.add_identified(result, name)

                        if self.cache:
                            self.cache.set_api_response(name, sq, result, papers_per_source)

                self.progress.update(ResearchPhase.BASELINE_SEARCH,
                                    f"Query {qi+1}/{len(search_queries)} complete",
                                    (qi + 1) / len(search_queries),
                                    len(all_papers))

            # =====================================================
            # PHASE 3: Pagination for more results
            # =====================================================
            if len(all_papers) < self.target_papers:
                self.progress.update(ResearchPhase.PAGINATION,
                                    "Fetching additional pages", 0.0)

                remaining = self.target_papers - len(all_papers)
                # Prioritize highest-yield sources
                priority_sources = ['semantic_scholar', 'openalex', 'crossref']

                for si, src_name in enumerate(priority_sources):
                    if len(all_papers) >= self.target_papers:
                        break

                    src = self.sources[src_name]
                    per_source = remaining // len(priority_sources)

                    paginated = await src.search_paginated(
                        baseline_query, target=per_source, session=session
                    )

                    if paginated:
                        all_papers.extend(paginated)
                        self.prisma.add_identified(paginated, src_name)

                    self.progress.update(ResearchPhase.PAGINATION,
                                        f"Paginated {src_name}",
                                        (si + 1) / len(priority_sources),
                                        len(all_papers))

            # =====================================================
            # PHASE 4: Citation Network Crawling (CitationCrawlStrategyAgent)
            # =====================================================
            if self.enable_citation_crawl and len(all_papers) > 10:
                self.progress.update(ResearchPhase.CITATION_CRAWLING,
                                    "Planning citation crawl strategy", 0.0)

                # Get crawl strategy from Tier 3 agent
                target_additional = max(0, self.target_papers - len(all_papers))
                crawl_strategy = self.citation_strategist.execute({
                    'papers': all_papers,
                    'gaps': [],
                    'target_additional': min(500, target_additional)
                })

                crawl_depth = crawl_strategy.get('crawl_depth', 1)
                refs_per = crawl_strategy.get('refs_per_paper', 30)
                cites_per = crawl_strategy.get('cites_per_paper', 20)
                seed_papers = crawl_strategy.get('seed_papers', all_papers[:50])
                top_n_seeds = min(len(seed_papers), 20)

                self.progress.update(ResearchPhase.CITATION_CRAWLING,
                                    f"Crawling citations ({top_n_seeds} seeds, "
                                    f"depth={crawl_depth})", 0.1)

                crawler = CitationCrawler(max_papers=min(500, target_additional))

                async def citation_progress(phase, current, total, data):
                    self.progress.update(ResearchPhase.CITATION_CRAWLING,
                                        f"Snowball {phase}: {current}/{total}",
                                        current / max(total, 1),
                                        len(all_papers) + data.get('new_papers', 0))

                snowball_result = await crawler.snowball_search(
                    seed_papers=seed_papers[:50],
                    depth=crawl_depth,
                    top_n_seeds=top_n_seeds,
                    refs_per_paper=refs_per,
                    cites_per_paper=cites_per,
                    progress_callback=citation_progress,
                    session=session
                )

                new_from_snowball = snowball_result.get('new_papers', [])
                if new_from_snowball:
                    all_papers.extend(new_from_snowball)
                    self.prisma.add_identified(new_from_snowball, 'citation_crawl')

                self.progress.update(ResearchPhase.CITATION_CRAWLING,
                                    f"Snowball complete: +{len(new_from_snowball)} papers",
                                    1.0, len(all_papers))

        # =====================================================
        # PHASE 5: Deduplication (DeduplicationAgent - Tier 1)
        # =====================================================
        self.progress.update(ResearchPhase.DEDUPLICATION,
                            f"Deduplicating {len(all_papers)} papers", 0.0)

        dedup_result = self.dedup_agent.execute({'papers': all_papers})
        papers_dicts = dedup_result['papers']
        duplicates_removed = dedup_result['duplicates_removed']

        # Copy over extra fields from original dicts
        paper_id_map = {}
        for p in all_papers:
            pid = p.get('paper_id', '')
            if pid and pid not in paper_id_map:
                paper_id_map[pid] = p
        for pd in papers_dicts:
            orig = paper_id_map.get(pd.get('paper_id', ''), {})
            pd['tldr'] = orig.get('tldr', '')
            pd['discovery_method'] = orig.get('discovery_method', 'keyword_search')

        self.progress.update(ResearchPhase.DEDUPLICATION,
                            f"Removed {duplicates_removed} duplicates, {len(papers_dicts)} unique",
                            1.0, len(papers_dicts))

        # =====================================================
        # PHASE 6: Adaptive Search Rounds (AdaptiveStoppingAgent)
        # =====================================================
        round_num = 0
        discovery_rates = []
        source_saturation = {name: 0.0 for name in self.sources}

        while (len(papers_dicts) < self.target_papers and
               round_num < self.max_search_rounds):
            round_num += 1

            # Gap detection (Tier 2 agent)
            gap_result = self.gap_detector.execute({
                'papers': papers_dicts[:100],
                'dimensions': dimensions,
                'round': round_num
            })

            gaps = gap_result.get('gaps', [])
            coverage = gap_result.get('coverage_matrix', {})
            coverage_score = coverage.get('coverage_score', 0)

            # Adaptive stopping decision (Tier 3 agent)
            stop_result = self.adaptive_stopper.execute({
                'coverage_score': coverage_score,
                'gaps': gaps,
                'round_number': round_num,
                'papers_found': len(papers_dicts),
                'target_papers': self.target_papers,
                'discovery_rates': discovery_rates,
                'source_saturation': source_saturation,
            })

            if not stop_result['should_continue']:
                self.progress.update(ResearchPhase.RELEVANCE_FILTERING,
                                    f"Stopping: {stop_result['reason']}", 1.0,
                                    len(papers_dicts))
                break

            # Generate refinement queries (Tier 2 agent)
            refine_result = self.query_refiner.execute({'gaps': gaps, 'round': round_num})
            queries = refine_result.get('queries', [])

            if not queries:
                break

            # Execute refinement queries
            prev_count = len(papers_dicts)
            async with aiohttp.ClientSession() as session:
                for q_info in queries[:5]:
                    q_text = q_info.get('query', '')
                    dbs = q_info.get('databases', ['semantic_scholar', 'openalex'])

                    for db_name in dbs:
                        if db_name in self.sources:
                            new = await self.sources[db_name].search(
                                q_text, 50, 0, session
                            )
                            if new:
                                papers_dicts.extend(new)
                                self.prisma.add_identified(new, db_name)

            # Deduplicate again (Tier 1 agent)
            dedup_result = self.dedup_agent.execute({'papers': papers_dicts})
            new_unique = dedup_result['papers']
            discovery_rate = (len(new_unique) - prev_count) / max(prev_count, 1)
            discovery_rates.append(discovery_rate)
            papers_dicts = new_unique

            self.progress.update(ResearchPhase.RELEVANCE_FILTERING,
                                f"Round {round_num}: +{len(papers_dicts) - prev_count} papers, "
                                f"coverage {coverage_score:.0f}%",
                                round_num / self.max_search_rounds,
                                len(papers_dicts))

        # =====================================================
        # PHASE 7: Screening + Quality Assessment
        # =====================================================
        self.progress.update(ResearchPhase.SCREENING,
                            f"Screening {len(papers_dicts)} papers", 0.0)

        # LLM-based screening (ScreeningAgent - Tier 2)
        screen_result = self.screener.execute({
            'papers': papers_dicts,
            'query': query,
            'inclusion_criteria': dimensions
        })
        screened = screen_result['included']
        excluded = screen_result['excluded']

        # Track in PRISMA
        for p in screened:
            self.prisma.mark_screened(p.get('paper_id', ''), True)
        for p in excluded:
            reason = ExclusionReason.OFF_TOPIC
            cat = p.get('exclusion_category', '')
            if cat == 'no_abstract':
                reason = ExclusionReason.NO_ABSTRACT
            elif cat == 'low_quality':
                reason = ExclusionReason.LOW_QUALITY
            elif cat == 'wrong_type':
                reason = ExclusionReason.WRONG_TYPE
            self.prisma.mark_screened(p.get('paper_id', ''), False, reason,
                                     p.get('screening_reason', ''))

        # Quality assessment (QualityTierAgent - Tier 2)
        quality_result = self.quality_assessor.execute({
            'papers': screened,
            'domain': domain or ''
        })
        screened = quality_result['papers']
        quality_tiers = quality_result['tier_distribution']

        # Mark all screened as eligible and included in PRISMA
        for p in screened:
            pid = p.get('paper_id', '')
            tier = p.get('quality_tier', 'C')
            self.prisma.mark_eligible(pid, True, p.get('citation_count', 0))
            self.prisma.mark_included(pid, tier)

        included_papers = screened
        self.progress.update(ResearchPhase.SCREENING,
                            f"Screening complete: {len(included_papers)} included "
                            f"({len(excluded)} excluded)",
                            1.0, len(included_papers))

        # PRISMA compliance check (PRISMAComplianceAgent - Tier 1)
        compliance = self.prisma_agent.execute({
            'prisma_tracker': self.prisma,
            'papers': included_papers,
            'stage': 'screening'
        })
        if compliance.get('warnings'):
            for w in compliance['warnings'][:3]:
                self.progress.update(ResearchPhase.SCREENING,
                                    f"PRISMA warning: {w}", 1.0, len(included_papers))

        # Cache included papers
        if self.cache:
            self.cache.set_papers_batch(included_papers)

        # =====================================================
        # PHASE 8: Semantic Clustering + Theme Labeling
        # =====================================================
        cluster_result = None
        if self.enable_clustering and len(included_papers) >= 10:
            self.progress.update(ResearchPhase.CLUSTERING,
                                "Clustering papers by topic", 0.0)
            try:
                self.clusterer = SemanticClusterer()
                cluster_result = self.clusterer.cluster_papers(included_papers)
                n_clusters = cluster_result.get('n_clusters', 0)

                # Theme labeling (ClusterThemingAgent - Tier 2)
                if cluster_result.get('clusters'):
                    self.progress.update(ResearchPhase.CLUSTERING,
                                        f"Labeling {n_clusters} clusters with themes", 0.5,
                                        len(included_papers))
                    theme_result = self.cluster_themer.execute({
                        'clusters': cluster_result['clusters'],
                        'query': query
                    })
                    cluster_result['clusters'] = theme_result['labeled_clusters']

                self.progress.update(ResearchPhase.CLUSTERING,
                                    f"Found {n_clusters} themed topic clusters",
                                    1.0, len(included_papers))
            except Exception as e:
                self.progress.update(ResearchPhase.CLUSTERING,
                                    f"Clustering skipped: {e}", 1.0)

        # =====================================================
        # PHASE 9: Coordinated Synthesis (SynthesisCoordinatorAgent)
        # =====================================================
        self.progress.update(ResearchPhase.MAP_SYNTHESIS,
                            f"Planning synthesis for {len(included_papers)} papers",
                            0.0)

        # Get synthesis plan from coordinator (Tier 3)
        synth_plan = self.synthesis_coordinator.execute({
            'papers': included_papers,
            'clusters': cluster_result.get('clusters', []) if cluster_result else [],
            'query': query
        })

        # Apply plan config to map-reduce
        mr_config = synth_plan.get('map_reduce_config', {})

        def map_reduce_progress(phase, current, total, data):
            if phase == 'map':
                self.progress.update(ResearchPhase.MAP_SYNTHESIS,
                                    f"Mapping chunk {current}/{total}",
                                    current / max(total, 1),
                                    len(included_papers))
            elif phase == 'reduce_complete':
                self.progress.update(ResearchPhase.REDUCE_SYNTHESIS,
                                    "Reduce phase complete", 1.0,
                                    len(included_papers))
            elif phase == 'final_complete':
                self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                    "Final synthesis complete", 1.0,
                                    len(included_papers))

        mr_result = self.map_reduce.synthesize(
            included_papers, query, progress_callback=map_reduce_progress
        )

        # =====================================================
        # PHASE 10: Report Generation (ReportComposerAgent)
        # =====================================================
        self.progress.update(ResearchPhase.REPORT_GENERATION,
                            "Composing report", 0.0)

        elapsed = time.monotonic() - start_time
        prisma_flow = self.prisma.get_flow_counts()
        prisma_diagram = self.prisma.generate_prisma_text()

        # Get intelligent report structure from composer (Tier 3)
        report_structure = self.report_composer.execute({
            'synthesis': mr_result,
            'papers': included_papers,
            'query': query,
            'clusters': cluster_result.get('clusters', []) if cluster_result else [],
            'deep_analysis': {},
            'prisma_flow': prisma_flow
        })

        report = self._generate_report(
            query=query,
            papers=included_papers,
            mr_result=mr_result,
            cluster_result=cluster_result,
            prisma_flow=prisma_flow,
            prisma_diagram=prisma_diagram,
            decomposition=decomposition,
            elapsed=elapsed,
            report_structure=report_structure
        )

        self.progress.update(ResearchPhase.COMPLETE,
                            f"Systematic review complete: {len(included_papers)} papers analyzed",
                            1.0, len(included_papers))

        # Print PRISMA diagram
        if self.verbose:
            print(prisma_diagram)

        return {
            'query': query,
            'papers': included_papers,
            'total_papers': len(included_papers),
            'prisma_flow': prisma_flow,
            'prisma_diagram': prisma_diagram,
            'clusters': cluster_result,
            'synthesis': mr_result,
            'decomposition': decomposition,
            'report': report,
            'quality_tiers': quality_tiers,
            'statistics': {
                'total_identified': prisma_flow.get('identified', 0),
                'duplicates_removed': prisma_flow.get('duplicates_removed', 0),
                'included': len(included_papers),
                'search_rounds': round_num,
                'elapsed_seconds': elapsed,
                'clusters': cluster_result.get('n_clusters', 0) if cluster_result else 0,
                'themes_found': len(mr_result.get('reduced_synthesis', {}).get('major_themes', [])),
                'cache_stats': self.cache.get_stats() if self.cache else {}
            },
            'progress_timeline': self.progress.get_timeline()
        }

    def _quick_dedup(self, papers: List[Dict]) -> List[Dict]:
        """Fast deduplication by DOI/title."""
        seen = set()
        unique = []
        for p in papers:
            key = p.get('doi') or (p.get('title', '') or '').lower().strip()[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def _generate_report(self, query: str, papers: List[Dict],
                          mr_result: Dict, cluster_result: Dict,
                          prisma_flow: Dict, prisma_diagram: str,
                          decomposition: Dict, elapsed: float,
                          report_structure: Dict = None) -> str:
        """Generate the final markdown report."""

        final = mr_result.get('final_synthesis', {})
        reduced = mr_result.get('reduced_synthesis', {})
        rs = report_structure or {}

        # Cluster section - now uses themed labels from ClusterThemingAgent
        cluster_section = ""
        if cluster_result and cluster_result.get('clusters'):
            cluster_section = "\n## Topic Clusters\n\n"
            for c in cluster_result['clusters'][:15]:
                label = c.get('themed_label') or c.get('label', 'Unknown')
                cluster_section += f"### {label}\n"
                cluster_section += f"- Papers: {c['size']}\n"
                cluster_section += f"- Years: {c.get('year_range', 'N/A')}\n"
                cluster_section += f"- Avg Citations: {c.get('avg_citations', 0):.0f}\n"
                if c.get('theme_description'):
                    cluster_section += f"- Theme: {c['theme_description']}\n"
                if c.get('key_concepts'):
                    cluster_section += f"- Key Concepts: {', '.join(c['key_concepts'])}\n"
                cluster_section += "\n"

        # Quality tier section
        tier_a = prisma_flow.get('quality_tiers', {}).get('A', 0)
        tier_b = prisma_flow.get('quality_tiers', {}).get('B', 0)
        tier_c = prisma_flow.get('quality_tiers', {}).get('C', 0)

        # Use ReportComposerAgent's title and summary if available
        report_title = rs.get('title', 'SYSTEMATIC REVIEW REPORT')
        exec_summary = rs.get('executive_summary') or final.get('executive_summary', 'Analysis complete.')

        report = f"""# {report_title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Query:** {query}
**Engine:** Systematic Review Protocol v3.0 (Multi-Agent)

---

## Executive Summary

{exec_summary}

**Papers Analyzed:** {len(papers)}
**Search Time:** {elapsed:.1f} seconds
**Quality Tiers:** A={tier_a} | B={tier_b} | C={tier_c}

---

## PRISMA Flow

```
{prisma_diagram}
```

**Identification:** {prisma_flow.get('identified', 0)} records from {len(prisma_flow.get('by_source', {}))} databases
**After Deduplication:** {prisma_flow.get('after_dedup', 0)}
**Screened:** {prisma_flow.get('screened', 0)}
**Included:** {prisma_flow.get('included', 0)}

---

## State of the Field

{final.get('state_of_field', '')}

---

## Key Findings

"""
        for i, finding in enumerate(final.get('key_findings', [])[:10], 1):
            report += f"{i}. **{finding.get('finding', '')}**\n"
            report += f"   - Evidence: {finding.get('evidence_strength', 'N/A')}\n"
            report += f"   - Papers: {finding.get('paper_count', 'N/A')}\n\n"

        report += f"""
---

## Major Themes

"""
        for theme in reduced.get('major_themes', [])[:10]:
            report += f"### {theme.get('theme', 'Unknown')}\n"
            report += f"- Prevalence: {theme.get('prevalence', 'N/A')}\n"
            report += f"- Est. Papers: {theme.get('paper_count_est', 'N/A')}\n"
            report += f"- {theme.get('description', '')}\n\n"

        report += f"""
---

## Cross-Cutting Patterns

"""
        for pattern in reduced.get('cross_chunk_patterns', [])[:8]:
            report += f"### {pattern.get('pattern', 'Unknown')}\n"
            report += f"{pattern.get('insight', '')}\n"
            report += f"*Confidence: {pattern.get('confidence', 0):.0%}*\n\n"

        report += f"""
---

## Contradictions & Debates

"""
        for debate in final.get('unresolved_debates', [])[:5]:
            report += f"### {debate.get('debate', 'Unknown')}\n"
            for side in debate.get('sides', []):
                report += f"- {side}\n"
            report += f"\n*Current Evidence:* {debate.get('current_evidence', 'N/A')}\n\n"

        report += f"""
---

## Methodological Landscape

"""
        methods = reduced.get('methodological_landscape', {})
        if methods.get('dominant_methods'):
            report += f"**Dominant:** {', '.join(methods['dominant_methods'][:5])}\n\n"
        if methods.get('emerging_methods'):
            report += f"**Emerging:** {', '.join(methods['emerging_methods'][:5])}\n\n"
        if methods.get('methodology_gaps'):
            report += f"**Gaps:** {', '.join(methods['methodology_gaps'][:5])}\n\n"

        report += cluster_section

        report += f"""
---

## Future Research Directions

"""
        for i, direction in enumerate(final.get('future_directions', [])[:5], 1):
            report += f"{i}. {direction}\n"

        report += f"""

---

## Research Gaps

"""
        for gap in reduced.get('research_gaps', [])[:8]:
            report += f"- {gap}\n"

        report += f"""

---

## Practical Implications

"""
        for imp in final.get('practical_implications', [])[:5]:
            report += f"- {imp}\n"

        # Add recommendations from ReportComposerAgent
        if rs.get('recommendations'):
            report += "\n\n---\n\n## Recommendations\n\n"
            for rec in rs['recommendations'][:8]:
                priority = rec.get('priority', 'MEDIUM')
                strength = rec.get('evidence_strength', 'MEDIUM')
                report += f"- **[{priority}]** {rec.get('recommendation', '')}\n"
                report += f"  *Evidence: {strength}*\n\n"

        # Add key takeaways
        if rs.get('key_takeaways'):
            report += "\n---\n\n## Key Takeaways\n\n"
            for i, t in enumerate(rs['key_takeaways'][:5], 1):
                report += f"{i}. {t}\n"

        # Add limitations
        if rs.get('limitations'):
            report += "\n\n---\n\n## Limitations\n\n"
            for lim in rs['limitations'][:5]:
                report += f"- {lim}\n"

        confidence = rs.get('confidence_assessment') or final.get('confidence_assessment', 'N/A')

        report += f"""

---

*Generated by Systematic Review Engine v3.0 (Multi-Agent)*
*Papers: {len(papers)} | Time: {elapsed:.1f}s | Confidence: {confidence}*
*Agents: 15 specialized agents across 3 tiers*
"""
        return report

    def export_results(self, results: Dict, output_dir: str = None) -> Dict[str, str]:
        """Export results to multiple formats."""
        if output_dir is None:
            output_dir = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Markdown report
        report_path = output_path / "report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(results.get('report', ''))
        files['report'] = str(report_path)

        # JSON data
        json_path = output_path / "data.json"
        export_data = {
            'query': results['query'],
            'statistics': results['statistics'],
            'prisma_flow': results['prisma_flow'],
            'quality_tiers': results.get('quality_tiers', {}),
            'synthesis': results.get('synthesis', {}),
            'papers': results['papers'][:100],  # Top 100 for JSON
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        files['data'] = str(json_path)

        # Papers list (CSV-like)
        papers_path = output_path / "papers.md"
        with open(papers_path, 'w', encoding='utf-8') as f:
            f.write("# Included Papers\n\n")
            f.write("| # | Title | Year | Citations | Source | DOI |\n")
            f.write("|---|-------|------|-----------|--------|-----|\n")
            for i, p in enumerate(results['papers'], 1):
                title = (p.get('title') or 'Unknown')[:60]
                year = p.get('year', 'N/A')
                cites = p.get('citation_count', 0)
                source = p.get('source', '')
                doi = p.get('doi', '')
                f.write(f"| {i} | {title} | {year} | {cites} | {source} | {doi} |\n")
        files['papers'] = str(papers_path)

        # Timeline
        timeline_path = output_path / "timeline.json"
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump(results.get('progress_timeline', []), f, indent=2)
        files['timeline'] = str(timeline_path)

        print(f"\n  Exported to: {output_path}")
        for name, path in files.items():
            print(f"    {name}: {path}")

        return files


def main():
    """Run a systematic review."""
    import argparse

    parser = argparse.ArgumentParser(description='Systematic Review Engine v2.0')
    parser.add_argument('query', nargs='?',
                        default="Mechanisms of cryptocurrency liquidation cascades in DeFi protocols",
                        help='Research query')
    parser.add_argument('--target', type=int, default=500,
                        help='Target number of papers (default: 500)')
    parser.add_argument('--rounds', type=int, default=8,
                        help='Max search rounds (default: 8)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--no-citations', action='store_true',
                        help='Disable citation crawling')
    parser.add_argument('--no-clusters', action='store_true',
                        help='Disable clustering')
    parser.add_argument('--export', type=str, default=None,
                        help='Export directory')
    parser.add_argument('--llm', type=str, default='ollama',
                        choices=['ollama', 'deepseek', 'gemini'],
                        help='Primary LLM provider (default: ollama)')
    parser.add_argument('--model', type=str, default=None,
                        help='Ollama model name (auto-detected if not set)')

    args = parser.parse_args()

    protocol = SystematicReviewProtocol(
        target_papers=args.target,
        max_search_rounds=args.rounds,
        enable_cache=not args.no_cache,
        enable_citation_crawl=not args.no_citations,
        enable_clustering=not args.no_clusters,
        verbose=True,
        llm_provider=args.llm,
        ollama_model=args.model
    )

    results = protocol.execute(args.query)

    # Print summary
    stats = results['statistics']
    print(f"\n{'='*80}")
    print(f"  REVIEW COMPLETE")
    print(f"{'='*80}")
    print(f"  Papers Identified:  {stats['total_identified']}")
    print(f"  Duplicates Removed: {stats['duplicates_removed']}")
    print(f"  Papers Included:    {stats['included']}")
    print(f"  Search Rounds:      {stats['search_rounds']}")
    print(f"  Topic Clusters:     {stats['clusters']}")
    print(f"  Themes Found:       {stats['themes_found']}")
    print(f"  Total Time:         {stats['elapsed_seconds']:.1f}s")
    print(f"{'='*80}")

    # Export
    export_dir = args.export or f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    protocol.export_results(results, export_dir)


if __name__ == "__main__":
    main()
