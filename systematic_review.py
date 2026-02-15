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
from agents.tier3.synthesis_agents import (
    ContradictionAnalyzer, TemporalEvolutionAnalyzer,
    CausalChainExtractor, ConsensusQuantifier, PredictiveInsightsGenerator
)
from synthesis.report_generator import generate_synthesis_report
from core.rag_engine import RAGEngine
from core.fulltext_pipeline import FullTextPipeline


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

        # Agents - Tier 3 (Deep Synthesis)
        self.contradiction_analyzer = ContradictionAnalyzer()
        self.temporal_analyzer = TemporalEvolutionAnalyzer()
        self.causal_extractor = CausalChainExtractor()
        self.consensus_quantifier = ConsensusQuantifier()
        self.prediction_generator = PredictiveInsightsGenerator()

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

        # RAG Pipeline (pass user-selected model so it uses the same model as LLMClient)
        self.rag_engine = RAGEngine(ollama_model=ollama_model)
        self.fulltext_pipeline = FullTextPipeline()

        # Synthesis engines
        self.map_reduce = MapReduceSynthesizer(llm=self.llm, rag_engine=self.rag_engine)
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

        # Track dedup in PRISMA - identify removed papers
        kept_ids = set()
        for p in papers_dicts:
            pid = p.get('paper_id') or p.get('doi') or (p.get('title', '') or '')[:100]
            kept_ids.add(pid)
        removed_ids = []
        for p in all_papers:
            pid = p.get('paper_id') or p.get('doi') or (p.get('title', '') or '')[:100]
            if pid and pid not in kept_ids:
                removed_ids.append(pid)
        if removed_ids:
            self.prisma.mark_deduplicated(removed_ids)

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

        # Track in PRISMA (use same ID fallback as add_identified)
        def _prisma_id(p):
            return p.get('paper_id') or p.get('doi') or (p.get('title', '') or '')[:100]

        for p in screened:
            self.prisma.mark_screened(_prisma_id(p), True)
        for p in excluded:
            reason = ExclusionReason.OFF_TOPIC
            cat = p.get('exclusion_category', '')
            if cat == 'no_abstract':
                reason = ExclusionReason.NO_ABSTRACT
            elif cat == 'low_quality':
                reason = ExclusionReason.LOW_QUALITY
            elif cat == 'wrong_type':
                reason = ExclusionReason.WRONG_TYPE
            self.prisma.mark_screened(_prisma_id(p), False, reason,
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
            pid = _prisma_id(p)
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
        # PHASE 8.5: RAG Indexing (Full-Text Download + ChromaDB)
        # =====================================================
        rag_stats = {}
        try:
            self.progress.update(ResearchPhase.RAG_INDEXING,
                                "Starting RAG indexing pipeline", 0.0,
                                len(included_papers))

            # Step 1: Download full texts for top papers (by citation count)
            papers_for_fulltext = sorted(
                included_papers,
                key=lambda p: p.get('citation_count', 0) or 0,
                reverse=True
            )[:100]  # Top 100 by citations

            self.progress.update(ResearchPhase.RAG_INDEXING,
                                f"Downloading full texts ({len(papers_for_fulltext)} papers)",
                                0.1, len(included_papers))

            import nest_asyncio
            nest_asyncio.apply()

            loop = asyncio.get_event_loop()
            full_text_papers = loop.run_until_complete(
                self.fulltext_pipeline.batch_get_full_text(
                    papers_for_fulltext,
                    max_concurrent=5
                )
            )

            ft_count = len(full_text_papers) if full_text_papers else 0
            self.progress.update(ResearchPhase.RAG_INDEXING,
                                f"Downloaded {ft_count} full texts",
                                0.4, len(included_papers))

            # Step 2: Index all papers into ChromaDB
            def rag_progress(msg, pct):
                self.progress.update(ResearchPhase.RAG_INDEXING,
                                    msg, 0.4 + pct * 0.6,
                                    len(included_papers))

            rag_stats = self.rag_engine.ingest_papers(
                papers=included_papers,
                full_text_papers=full_text_papers,
                progress_callback=rag_progress
            )

            self.progress.update(ResearchPhase.RAG_INDEXING,
                                f"RAG indexed: {rag_stats.get('total_chunks', 0)} chunks "
                                f"from {rag_stats.get('total_papers', 0)} papers "
                                f"({ft_count} full-text)",
                                1.0, len(included_papers))

        except Exception as e:
            if self.verbose:
                print(f"  RAG indexing failed (non-fatal): {e}")
            self.progress.update(ResearchPhase.RAG_INDEXING,
                                f"RAG indexing skipped: {str(e)[:80]}", 1.0)

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
        # PHASE 9.5: Deep Synthesis (5 Tier-3 Agents)
        # =====================================================
        self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                            "Running deep synthesis (5 agents)", 0.0,
                            len(included_papers))

        deep_synthesis = {}
        try:
            # 1. Contradiction Analysis
            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                "Analyzing contradictions (1/5)", 0.1)
            deep_synthesis['contradictions'] = self.contradiction_analyzer.execute({
                'papers': included_papers, 'topic': query
            })

            # 2. Temporal Evolution
            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                "Tracking temporal evolution (2/5)", 0.3)
            deep_synthesis['temporal_evolution'] = self.temporal_analyzer.execute({
                'papers': included_papers
            })

            # 3. Causal Chains
            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                "Extracting causal chains (3/5)", 0.5)
            deep_synthesis['causal_chains'] = self.causal_extractor.execute({
                'papers': included_papers, 'topic': query
            })

            # 4. Consensus Quantification
            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                "Quantifying consensus (4/5)", 0.7)
            deep_synthesis['consensus'] = self.consensus_quantifier.execute({
                'papers': included_papers
            })

            # 5. Predictive Insights
            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                "Generating predictions (5/5)", 0.9)
            deep_synthesis['predictions'] = self.prediction_generator.execute({
                'temporal_evolution': deep_synthesis.get('temporal_evolution', {}),
                'gaps': mr_result.get('reduced_synthesis', {}).get('research_gaps', []),
                'papers': included_papers
            })

            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                "Deep synthesis complete", 1.0,
                                len(included_papers))
        except Exception as e:
            if self.verbose:
                print(f"  Deep synthesis partial failure: {e}")
            self.progress.update(ResearchPhase.DEEP_SYNTHESIS,
                                f"Deep synthesis completed with warnings", 1.0)

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
            'deep_analysis': deep_synthesis,
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
            report_structure=report_structure,
            deep_synthesis=deep_synthesis
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
            'deep_synthesis': deep_synthesis,
            'decomposition': decomposition,
            'report': report,
            'quality_tiers': quality_tiers,
            'rag_engine': self.rag_engine,
            'rag_stats': rag_stats,
            'statistics': {
                'total_identified': prisma_flow.get('identified', 0),
                'duplicates_removed': prisma_flow.get('duplicates_removed', 0),
                'included': len(included_papers),
                'search_rounds': round_num,
                'elapsed_seconds': elapsed,
                'clusters': cluster_result.get('n_clusters', 0) if cluster_result else 0,
                'themes_found': len(mr_result.get('reduced_synthesis', {}).get('major_themes', [])),
                'rag_chunks': rag_stats.get('total_chunks', 0),
                'rag_fulltext_papers': rag_stats.get('total_papers', 0),
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
                          report_structure: Dict = None,
                          deep_synthesis: Dict = None) -> str:
        """Generate the final markdown report with data-driven fallbacks."""

        final = mr_result.get('final_synthesis', {})
        reduced = mr_result.get('reduced_synthesis', {})
        ds = deep_synthesis or {}
        rs = report_structure or {}

        # ============================================================
        # DATA-DRIVEN ANALYSIS (always available, even when LLM fails)
        # ============================================================

        # Top papers by citation
        sorted_by_cites = sorted(papers, key=lambda p: p.get('citation_count', 0) or 0, reverse=True)
        top_cited = sorted_by_cites[:15]

        # Year distribution
        year_counts = {}
        for p in papers:
            y = p.get('year', 0)
            if y and y > 1900:
                year_counts[y] = year_counts.get(y, 0) + 1

        # Source distribution
        source_counts = {}
        for p in papers:
            s = p.get('source', 'unknown')
            source_counts[s] = source_counts.get(s, 0) + 1

        # Venue distribution (top venues)
        venue_counts = {}
        for p in papers:
            v = p.get('venue', '') or ''
            if v and v.lower() not in ('', 'unknown', 'n/a', 'none'):
                venue_counts[v] = venue_counts.get(v, 0) + 1
        top_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Author frequency
        author_counts = {}
        for p in papers:
            for a in (p.get('authors') or []):
                if a and a != 'Unknown':
                    author_counts[a] = author_counts.get(a, 0) + 1
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Citation statistics
        cites = [p.get('citation_count', 0) or 0 for p in papers]
        total_cites = sum(cites)
        avg_cites = total_cites / max(len(cites), 1)
        max_cite = max(cites) if cites else 0
        median_cite = sorted(cites)[len(cites) // 2] if cites else 0

        # Papers with DOI (verified)
        doi_count = sum(1 for p in papers if p.get('doi'))
        oa_count = sum(1 for p in papers if p.get('has_full_text') or p.get('source') == 'core')

        # ============================================================
        # CLUSTER SECTION
        # ============================================================
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

        # ============================================================
        # QUALITY TIERS
        # ============================================================
        tier_a = prisma_flow.get('quality_tiers', {}).get('A', 0)
        tier_b = prisma_flow.get('quality_tiers', {}).get('B', 0)
        tier_c = prisma_flow.get('quality_tiers', {}).get('C', 0)

        # ============================================================
        # BUILD REPORT
        # ============================================================
        report_title = rs.get('title', 'SYSTEMATIC REVIEW REPORT')
        exec_summary = (rs.get('executive_summary') or
                        final.get('executive_summary') or
                        f"Systematic review of {len(papers)} papers on \"{query}\". "
                        f"Papers span {min(year_counts.keys()) if year_counts else 'N/A'}-"
                        f"{max(year_counts.keys()) if year_counts else 'N/A'}, "
                        f"sourced from {len(source_counts)} databases, "
                        f"with {total_cites:,} total citations (avg {avg_cites:.0f}/paper). "
                        f"The corpus includes {doi_count} DOI-verified papers "
                        f"across {len(venue_counts)} distinct venues.")

        report = f"""# {report_title}

**{datetime.now().strftime('%B %d, %Y')}** · {len(papers)} papers · {elapsed:.0f}s · Quality A={tier_a} B={tier_b} C={tier_c}

---

## Executive Summary

{exec_summary}

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

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Total papers | {len(papers)} |
| Year range | {min(year_counts.keys()) if year_counts else 'N/A'} - {max(year_counts.keys()) if year_counts else 'N/A'} |
| Total citations | {total_cites:,} |
| Avg citations/paper | {avg_cites:.1f} |
| Median citations | {median_cite} |
| Max citations | {max_cite:,} |
| DOI verified | {doi_count} ({100*doi_count/max(len(papers),1):.0f}%) |
| Distinct venues | {len(venue_counts)} |
| Distinct authors | {len(author_counts)} |

### Source Distribution

"""
        for src, cnt in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * cnt / max(len(papers), 1)
            report += f"- **{src}**: {cnt} papers ({pct:.0f}%)\n"

        report += "\n### Year Distribution\n\n"
        for year in sorted(year_counts.keys(), reverse=True)[:15]:
            cnt = year_counts[year]
            bar = "█" * min(cnt, 40)
            report += f"- {year}: {bar} ({cnt})\n"

        # ============================================================
        # STATE OF THE FIELD (concise — stats already in Corpus Statistics)
        # ============================================================
        state_of_field = final.get('state_of_field', '')
        if not state_of_field:
            # Concise narrative — no repeating stats shown above
            peak_year = max(year_counts, key=year_counts.get) if year_counts else 'N/A'
            recent_pct = sum(v for k, v in year_counts.items() if k >= 2020) / max(len(papers), 1) * 100
            trend_label = 'rapidly growing' if recent_pct > 60 else ('actively evolving' if recent_pct > 40 else 'mature and established')
            state_of_field = (
                f"Research on \"{query}\" is **{trend_label}**, with peak output in "
                f"**{peak_year}**. {recent_pct:.0f}% of the corpus is from 2020 onward."
            )

        report += f"""
---

## State of the Field

{state_of_field}

---

## Key Findings

"""
        findings = final.get('key_findings', [])
        if findings:
            for i, finding in enumerate(findings[:10], 1):
                report += f"{i}. **{finding.get('finding', '')}**\n"
                report += f"   - Evidence: {finding.get('evidence_strength', 'N/A')}\n"
                report += f"   - Papers: {finding.get('paper_count', 'N/A')}\n\n"
        else:
            # Data-driven fallback: highlight top-cited papers as key contributions
            report += "*Key findings from the most influential papers in this corpus:*\n\n"
            for i, p in enumerate(top_cited[:8], 1):
                title = (p.get('title') or 'Unknown')[:100]
                cites_val = p.get('citation_count', 0) or 0
                year_val = p.get('year', 'N/A')
                abstract = (p.get('abstract') or '')[:200]
                report += f"{i}. **{title}** ({year_val}, {cites_val:,} citations)\n"
                if abstract:
                    report += f"   - {abstract}...\n\n"
                else:
                    report += "\n"

        # ============================================================
        # RAG-ENHANCED DEEP ANALYSIS (gpt-oss:120b-cloud grounded)
        # ============================================================
        rag_stats = self.rag_engine.get_stats() if self.rag_engine else {}
        if rag_stats.get('total_chunks', 0) > 0:
            report += """
---

## Deep Evidence Analysis (RAG-Grounded)

*Generated using retrieval-augmented generation over full-text papers.*

"""
            try:
                # Generate comprehensive RAG section using frontier model
                rag_result = self.rag_engine.generate(
                    query=f"Provide a comprehensive analysis of the key findings, methodological approaches, "
                          f"and practical implications in the research on: {query}. "
                          f"Include specific statistics, data points, and author citations.",
                    top_k=30,
                    temperature=0.3
                )

                rag_answer = rag_result.get('answer', '')
                rag_citations = rag_result.get('citations', [])
                rag_chunks_used = rag_result.get('chunks_used', 0)

                if rag_answer and len(rag_answer) > 100:
                    report += f"{rag_answer}\n\n"
                    report += f"*Based on {rag_chunks_used} evidence chunks | "
                    report += f"Model: {rag_result.get('model', 'unknown')}*\n\n"

                    if rag_citations:
                        report += "**Sources cited:**\n"
                        for cite in rag_citations[:15]:
                            ref = cite.get('reference', '')
                            title = cite.get('title', '')[:80]
                            doi = cite.get('doi', '')
                            report += f"- {ref} {title}"
                            if doi:
                                report += f" (DOI: {doi})"
                            report += "\n"
                        report += "\n"
            except Exception as e:
                report += f"*RAG analysis unavailable: {str(e)[:100]}*\n\n"

        report += """
---

## Major Themes

"""
        themes = reduced.get('major_themes', [])
        if themes:
            for theme in themes[:10]:
                report += f"### {theme.get('theme', 'Unknown')}\n"
                report += f"- Prevalence: {theme.get('prevalence', 'N/A')}\n"
                report += f"- Est. Papers: {theme.get('paper_count_est', 'N/A')}\n"
                report += f"- {theme.get('description', '')}\n\n"
        elif cluster_result and cluster_result.get('clusters'):
            # Fallback: use cluster themes
            report += "*Themes identified via semantic clustering:*\n\n"
            for c in cluster_result['clusters'][:10]:
                label = c.get('themed_label') or c.get('label', 'Unknown')
                report += f"### {label}\n"
                report += f"- Papers: {c['size']}\n"
                if c.get('theme_description'):
                    report += f"- {c['theme_description']}\n"
                report += "\n"
        else:
            report += "*No thematic analysis available. See Top Venues for topic distribution:*\n\n"
            for v, cnt in top_venues[:8]:
                report += f"- **{v}**: {cnt} papers\n"

        report += """
---

## Cross-Cutting Patterns

"""
        patterns = reduced.get('cross_chunk_patterns', [])
        if patterns:
            for pattern in patterns[:8]:
                report += f"### {pattern.get('pattern', 'Unknown')}\n"
                report += f"{pattern.get('insight', '')}\n"
                conf = pattern.get('confidence', 0)
                if isinstance(conf, (int, float)):
                    report += f"*Confidence: {conf:.0%}*\n\n"
                else:
                    report += f"*Confidence: {conf}*\n\n"
        else:
            report += "*Cross-cutting analysis pending. See cluster and citation data above.*\n\n"

        # ============================================================
        # DEEP SYNTHESIS: CONTRADICTIONS (from ContradictionAnalyzer)
        # ============================================================
        report += """
---

## Contradictions & Debates

"""
        ds_contradictions = ds.get('contradictions', {}).get('contradictions', [])
        debates = final.get('unresolved_debates', [])

        if ds_contradictions:
            for i, c in enumerate(ds_contradictions[:8], 1):
                topic = c.get('topic', 'Unknown')
                pos_a = c.get('position_a', {})
                pos_b = c.get('position_b', {})
                resolution = c.get('resolution', 'Unresolved')
                consensus = c.get('consensus_strength', 'N/A')

                report += f"### {i}. {topic}\n\n"
                if pos_a:
                    strength_a = pos_a.get('evidence_strength', 'N/A')
                    papers_a = pos_a.get('paper_ids', [])
                    report += f"**Position A** ({strength_a} evidence, Papers: {papers_a}):\n"
                    report += f"> {pos_a.get('claim', 'N/A')}\n\n"
                if pos_b:
                    strength_b = pos_b.get('evidence_strength', 'N/A')
                    papers_b = pos_b.get('paper_ids', [])
                    report += f"**Position B** ({strength_b} evidence, Papers: {papers_b}):\n"
                    report += f"> {pos_b.get('claim', 'N/A')}\n\n"
                report += f"**Resolution:** {resolution} | **Consensus:** {consensus}\n\n"
        elif debates:
            for debate in debates[:5]:
                report += f"### {debate.get('debate', 'Unknown')}\n"
                for side in debate.get('sides', []):
                    report += f"- {side}\n"
                report += f"\n*Current Evidence:* {debate.get('current_evidence', 'N/A')}\n\n"
        else:
            chunk_contradictions = reduced.get('contradictions', [])
            if chunk_contradictions:
                for c in chunk_contradictions[:5]:
                    if isinstance(c, dict):
                        report += f"- **{c.get('topic', 'Unknown')}**: {c.get('resolution', 'Unresolved')}\n"
                    elif isinstance(c, str):
                        report += f"- {c}\n"
                report += "\n"
            else:
                report += "*No significant contradictions identified in the current corpus.*\n\n"

        # ============================================================
        # DEEP SYNTHESIS: TEMPORAL EVOLUTION (from TemporalEvolutionAnalyzer)
        # ============================================================
        ds_temporal = ds.get('temporal_evolution', {})
        if ds_temporal and 'error' not in ds_temporal:
            report += """
---

## Temporal Evolution of Research

"""
            emerging = ds_temporal.get('emerging_themes', [])
            declining = ds_temporal.get('declining_themes', [])
            stable = ds_temporal.get('stable_themes', [])
            interpretation = ds_temporal.get('interpretation', '')

            if emerging:
                report += "### Emerging Themes\n\n"
                for t in emerging[:6]:
                    name = t.get('theme', 'Unknown')
                    appeared = t.get('first_appeared', 'N/A')
                    growth = t.get('growth_rate', 'N/A')
                    count = t.get('paper_count', 0)
                    report += f"- **{name}** (since {appeared}) - {count} papers, growth: {growth}\n"
                report += "\n"

            if declining:
                report += "### Declining Themes\n\n"
                for t in declining[:6]:
                    name = t.get('theme', 'Unknown')
                    peak = t.get('peak_year', 'N/A')
                    decline = t.get('decline_rate', 'N/A')
                    report += f"- **{name}** (peaked {peak}) - decline: {decline}\n"
                report += "\n"

            if stable:
                report += f"### Stable Themes\n\n"
                report += f"{', '.join(stable[:8])}\n\n"

            if interpretation:
                report += f"**Interpretation:** {interpretation}\n\n"

        # ============================================================
        # DEEP SYNTHESIS: CAUSAL CHAINS (from CausalChainExtractor)
        # ============================================================
        ds_chains = ds.get('causal_chains', {}).get('causal_chains', [])
        if ds_chains:
            report += """
---

## Causal Chains

"""
            for i, chain in enumerate(ds_chains[:6], 1):
                steps = chain.get('chain', [])
                strength = chain.get('evidence_strength', 'N/A')
                loe = chain.get('loe_range', 'N/A')

                if steps:
                    chain_str = " → ".join(
                        f"**{s.get('step', '?')}** ({s.get('description', '')})"
                        for s in steps
                    )
                    report += f"{i}. {chain_str}\n"
                    report += f"   *Evidence: {strength} | Level of Evidence: {loe}*\n\n"

        # ============================================================
        # DEEP SYNTHESIS: CONSENSUS (from ConsensusQuantifier)
        # ============================================================
        ds_consensus = ds.get('consensus', {}).get('consensus_results', [])
        if ds_consensus:
            report += """
---

## Evidence Consensus

| Theme | Papers | Consensus | Quality (H/M/L) | Actionable |
|-------|--------|-----------|------------------|------------|
"""
            for c in ds_consensus[:10]:
                theme = c.get('theme', 'Unknown')
                count = c.get('paper_count', 0)
                strength = c.get('consensus_strength', 'N/A')
                qual = c.get('quality_distribution', {})
                h = qual.get('high_quality', 0)
                m = qual.get('medium_quality', 0)
                l_val = qual.get('low_quality', 0)
                actionable = "Yes" if c.get('actionable') else "No"
                report += f"| {theme} | {count} | {strength} | {h}/{m}/{l_val} | {actionable} |\n"
            report += "\n"

        # ============================================================
        # DEEP SYNTHESIS: PREDICTIONS (from PredictiveInsightsGenerator)
        # ============================================================
        ds_predictions = ds.get('predictions', {}).get('predictions', [])
        if ds_predictions:
            report += """
---

## Predictive Insights (1-3 Year Forecast)

"""
            for i, pred in enumerate(ds_predictions[:6], 1):
                prediction = pred.get('prediction', 'Unknown')
                basis = pred.get('basis', 'N/A')
                confidence = pred.get('confidence', 'N/A')
                timeframe = pred.get('timeframe', 'N/A')
                metric = pred.get('testable_metric', 'N/A')

                report += f"### {i}. {prediction}\n\n"
                report += f"- **Basis:** {basis}\n"
                report += f"- **Confidence:** {confidence} | **Timeframe:** {timeframe}\n"
                report += f"- **Testable Metric:** {metric}\n\n"

        # ============================================================
        # METHODOLOGICAL LANDSCAPE
        # ============================================================
        report += """
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
        if not methods:
            report += "*Methodological analysis pending.*\n\n"

        # Insert cluster section
        if cluster_section:
            report += cluster_section

        # ============================================================
        # MOST INFLUENTIAL PAPERS (always data-driven)
        # ============================================================
        report += """
---

## Most Influential Papers

"""
        for i, p in enumerate(top_cited[:10], 1):
            title = (p.get('title') or 'Unknown')[:100]
            cites_val = p.get('citation_count', 0) or 0
            year_val = p.get('year', 'N/A')
            authors = ', '.join((p.get('authors') or ['Unknown'])[:3])
            source = p.get('source', '')
            doi = p.get('doi', '')
            report += f"{i}. **{title}**\n"
            report += f"   - {authors} ({year_val}) | {cites_val:,} citations | {source}"
            if doi:
                report += f" | DOI: {doi}"
            report += "\n\n"

        # ============================================================
        # PROLIFIC AUTHORS (always data-driven)
        # ============================================================
        if top_authors:
            report += """
---

## Most Prolific Authors

"""
            for i, (author, cnt) in enumerate(top_authors[:10], 1):
                report += f"{i}. **{author}** - {cnt} papers\n"

        # ============================================================
        # TOP VENUES (always data-driven)
        # ============================================================
        if top_venues:
            report += """

---

## Top Publication Venues

"""
            for i, (venue, cnt) in enumerate(top_venues[:10], 1):
                report += f"{i}. **{venue}** - {cnt} papers\n"

        # ============================================================
        # FUTURE DIRECTIONS
        # ============================================================
        report += """

---

## Future Research Directions

"""
        future = final.get('future_directions', [])
        if future:
            for i, direction in enumerate(future[:8], 1):
                report += f"{i}. {direction}\n"
        else:
            gaps = reduced.get('research_gaps', [])
            if gaps:
                report += "*Based on identified research gaps:*\n\n"
                for i, gap in enumerate(gaps[:8], 1):
                    report += f"{i}. {gap}\n"
            else:
                report += "*Future research directions pending deeper synthesis analysis.*\n"

        # ============================================================
        # RESEARCH GAPS
        # ============================================================
        report += """

---

## Research Gaps

"""
        gaps = reduced.get('research_gaps', [])
        if gaps:
            for gap in gaps[:8]:
                report += f"- {gap}\n"
        else:
            report += "*No specific gaps identified. Consider running Deep Review for gap mapping.*\n"

        # ============================================================
        # PRACTICAL IMPLICATIONS
        # ============================================================
        report += """

---

## Practical Implications

"""
        implications = final.get('practical_implications', [])
        if implications:
            for imp in implications[:5]:
                report += f"- {imp}\n"
        else:
            report += "*Practical implications pending deeper synthesis analysis.*\n"

        # ============================================================
        # RECOMMENDATIONS (from ReportComposerAgent)
        # ============================================================
        if rs.get('recommendations'):
            report += "\n\n---\n\n## Recommendations\n\n"
            for rec in rs['recommendations'][:8]:
                priority = rec.get('priority', 'MEDIUM')
                strength = rec.get('evidence_strength', 'MEDIUM')
                report += f"- **[{priority}]** {rec.get('recommendation', '')}\n"
                report += f"  *Evidence: {strength}*\n\n"

        # Key takeaways
        if rs.get('key_takeaways'):
            report += "\n---\n\n## Key Takeaways\n\n"
            for i, t in enumerate(rs['key_takeaways'][:5], 1):
                report += f"{i}. {t}\n"

        # Limitations
        if rs.get('limitations'):
            report += "\n\n---\n\n## Limitations\n\n"
            for lim in rs['limitations'][:5]:
                report += f"- {lim}\n"

        confidence = rs.get('confidence_assessment') or final.get('confidence_assessment', 'N/A')

        rag_info = ""
        rag_s = self.rag_engine.get_stats() if self.rag_engine else {}
        if rag_s.get('total_chunks', 0) > 0:
            rag_info = (f" | RAG: {rag_s['total_chunks']} chunks, "
                        f"{rag_s.get('total_papers', 0)} papers, "
                        f"model: {rag_s.get('llm_model', 'N/A')}")

        report += f"""

---

*Generated by Systematic Review Engine v3.0 + SciSpace AI + RAG Pipeline (22 Multi-Agents)*
*Papers: {len(papers)} | Sources: {len(source_counts)} | Time: {elapsed:.1f}s | Confidence: {confidence}{rag_info}*
*22 specialized agents across 3 tiers | 7 academic databases | ChromaDB vector store*
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
