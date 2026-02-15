"""Tier 3 Agents - Strategic Council + Deep Analysis + SciSpace Capabilities"""

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
from agents.tier3.scispace_agents import (
    PaperChatAgent,
    DeepReviewAgent
)
from agents.tier3.rag_chat_agent import RAGChatAgent

__all__ = [
    'ContradictionAnalyzer',
    'TemporalEvolutionAnalyzer',
    'CausalChainExtractor',
    'ConsensusQuantifier',
    'PredictiveInsightsGenerator',
    'AdaptiveStoppingAgent',
    'SynthesisCoordinatorAgent',
    'ReportComposerAgent',
    'CitationCrawlStrategyAgent',
    'PaperChatAgent',
    'DeepReviewAgent',
    'RAGChatAgent',
]
