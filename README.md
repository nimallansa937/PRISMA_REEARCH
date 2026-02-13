# PRISMA Research Agent

A production-grade multi-agent systematic review engine capable of processing 1000+ academic papers with PRISMA methodology compliance. Features a 20-agent 3-tier architecture with adaptive search, deep synthesis, and real-time Streamlit dashboard.

## Architecture

### 3-Tier Multi-Agent System (20 Agents)

| Tier | Agents | Engine | Role |
|------|--------|--------|------|
| **Tier 1** | 3 | Scripted (no LLM) | Database queries, deduplication, PRISMA compliance |
| **Tier 2** | 6 | Gemini (primary) | Gap detection, query refinement, relevance filtering, screening, quality assessment, cluster theming |
| **Tier 3** | 11 | DeepSeek (primary) | Strategic decomposition, pattern synthesis, contradiction analysis, temporal evolution, causal chains, consensus quantification, predictive insights, adaptive stopping, synthesis coordination, report composition, citation strategy |

### Agent Roster

**Tier 1 - Scripted Executors:**
- `DatabaseQueryAgent` - Executes searches across 6 academic databases
- `DeduplicationAgent` - 6-strategy fuzzy matching (DOI, arXiv ID, PubMed ID, normalized title, fuzzy title, author-year fingerprint)
- `PRISMAComplianceAgent` - Enforces PRISMA methodology with audit logging

**Tier 2 - Specialist Analysts (Gemini):**
- `GapDetectionAgent` - Identifies coverage gaps across research dimensions
- `QueryRefinementAgent` - Generates targeted refinement queries for gaps
- `RelevanceFilterAgent` - LLM-based relevance scoring and filtering
- `ScreeningAgent` - Inclusion/exclusion screening with score thresholds
- `QualityTierAgent` - Evidence quality assessment (A/B/C tiers)
- `ClusterThemingAgent` - Semantic cluster labeling with theme descriptions

**Tier 3 - Strategic Council (DeepSeek):**
- `ResearchStrategist` - Query decomposition and dimension analysis
- `PatternSynthesizer` - Cross-cutting pattern identification
- `ContradictionAnalyzer` - Detects conflicting findings across papers
- `TemporalEvolutionAnalyzer` - Tracks emerging and declining themes
- `CausalChainExtractor` - Maps causal relationships between concepts
- `ConsensusQuantifier` - Measures agreement levels across research
- `PredictiveInsightsGenerator` - Forecasts research directions
- `AdaptiveStoppingAgent` - Multi-factor search termination decisions
- `SynthesisCoordinatorAgent` - Plans which analysis agents to activate
- `ReportComposerAgent` - Generates titles, summaries, recommendations
- `CitationCrawlStrategyAgent` - Optimizes seed selection and crawl depth

## Features

- **Multi-Source Search**: Semantic Scholar, arXiv, PubMed, CrossRef, SSRN, OpenAlex
- **Async Concurrent Searching**: Parallel queries across all sources
- **PRISMA Compliance**: Full tracking from identification through inclusion
- **Citation Network Crawling**: Snowball search with intelligent seed selection
- **Semantic Clustering**: HDBSCAN-based topic clustering with themed labels
- **Map-Reduce Synthesis**: Chunked analysis with cross-cutting pattern extraction
- **Deep Synthesis**: Contradiction analysis, temporal evolution, causal chains, consensus quantification, predictive insights
- **Adaptive Search Rounds**: Automatically stops when information gain plateaus
- **Quality Assessment**: Citation-based scoring with A/B/C tier classification
- **Live Dashboard**: Real-time Streamlit dashboard with progress tracking and network graph
- **LLM Flexibility**: Supports Gemini, DeepSeek, and local Ollama models

## Installation

```bash
# Clone the repository
git clone https://github.com/nimallansa937/PRISMA_REEARCH.git
cd PRISMA_REEARCH

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Configuration

Create a `.env` file in the project root:

```env
# Required - LLM API Keys (at least one)
GEMINI_API_KEY=your_gemini_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Optional - Local LLM
OLLAMA_URL=http://localhost:11434

# Optional - Higher rate limits
SEMANTIC_SCHOLAR_API_KEY=your_key
NCBI_API_KEY=your_key
```

## Usage

### 1. Streamlit Dashboard (Recommended)

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser. The dashboard provides:
- Query input with domain selection
- Real-time progress tracking across all 10 phases
- Live network graph of paper relationships
- Results with PRISMA flow, clusters, synthesis, and full report

### 2. FastAPI Server (for React/Frontend Integration)

```bash
uvicorn api:app --reload --port 8000
```

**Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /search?query=...` | Rule-based search (fast, no AI) |
| `GET /llm-search?query=...` | AI-powered search with relevance scoring |
| `GET /multi-agent-search?query=...` | Full 20-agent pipeline with adaptive refinement |
| `GET /analyze?query=...` | Query structure analysis |
| `GET /format?query=...` | Search + format for LLM consumption |
| `GET /health` | Health check and LLM availability |

### 3. Python API

```python
# Quick search
from agent import ResearchAgent
agent = ResearchAgent()
results = agent.search("CRISPR gene editing therapeutic applications")

# Full systematic review (1000+ papers)
from systematic_review import SystematicReviewProtocol
protocol = SystematicReviewProtocol(
    target_papers=500,
    max_search_rounds=5,
    llm_provider="gemini"
)
results = await protocol.run("Mechanisms of cryptocurrency liquidation cascades")

# Multi-agent protocol
from multi_agent_protocol import MultiAgentProtocol
protocol = MultiAgentProtocol(max_refinement_rounds=3)
results = protocol.execute(
    query="Deep learning for drug discovery",
    domain="computer_science"
)

# Deep synthesis (PhD-level analysis)
deep = protocol.run_deep_synthesis(
    papers=results['papers'],
    topic="Deep learning for drug discovery"
)
```

## Project Structure

```
academic-research-agent/
├── agents/                     # 20-agent multi-agent system
│   ├── base_agent.py           # BaseAgent, Tier1/2/3Agent base classes
│   ├── tier1/                  # Scripted executors (no LLM)
│   │   ├── database_agent.py   # Multi-source search execution
│   │   ├── deduplication_agent.py  # 6-strategy fuzzy dedup
│   │   └── prisma_agent.py     # PRISMA compliance checks
│   ├── tier2/                  # Specialist analysts (Gemini)
│   │   ├── specialists.py      # Gap detection, query refinement, relevance
│   │   └── screening_agents.py # Screening, quality tiers, cluster theming
│   └── tier3/                  # Strategic council (DeepSeek)
│       ├── council.py          # Research strategist, pattern synthesizer
│       ├── synthesis_agents.py # Contradiction, temporal, causal, consensus, predictions
│       └── strategic_agents.py # Adaptive stopping, synthesis coord, report, citations
├── config/
│   └── settings.py             # Global configuration
├── core/                       # Core engines
│   ├── query_analyzer.py       # NLP query analysis (spaCy)
│   ├── search_strategy.py      # Multi-tier search plan generation
│   ├── llm_client.py           # LLM client with fallback chains
│   ├── async_engine.py         # Async search execution
│   ├── cache_layer.py          # Response caching (7-day TTL)
│   ├── citation_crawler.py     # Citation network traversal
│   ├── prisma_tracker.py       # PRISMA flow tracking
│   └── progress_streamer.py    # Real-time progress events
├── sources/                    # Academic database connectors
│   ├── semantic_scholar.py     # Semantic Scholar API
│   ├── arxiv.py                # arXiv API
│   ├── pubmed.py               # PubMed/NCBI API
│   ├── crossref.py             # CrossRef API
│   ├── ssrn.py                 # SSRN API
│   ├── openalex.py             # OpenAlex API
│   └── async_sources.py        # Async wrappers for all sources
├── synthesis/                  # Analysis and report generation
│   ├── map_reduce.py           # Map-reduce synthesis
│   ├── clustering.py           # Semantic clustering (HDBSCAN)
│   └── report_generator.py     # Report formatting
├── validation/                 # Quality control
│   ├── quality_scorer.py       # Citation-based scoring
│   └── deduplicator.py         # Legacy deduplication
├── dashboard/
│   └── app.py                  # Streamlit live dashboard
├── agent.py                    # Simple ResearchAgent orchestrator
├── systematic_review.py        # Full systematic review protocol
├── multi_agent_protocol.py     # 20-agent orchestrator (v3.0)
├── api.py                      # FastAPI server (port 8000)
├── requirements.txt            # Python dependencies
└── .env                        # API keys (not committed)
```

## Workflow (10 Phases)

```
Phase 1:  Strategic Decomposition    (Tier 3 - DeepSeek)
Phase 2:  Multi-Source Search        (Tier 1 - Scripted)
Phase 3:  Pagination & Expansion     (Tier 1 - Scripted)
Phase 4:  Citation Network Crawling  (Tier 3 - CitationCrawlStrategy)
Phase 5:  Deduplication              (Tier 1 - DeduplicationAgent)
Phase 6:  Adaptive Search Rounds     (Tier 2 + Tier 3 loop)
Phase 7:  Screening + Quality        (Tier 2 - Screening + Quality)
Phase 8:  Semantic Clustering        (Tier 2 - ClusterThemingAgent)
Phase 9:  Coordinated Synthesis      (Tier 3 - SynthesisCoordinator)
Phase 10: Report Generation          (Tier 3 - ReportComposer)
```

## Requirements

- Python 3.10+
- At least one LLM API key (Gemini or DeepSeek) for Tier 2/3 agents
- Optional: Ollama for local LLM inference
- Optional: Semantic Scholar API key for higher rate limits

## License

MIT
