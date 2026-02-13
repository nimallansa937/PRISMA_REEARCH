"""
Systematic Review Engine v2.0 - Dashboard
Full-featured Streamlit GUI for 1000+ paper systematic reviews.

Run with: streamlit run dashboard/app.py

Features:
- Ollama / DeepSeek / Gemini model selection
- Real-time progress tracking
- PRISMA flow visualization
- Topic cluster explorer
- Map-Reduce synthesis viewer
- Quality tier breakdown
- Export to Markdown / JSON
- Year & source distribution charts
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import sys
import time
import math
import hashlib
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# GRAPH HELPER FUNCTIONS
# ============================================================

# Source colors for graph nodes
SOURCE_COLORS = {
    'semantic_scholar': '#3B82F6',  # blue
    'openalex': '#10B981',          # green
    'arxiv': '#F59E0B',             # amber
    'pubmed': '#EF4444',            # red
    'crossref': '#8B5CF6',          # purple
    'ssrn': '#EC4899',              # pink
    'core': '#F97316',              # orange (CORE API - 140M+ OA papers)
    'citation_crawl': '#6366F1',    # indigo
    'unknown': '#9CA3AF',           # gray
}

CLUSTER_COLORS = [
    '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
    '#EC4899', '#6366F1', '#14B8A6', '#F97316', '#06B6D4',
    '#84CC16', '#E11D48', '#7C3AED', '#0EA5E9', '#D946EF',
]


def build_live_graph_figure(papers_list, color_by='source', max_nodes=150):
    """
    Build a Plotly scatter figure that looks like a network graph.
    Papers are positioned using year (x-axis) and a hash-based spread (y-axis).
    Node size = citation count, color = source or cluster.

    This is fast enough to update live during search.
    """
    if not papers_list:
        fig = go.Figure()
        fig.update_layout(
            height=420,
            paper_bgcolor='#0F172A', plot_bgcolor='#0F172A',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Paper Discovery Network", font=dict(color='#94A3B8', size=14)),
        )
        fig.add_annotation(text="Waiting for papers...", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color='#475569', size=16))
        return fig

    # Limit nodes for performance
    show_papers = papers_list[:max_nodes]

    # Position: x = year, y = hash-spread within year
    xs, ys, sizes, colors, hovers, opacities = [], [], [], [], [], []
    year_buckets = {}

    for i, p in enumerate(show_papers):
        year = p.get('year') or 2020
        title = (p.get('title') or 'Untitled')[:60]
        source = p.get('source', 'unknown')
        cites = max(1, p.get('citation_count', 0) or 0)
        cluster_id = p.get('cluster_id', -1)

        # x = year with jitter
        h = int(hashlib.md5(title.encode()).hexdigest()[:8], 16)
        jitter_x = (h % 100) / 100 * 0.8 - 0.4
        x = year + jitter_x

        # y = spread within year band
        if year not in year_buckets:
            year_buckets[year] = 0
        idx_in_year = year_buckets[year]
        year_buckets[year] += 1
        # Spiral-like spread
        angle = idx_in_year * 2.39996  # golden angle
        radius = math.sqrt(idx_in_year + 1) * 0.3
        y = math.sin(angle) * radius

        xs.append(x)
        ys.append(y)
        sizes.append(min(4 + math.log2(cites + 1) * 4, 30))

        if color_by == 'cluster' and cluster_id >= 0:
            colors.append(CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)])
        else:
            colors.append(SOURCE_COLORS.get(source, SOURCE_COLORS['unknown']))

        first_author = (p.get('authors') or ['Unknown'])[0] if p.get('authors') else 'Unknown'
        hovers.append(f"<b>{title}</b><br>{first_author}, {year}<br>"
                      f"Source: {source}<br>Citations: {cites}")
        # Newer papers = more opaque (fade-in effect)
        opacities.append(min(0.4 + (i / max(len(show_papers), 1)) * 0.6, 1.0))

    # Draw edges: connect papers with same first author or very close years from same source
    edge_x, edge_y = [], []
    # Simple proximity edges (papers close in x,y space from same source)
    for i in range(len(show_papers)):
        for j in range(i + 1, min(i + 8, len(show_papers))):  # check nearby papers
            pi, pj = show_papers[i], show_papers[j]
            # Same first author
            ai = (pi.get('authors') or [''])[0].lower() if pi.get('authors') else ''
            aj = (pj.get('authors') or [''])[0].lower() if pj.get('authors') else ''
            same_author = ai and aj and ai == aj

            # Same source & close year
            close = (pi.get('source') == pj.get('source') and
                     abs((pi.get('year') or 0) - (pj.get('year') or 0)) <= 1)

            if same_author or close:
                edge_x.extend([xs[i], xs[j], None])
                edge_y.extend([ys[i], ys[j], None])

    fig = go.Figure()

    # Edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.4, color='rgba(100,116,139,0.2)'),
            hoverinfo='skip', showlegend=False,
        ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=opacities,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
        ),
        text=hovers, hoverinfo='text', showlegend=False,
    ))

    # Year axis labels
    if xs:
        min_yr = int(min(xs)) - 1
        max_yr = int(max(xs)) + 1
    else:
        min_yr, max_yr = 2015, 2025

    fig.update_layout(
        height=420,
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        margin=dict(l=10, r=10, t=35, b=30),
        title=dict(
            text=f"Paper Discovery Network ({len(show_papers)} papers)",
            font=dict(color='#94A3B8', size=13),
        ),
        xaxis=dict(
            showgrid=True, gridcolor='rgba(51,65,85,0.3)',
            tickfont=dict(color='#64748B', size=10),
            range=[min_yr, max_yr],
            dtick=2, title=None,
        ),
        yaxis=dict(
            showgrid=False, visible=False,
        ),
        hovermode='closest',
    )

    return fig


def build_full_network_html(papers, clusters=None, max_nodes=200):
    """
    Build an interactive pyvis network graph and return HTML string.
    Used in the results view (not during live updates).
    """
    try:
        from pyvis.network import Network
        import networkx as nx
    except ImportError:
        return "<p>pyvis/networkx not installed. Run: pip install pyvis networkx</p>"

    G = nx.Graph()
    show_papers = papers[:max_nodes]

    # Assign cluster colors
    paper_clusters = {}
    if clusters and clusters.get('paper_labels'):
        labels = clusters['paper_labels']
        for i, lbl in enumerate(labels[:len(show_papers)]):
            paper_clusters[i] = lbl

    # Add nodes
    for i, p in enumerate(show_papers):
        title = (p.get('title') or 'Untitled')[:80]
        year = p.get('year') or 2020
        source = p.get('source', 'unknown')
        cites = max(1, p.get('citation_count', 0) or 0)
        first_author = (p.get('authors') or ['Unknown'])[0] if p.get('authors') else 'Unknown'
        cluster_id = paper_clusters.get(i, -1)

        if cluster_id >= 0:
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        else:
            color = SOURCE_COLORS.get(source, SOURCE_COLORS['unknown'])

        node_size = min(8 + math.log2(cites + 1) * 5, 40)

        label = f"{first_author}, {year}"
        hover = f"{title}\n{first_author}, {year}\nSource: {source}\nCitations: {cites}"

        G.add_node(i, label=label, title=hover, size=node_size,
                   color=color, font={'size': 8, 'color': '#CBD5E1'})

    # Add edges based on shared authorship / cluster / year proximity
    for i in range(len(show_papers)):
        for j in range(i + 1, len(show_papers)):
            pi, pj = show_papers[i], show_papers[j]
            weight = 0

            # Same first author
            ai = (pi.get('authors') or [''])[0].lower() if pi.get('authors') else ''
            aj = (pj.get('authors') or [''])[0].lower() if pj.get('authors') else ''
            if ai and aj and ai == aj:
                weight += 3

            # Same cluster
            ci = paper_clusters.get(i, -1)
            cj = paper_clusters.get(j, -1)
            if ci >= 0 and ci == cj:
                weight += 1

            # Close years, same source
            if (pi.get('source') == pj.get('source') and
                    abs((pi.get('year') or 0) - (pj.get('year') or 0)) <= 1):
                weight += 1

            if weight >= 2:
                G.add_edge(i, j, weight=weight, color='rgba(100,116,139,0.25)')

    # Build pyvis network
    net = Network(height="550px", width="100%", bgcolor="#0F172A",
                  font_color="#CBD5E1", directed=False)
    net.from_nx(G)
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -30,
                "centralGravity": 0.005,
                "springLength": 100,
                "springConstant": 0.02,
                "damping": 0.8
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 80}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "width": 0.5
        }
    }
    """)

    return net.generate_html()


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Systematic Review Engine v2.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: -10px;
    }

    /* Tier badges */
    .tier-badge-1 {
        background-color: #DBEAFE; color: #1E40AF;
        padding: 0.2rem 0.6rem; border-radius: 0.4rem;
        font-size: 0.8rem; font-weight: 600;
    }
    .tier-badge-2 {
        background-color: #FEF3C7; color: #92400E;
        padding: 0.2rem 0.6rem; border-radius: 0.4rem;
        font-size: 0.8rem; font-weight: 600;
    }
    .tier-badge-3 {
        background-color: #DCFCE7; color: #166534;
        padding: 0.2rem 0.6rem; border-radius: 0.4rem;
        font-size: 0.8rem; font-weight: 600;
    }

    /* LLM provider badges */
    .llm-ollama {
        background: #E0F2FE; color: #0369A1;
        padding: 0.15rem 0.5rem; border-radius: 0.3rem;
        font-size: 0.75rem; font-weight: 600;
    }
    .llm-gemini {
        background: #FEF3C7; color: #92400E;
        padding: 0.15rem 0.5rem; border-radius: 0.3rem;
        font-size: 0.75rem; font-weight: 600;
    }
    .llm-deepseek {
        background: #DCFCE7; color: #166534;
        padding: 0.15rem 0.5rem; border-radius: 0.3rem;
        font-size: 0.75rem; font-weight: 600;
    }

    /* Quality tier colors */
    .tier-a { color: #059669; font-weight: bold; }
    .tier-b { color: #D97706; font-weight: bold; }
    .tier-c { color: #9CA3AF; font-weight: bold; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }

    /* PRISMA flow */
    .prisma-box {
        background: #F1F5F9;
        border: 2px solid #CBD5E1;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin: 0.25rem;
    }
    .prisma-arrow {
        text-align: center;
        font-size: 1.5rem;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">Systematic Review Engine v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">1000+ paper systematic reviews with PRISMA methodology</div>', unsafe_allow_html=True)
st.markdown("---")


# ============================================================
# SIDEBAR - Configuration
# ============================================================
with st.sidebar:
    st.header("Research Configuration")

    # Query input
    query = st.text_area(
        "Research Query",
        value=st.session_state.get('last_query', ''),
        placeholder="e.g., Impact of AI on drug discovery pipelines",
        height=80
    )

    # LLM Provider selection
    st.subheader("LLM Provider")

    # Detect available providers
    try:
        from core.llm_client import LLMClient
        _test_client = LLMClient.__new__(LLMClient)
        _test_client.ollama_url = "http://localhost:11434"
        _test_client.ollama_model = ""
        _test_client.ollama_available = False
        _test_client.deepseek_client = None
        _test_client.gemini_model = None

        # Quick Ollama check
        import requests
        try:
            _r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if _r.status_code == 200:
                _ollama_models = [m['name'] for m in _r.json().get('models', [])]
                _ollama_ok = len(_ollama_models) > 0
            else:
                _ollama_models = []
                _ollama_ok = False
        except:
            _ollama_models = []
            _ollama_ok = False
    except:
        _ollama_ok = False
        _ollama_models = []

    provider_options = []
    if _ollama_ok:
        provider_options.append("ollama (local, free)")
    provider_options.append("gemini (cloud API)")
    provider_options.append("deepseek (cloud API)")

    provider_choice = st.selectbox("Primary LLM", provider_options)
    llm_provider = provider_choice.split(" ")[0]

    # Ollama model selector
    ollama_model = None
    if llm_provider == "ollama" and _ollama_models:
        ollama_model = st.selectbox("Ollama Model", _ollama_models)

    if not _ollama_ok:
        st.caption("Start Ollama for free local LLM: `ollama serve`")

    st.markdown("---")

    # Search parameters
    st.subheader("Search Parameters")

    col_a, col_b = st.columns(2)
    with col_a:
        target_papers = st.number_input("Target Papers", 50, 5000, 500, step=50)
    with col_b:
        max_rounds = st.number_input("Max Rounds", 1, 15, 8)

    enable_citations = st.checkbox("Citation Crawling", value=True)
    enable_clustering = st.checkbox("Topic Clustering", value=True)
    enable_cache = st.checkbox("Cache Results", value=True)

    st.markdown("---")

    # Execute button
    if st.button("Execute Systematic Review", type="primary", width='stretch'):
        if query.strip():
            st.session_state['query'] = query.strip()
            st.session_state['last_query'] = query.strip()
            st.session_state['running'] = True
            st.session_state['config'] = {
                'target_papers': target_papers,
                'max_rounds': max_rounds,
                'enable_citations': enable_citations,
                'enable_clustering': enable_clustering,
                'enable_cache': enable_cache,
                'llm_provider': llm_provider,
                'ollama_model': ollama_model,
            }
            st.rerun()
        else:
            st.warning("Please enter a research query")

    st.markdown("---")

    # Agent architecture info
    st.subheader("Agent Architecture")
    st.markdown("""
    <span class="tier-badge-1">Tier 1</span> Database Executors (6 sources)<br>
    <span class="tier-badge-2">Tier 2</span> Specialist Analysts (Gemini)<br>
    <span class="tier-badge-3">Tier 3</span> Strategic Council (DeepSeek)
    """, unsafe_allow_html=True)

    # LLM status
    st.markdown("---")
    st.subheader("LLM Status")
    if _ollama_ok:
        st.success(f"Ollama: {len(_ollama_models)} models")
    else:
        st.warning("Ollama: offline")
    st.caption("Gemini + DeepSeek: via API keys in .env")


# ============================================================
# MAIN CONTENT - Live Execution Dashboard
# ============================================================
if st.session_state.get('running'):
    config = st.session_state.get('config', {})
    q = st.session_state['query']

    # --- Header ---
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
                padding: 1rem 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem;">
        <span style="color: white; font-size: 1.1rem; font-weight: 600;">
            Researching: {q}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # --- Live metrics row (update in real-time) ---
    metric_cols = st.columns(5)
    metric_phase = metric_cols[0].empty()
    metric_papers = metric_cols[1].empty()
    metric_sources = metric_cols[2].empty()
    metric_elapsed = metric_cols[3].empty()
    metric_status = metric_cols[4].empty()

    metric_phase.metric("Current Phase", "Initializing")
    metric_papers.metric("Papers Found", 0)
    metric_sources.metric("Sources Queried", "0 / 6")
    metric_elapsed.metric("Elapsed", "0s")
    metric_status.metric("Status", "Starting")

    # --- Progress bar ---
    progress_bar = st.progress(0, text="Initializing systematic review...")

    # --- Active Agents Panel ---
    st.markdown("##### Active Sub-Agents")
    agent_panel_slot = st.empty()

    # Agent-to-phase mapping: which agents are active during each phase
    _PHASE_AGENTS = {
        'initialization': [],
        'query_decomposition': [
            ('T3', 'ResearchStrategist', 'DeepSeek'),
        ],
        'baseline_search': [
            ('T1', 'DatabaseQueryAgent', 'Scripted'),
        ],
        'pagination': [
            ('T1', 'DatabaseQueryAgent', 'Scripted'),
        ],
        'citation_crawling': [
            ('T3', 'CitationCrawlStrategyAgent', 'DeepSeek'),
            ('T1', 'DatabaseQueryAgent', 'Scripted'),
        ],
        'deduplication': [
            ('T1', 'DeduplicationAgent', 'Scripted'),
            ('T1', 'PRISMAComplianceAgent', 'Scripted'),
        ],
        'relevance_filtering': [
            ('T2', 'GapDetectionAgent', 'Gemini'),
            ('T3', 'AdaptiveStoppingAgent', 'DeepSeek'),
            ('T2', 'QueryRefinementAgent', 'Gemini'),
            ('T2', 'RelevanceFilterAgent', 'Gemini'),
            ('T1', 'DeduplicationAgent', 'Scripted'),
        ],
        'screening': [
            ('T2', 'ScreeningAgent', 'Gemini'),
            ('T2', 'QualityTierAgent', 'Gemini'),
            ('T1', 'PRISMAComplianceAgent', 'Scripted'),
        ],
        'clustering': [
            ('T2', 'ClusterThemingAgent', 'Gemini'),
        ],
        'map_synthesis': [
            ('T3', 'SynthesisCoordinatorAgent', 'DeepSeek'),
            ('T3', 'PatternSynthesizer', 'DeepSeek'),
        ],
        'reduce_synthesis': [
            ('T3', 'PatternSynthesizer', 'DeepSeek'),
        ],
        'deep_synthesis': [
            ('T3', 'ContradictionAnalyzer', 'DeepSeek'),
            ('T3', 'TemporalEvolutionAnalyzer', 'DeepSeek'),
            ('T3', 'CausalChainExtractor', 'DeepSeek'),
            ('T3', 'ConsensusQuantifier', 'DeepSeek'),
            ('T3', 'PredictiveInsightsGenerator', 'DeepSeek'),
        ],
        'report_generation': [
            ('T3', 'ReportComposerAgent', 'DeepSeek'),
        ],
        'complete': [],
    }

    # Tier badge colors
    _TIER_COLORS = {
        'T1': '#3B82F6',   # blue
        'T2': '#F59E0B',   # amber
        'T3': '#EF4444',   # red
    }

    def _render_agent_panel(phase_name: str):
        """Render the agent activity panel as styled HTML badges."""
        agents = _PHASE_AGENTS.get(phase_name, [])
        if not agents:
            html = '<div style="padding:6px 12px;color:#6B7280;font-size:13px;">No agents active</div>'
        else:
            badges = []
            for tier, name, engine in agents:
                color = _TIER_COLORS.get(tier, '#9CA3AF')
                badges.append(
                    f'<span style="display:inline-block;margin:3px 4px;padding:4px 10px;'
                    f'border-radius:12px;font-size:12px;font-weight:600;'
                    f'background:{color}22;color:{color};border:1px solid {color}55;">'
                    f'{tier} {name} <span style="font-weight:400;opacity:0.7;">({engine})</span>'
                    f'</span>'
                )
            html = f'<div style="padding:4px 0;">{"".join(badges)}</div>'
        agent_panel_slot.markdown(html, unsafe_allow_html=True)

    _render_agent_panel('initialization')

    # --- Live network graph ---
    st.markdown("##### Live Paper Discovery Network")
    graph_slot = st.empty()
    graph_slot.plotly_chart(build_live_graph_figure([]), width='stretch', key="live_graph_init")

    # --- Two-column live view ---
    live_col_left, live_col_right = st.columns([3, 2])

    with live_col_left:
        st.markdown("##### Live Activity Feed")
        activity_container = st.container(height=350)

    with live_col_right:
        st.markdown("##### Source Results")
        source_table_slot = st.empty()
        st.markdown("##### Phase Timeline")
        phase_timeline_slot = st.empty()

    # --- Tracking state for live updates ---
    _live_log = []  # list of (icon, message, timestamp) tuples
    _source_counts = {}  # source_name -> paper_count
    _phase_times = {}  # phase_name -> (start_time, status)
    _discovered_papers = []  # accumulated paper dicts for live graph
    _graph_update_counter = [0]  # mutable counter for graph refresh throttling
    _start_ts = time.monotonic()

    # Phase display names and icons
    _phase_display = {
        'initialization': ('Initializing', '...'),
        'query_decomposition': ('Query Analysis', '...'),
        'baseline_search': ('Database Search', '...'),
        'pagination': ('Fetching More', '...'),
        'citation_crawling': ('Citation Crawling', '...'),
        'deduplication': ('Deduplication', '...'),
        'screening': ('Screening', '...'),
        'relevance_filtering': ('Adaptive Rounds', '...'),
        'clustering': ('Clustering', '...'),
        'map_synthesis': ('Map Synthesis', '...'),
        'reduce_synthesis': ('Reduce Synthesis', '...'),
        'deep_synthesis': ('Deep Synthesis', '...'),
        'report_generation': ('Report Gen', '...'),
        'complete': ('Complete', '...'),
    }

    _phase_icons = {
        'initialization': '....',
        'query_decomposition': '.....',
        'baseline_search': '......',
        'pagination': '.......',
        'citation_crawling': '........',
        'deduplication': '.........',
        'screening': '..........',
        'relevance_filtering': '...........',
        'clustering': '............',
        'map_synthesis': '.............',
        'reduce_synthesis': '..............',
        'deep_synthesis': '...............',
        'report_generation': '................',
        'complete': '.................',
    }

    try:
        from systematic_review import SystematicReviewProtocol

        protocol = SystematicReviewProtocol(
            target_papers=config.get('target_papers', 500),
            max_search_rounds=config.get('max_rounds', 8),
            enable_cache=config.get('enable_cache', True),
            enable_citation_crawl=config.get('enable_citations', True),
            enable_clustering=config.get('enable_clustering', True),
            verbose=True,
            llm_provider=config.get('llm_provider', 'ollama'),
            ollama_model=config.get('ollama_model')
        )

        # --- Hook progress updates to live UI ---
        original_update = protocol.progress.update

        def ui_progress_hook(phase, message, progress=0.0, papers_count=0, detail=None):
            original_update(phase, message, progress, papers_count, detail)

            elapsed = time.monotonic() - _start_ts
            overall = protocol.progress._compute_overall_progress()
            phase_name = phase.value

            # Update progress bar
            progress_bar.progress(
                min(overall, 1.0),
                text=f"{_phase_display.get(phase_name, (phase_name,))[0]}: {message}"
            )

            # Update live metrics
            phase_label = _phase_display.get(phase_name, (phase_name,))[0]
            metric_phase.metric("Current Phase", phase_label)
            metric_papers.metric("Papers Found", f"{papers_count:,}" if papers_count else "0")
            metric_elapsed.metric("Elapsed", f"{elapsed:.0f}s")
            metric_status.metric("Status", f"{overall:.0%}")

            # Update active agents panel
            _render_agent_panel(phase_name)

            # Track source counts from message
            msg_lower = message.lower()
            for src_name in ['semantic_scholar', 'openalex', 'arxiv', 'pubmed', 'crossref', 'ssrn']:
                if src_name in msg_lower or src_name.replace('_', ' ') in msg_lower:
                    if papers_count > 0:
                        _source_counts[src_name] = _source_counts.get(src_name, 0)
                        # Try to extract per-source count from detail
                        if detail and detail.get('source_count'):
                            _source_counts[src_name] = detail['source_count']

            # Count unique sources that have been queried
            if phase_name == 'baseline_search':
                # Estimate sources queried from progress
                n_sources = max(1, int(progress * 6))
                metric_sources.metric("Sources Queried", f"{min(n_sources, 6)} / 6")
            elif phase_name in ('pagination', 'citation_crawling', 'deduplication',
                                'screening', 'relevance_filtering', 'clustering',
                                'map_synthesis', 'reduce_synthesis', 'deep_synthesis',
                                'report_generation', 'complete'):
                metric_sources.metric("Sources Queried", "6 / 6")

            # Track phase timeline
            if phase_name not in _phase_times:
                _phase_times[phase_name] = elapsed
            if progress >= 1.0:
                _phase_times[phase_name + '_done'] = elapsed

            # Add to live activity log
            ts = datetime.now().strftime('%H:%M:%S')
            icon_map = {
                'initialization': 'gear',
                'query_decomposition': 'mag',
                'baseline_search': 'books',
                'pagination': 'page_facing_up',
                'citation_crawling': 'spider_web',
                'deduplication': 'recycle',
                'screening': 'clipboard',
                'relevance_filtering': 'dart',
                'clustering': 'card_file_box',
                'map_synthesis': 'world_map',
                'reduce_synthesis': 'microscope',
                'deep_synthesis': 'dna',
                'report_generation': 'memo',
                'complete': 'white_check_mark',
            }
            icon = icon_map.get(phase_name, 'small_blue_diamond')
            papers_str = f" **[{papers_count:,} papers]**" if papers_count else ""
            log_entry = f":{icon}: `{ts}` **{phase_label}** - {message}{papers_str}"
            _live_log.append(log_entry)

            # Render activity feed (show last 30 entries, newest at top)
            with activity_container:
                # Clear and re-render
                feed_text = "\n\n".join(reversed(_live_log[-30:]))
                activity_container.markdown(feed_text)

            # Update live network graph (throttled - every 3rd event or phase change)
            _graph_update_counter[0] += 1
            if _discovered_papers and (_graph_update_counter[0] % 3 == 0 or progress >= 1.0):
                try:
                    fig = build_live_graph_figure(_discovered_papers)
                    graph_slot.plotly_chart(fig, width='stretch',
                                           key=f"live_graph_{_graph_update_counter[0]}")
                except Exception:
                    pass  # Don't break execution on graph errors

            # Render source results table
            if _source_counts:
                src_data = []
                for s in ['semantic_scholar', 'openalex', 'arxiv', 'pubmed', 'crossref', 'ssrn']:
                    count = _source_counts.get(s, 0)
                    status_icon = 'Done' if count > 0 else ('Searching...' if phase_name == 'baseline_search' else 'Pending')
                    src_data.append({
                        'Source': s.replace('_', ' ').title(),
                        'Papers': count,
                        'Status': status_icon
                    })
                source_table_slot.dataframe(
                    pd.DataFrame(src_data),
                    width='stretch',
                    hide_index=True,
                    height=250
                )

            # Render phase timeline
            tl_data = []
            phases_ordered = [
                'initialization', 'query_decomposition', 'baseline_search',
                'pagination', 'citation_crawling', 'deduplication', 'screening',
                'relevance_filtering', 'clustering', 'map_synthesis',
                'reduce_synthesis', 'deep_synthesis', 'report_generation', 'complete'
            ]
            for ph in phases_ordered:
                display_name = _phase_display.get(ph, (ph,))[0]
                if ph in _phase_times:
                    started = _phase_times[ph]
                    done_time = _phase_times.get(ph + '_done')
                    if done_time is not None:
                        status = 'Done'
                        duration = f"{done_time - started:.1f}s"
                    elif ph == phase_name:
                        status = 'Running'
                        duration = f"{elapsed - started:.1f}s..."
                    else:
                        status = 'Done'
                        duration = '-'
                    tl_data.append({
                        'Phase': display_name,
                        'Status': status,
                        'Duration': duration
                    })
                else:
                    tl_data.append({
                        'Phase': display_name,
                        'Status': 'Pending',
                        'Duration': '-'
                    })
            phase_timeline_slot.dataframe(
                pd.DataFrame(tl_data),
                width='stretch',
                hide_index=True,
                height=250
            )

        protocol.progress.update = ui_progress_hook

        # --- Also track per-source paper counts & collect papers for live graph ---
        for src_name, src_obj in protocol.sources.items():
            _source_counts[src_name] = 0
            original_search = src_obj.search

            def make_tracked_search(name, orig_fn):
                async def tracked_search(*args, **kwargs):
                    result = await orig_fn(*args, **kwargs)
                    if isinstance(result, list):
                        _source_counts[name] = _source_counts.get(name, 0) + len(result)
                        # Collect papers for live graph
                        for p in result:
                            if isinstance(p, dict):
                                _discovered_papers.append(p)
                            elif hasattr(p, 'to_dict'):
                                _discovered_papers.append(p.to_dict())
                    return result
                return tracked_search

            src_obj.search = make_tracked_search(src_name, original_search)

        # --- Execute ---
        results = protocol.execute(q)

        st.session_state['results'] = results
        st.session_state['running'] = False
        st.session_state['protocol'] = protocol
        progress_bar.progress(1.0, text="Systematic review complete!")

        # Final metrics flash
        final_stats = results.get('statistics', {})
        metric_phase.metric("Current Phase", "Complete")
        metric_papers.metric("Papers Found", f"{results.get('total_papers', 0):,}")
        metric_sources.metric("Sources Queried", "6 / 6")
        metric_elapsed.metric("Elapsed", f"{final_stats.get('elapsed_seconds', 0):.0f}s")
        metric_status.metric("Status", "Done")

        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"Error during execution: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state['running'] = False

# ============================================================
# MAIN CONTENT - Results Display
# ============================================================
elif 'results' in st.session_state:
    results = st.session_state['results']
    stats = results.get('statistics', {})
    papers = results.get('papers', [])
    prisma = results.get('prisma_flow', {})
    clusters = results.get('clusters', {})
    synthesis = results.get('synthesis', {})
    quality = results.get('quality_tiers', {})

    # ---- TOP METRICS ROW ----
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Papers Found", stats.get('total_identified', len(papers)))
    c2.metric("Included", stats.get('included', len(papers)))
    c3.metric("Duplicates Removed", stats.get('duplicates_removed', 0))
    c4.metric("Topic Clusters", stats.get('clusters', 0))
    c5.metric("Themes", stats.get('themes_found', 0))
    c6.metric("Time", f"{stats.get('elapsed_seconds', 0):.0f}s")

    st.markdown("---")

    # ---- TABS ----
    tab_report, tab_graph, tab_papers, tab_prisma, tab_clusters, tab_synthesis, tab_scispace, tab_agents, tab_export = st.tabs([
        "Report", "Network Graph", "Papers", "PRISMA Flow",
        "Topic Clusters", "Synthesis", "SciSpace AI", "Agents", "Export"
    ])

    # ========== TAB: REPORT ==========
    with tab_report:
        report = results.get('report', '')
        if report:
            st.markdown(report)
        else:
            st.info("No report generated")

    # ========== TAB: NETWORK GRAPH ==========
    with tab_graph:
        if papers:
            st.subheader("Paper Relationship Network")

            # Controls
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                graph_color_by = st.radio(
                    "Color nodes by",
                    ['source', 'cluster'],
                    horizontal=True
                )
            with gc2:
                graph_max_nodes = st.slider("Max nodes", 50, 500, 200, step=50)
            with gc3:
                graph_mode = st.radio(
                    "Graph engine",
                    ['Plotly (fast)', 'PyVis (interactive)'],
                    horizontal=True
                )

            # Legend
            if graph_color_by == 'source':
                legend_items = ' '.join([
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'border-radius:50%;background:{c};margin-right:4px;"></span>'
                    f'<span style="color:#94A3B8;font-size:0.8rem;margin-right:12px;">'
                    f'{n.replace("_"," ").title()}</span>'
                    for n, c in SOURCE_COLORS.items() if n != 'unknown'
                ])
                st.markdown(f'<div style="margin-bottom:8px;">{legend_items}</div>',
                            unsafe_allow_html=True)
            else:
                st.caption("Colors represent topic clusters. Node size = citation count.")

            if 'Plotly' in graph_mode:
                # Fast Plotly scatter network
                # Add cluster_id to papers if clusters available
                if graph_color_by == 'cluster' and clusters and clusters.get('paper_labels'):
                    labels = clusters['paper_labels']
                    for i, p in enumerate(papers[:len(labels)]):
                        p['cluster_id'] = labels[i]

                fig = build_live_graph_figure(papers, color_by=graph_color_by,
                                              max_nodes=graph_max_nodes)
                fig.update_layout(height=600)
                st.plotly_chart(fig, width='stretch', key="results_graph")
            else:
                # Full interactive PyVis graph
                with st.spinner("Building interactive network..."):
                    html = build_full_network_html(papers, clusters, max_nodes=graph_max_nodes)
                    st.components.v1.html(html, height=600, scrolling=False)

            # Graph stats
            st.markdown("---")
            gs1, gs2, gs3, gs4 = st.columns(4)
            unique_authors = set()
            for p in papers:
                for a in (p.get('authors') or [])[:1]:
                    unique_authors.add(a.lower() if a else '')
            unique_authors.discard('')

            gs1.metric("Papers in Graph", min(len(papers), graph_max_nodes))
            gs2.metric("Unique First Authors", len(unique_authors))
            source_set = set(p.get('source', '') for p in papers)
            source_set.discard('')
            gs3.metric("Sources", len(source_set))
            gs4.metric("Year Span",
                        f"{min(p.get('year', 9999) for p in papers if p.get('year'))}-"
                        f"{max(p.get('year', 0) for p in papers if p.get('year'))}"
                        if any(p.get('year') for p in papers) else "N/A")
        else:
            st.info("No papers available. Run a review first.")

    # ========== TAB: PAPERS ==========
    with tab_papers:
        if papers:
            # Quality tier filter
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                tier_filter = st.multiselect(
                    "Quality Tier",
                    ['A', 'B', 'C', 'All'],
                    default=['All']
                )
            with col_f2:
                source_list = list(set(p.get('source', '') for p in papers if p.get('source')))
                source_filter = st.multiselect("Source", source_list, default=source_list)
            with col_f3:
                year_range = st.slider(
                    "Year Range",
                    min_value=min(p.get('year', 2000) for p in papers if p.get('year')),
                    max_value=max(p.get('year', 2025) for p in papers if p.get('year')),
                    value=(
                        min(p.get('year', 2000) for p in papers if p.get('year')),
                        max(p.get('year', 2025) for p in papers if p.get('year'))
                    )
                )

            # Filter papers
            filtered = papers
            if 'All' not in tier_filter:
                filtered = [p for p in filtered if p.get('quality_tier', 'C') in tier_filter]
            filtered = [p for p in filtered if p.get('source', '') in source_filter]
            filtered = [p for p in filtered if year_range[0] <= (p.get('year') or 0) <= year_range[1]]

            st.caption(f"Showing {len(filtered)} of {len(papers)} papers")

            # DataFrame
            df = pd.DataFrame([{
                'Title': (p.get('title') or '')[:70],
                'Year': p.get('year', ''),
                'Source': p.get('source', ''),
                'Citations': p.get('citation_count', 0) or 0,
                'Score': p.get('llm_relevance_score', p.get('relevance_score', '')),
                'DOI': p.get('doi', ''),
            } for p in filtered[:500]])

            st.dataframe(df, width='stretch', height=400)

            # Charts row
            ch1, ch2 = st.columns(2)

            with ch1:
                years = [p.get('year') for p in filtered if p.get('year')]
                if years:
                    fig_yr = px.histogram(
                        x=years, nbins=20,
                        title="Publication Year Distribution",
                        labels={'x': 'Year', 'y': 'Count'},
                        color_discrete_sequence=['#3B82F6']
                    )
                    fig_yr.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_yr, width='stretch')

            with ch2:
                sources = [p.get('source', 'unknown') for p in filtered]
                if sources:
                    source_counts = pd.Series(sources).value_counts()
                    fig_src = px.pie(
                        values=source_counts.values,
                        names=source_counts.index,
                        title="Papers by Source",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_src.update_layout(height=300)
                    st.plotly_chart(fig_src, width='stretch')
        else:
            st.info("No papers found")

    # ========== TAB: PRISMA FLOW ==========
    with tab_prisma:
        st.subheader("PRISMA 2020 Flow Diagram")

        # Show PRISMA text diagram
        prisma_diagram = results.get('prisma_diagram', '')
        if prisma_diagram:
            st.code(prisma_diagram, language=None)

        # Visual PRISMA metrics
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Identified", prisma.get('identified', 0))
        p2.metric("After Dedup", prisma.get('after_dedup', 0))
        p3.metric("Screened", prisma.get('screened', 0))
        p4.metric("Included", prisma.get('included', 0))

        # Source breakdown
        by_source = prisma.get('by_source', {})
        if by_source:
            st.subheader("Records by Source")
            src_df = pd.DataFrame([
                {'Source': k, 'Papers': v}
                for k, v in sorted(by_source.items(), key=lambda x: -x[1])
            ])
            fig_prisma = px.bar(
                src_df, x='Source', y='Papers',
                color='Papers', color_continuous_scale='Blues',
                title="Records Identified per Database"
            )
            fig_prisma.update_layout(height=300)
            st.plotly_chart(fig_prisma, width='stretch')

        # Exclusion reasons
        exclusions = prisma.get('exclusion_reasons', {})
        if exclusions:
            st.subheader("Exclusion Reasons")
            exc_df = pd.DataFrame([
                {'Reason': k.replace('_', ' ').title(), 'Count': v}
                for k, v in exclusions.items() if v > 0
            ])
            if not exc_df.empty:
                fig_exc = px.bar(exc_df, x='Reason', y='Count',
                                 color_discrete_sequence=['#EF4444'])
                fig_exc.update_layout(height=250)
                st.plotly_chart(fig_exc, width='stretch')

        # Quality tiers
        if quality:
            st.subheader("Quality Tiers")
            qt1, qt2, qt3 = st.columns(3)
            qt1.metric("Tier A (High)", quality.get('A', 0))
            qt2.metric("Tier B (Medium)", quality.get('B', 0))
            qt3.metric("Tier C (Low)", quality.get('C', 0))

    # ========== TAB: TOPIC CLUSTERS ==========
    with tab_clusters:
        if clusters and clusters.get('clusters'):
            cluster_list = clusters['clusters']
            st.subheader(f"{len(cluster_list)} Topic Clusters Found")

            # Cluster overview chart
            cluster_df = pd.DataFrame([{
                'Cluster': c.get('label', f"Cluster {i}")[:40],
                'Papers': c.get('size', 0),
                'Avg Citations': c.get('avg_citations', 0),
            } for i, c in enumerate(cluster_list)])

            fig_cl = px.bar(
                cluster_df, x='Cluster', y='Papers',
                color='Avg Citations', color_continuous_scale='Viridis',
                title="Cluster Size & Citation Impact"
            )
            fig_cl.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig_cl, width='stretch')

            # Detailed cluster cards
            for i, c in enumerate(cluster_list):
                with st.expander(
                    f"{c.get('label', f'Cluster {i}')} ({c.get('size', 0)} papers)",
                    expanded=(i < 3)
                ):
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Papers", c.get('size', 0))
                    mc2.metric("Year Range", c.get('year_range', 'N/A'))
                    mc3.metric("Avg Citations", f"{c.get('avg_citations', 0):.0f}")

                    # Show centroid paper if available
                    centroid = c.get('centroid_paper')
                    if centroid and isinstance(centroid, dict):
                        st.caption(f"Representative paper: *{centroid.get('title', '')}*")
        else:
            st.info("Clustering was not enabled or not enough papers for clustering.")

    # ========== TAB: SYNTHESIS ==========
    with tab_synthesis:
        final = synthesis.get('final_synthesis', {})
        reduced = synthesis.get('reduced_synthesis', {})
        chunk_summaries = synthesis.get('chunk_summaries', [])

        if final:
            # Executive Summary
            st.subheader("Executive Summary")
            st.markdown(final.get('executive_summary', '*No summary available*'))

            st.markdown("---")

            # Key Findings
            st.subheader("Key Findings")
            findings = final.get('key_findings', [])
            for i, f in enumerate(findings[:10], 1):
                finding_text = f if isinstance(f, str) else f.get('finding', str(f))
                evidence = f.get('evidence_strength', '') if isinstance(f, dict) else ''
                st.markdown(f"**{i}.** {finding_text}")
                if evidence:
                    st.caption(f"Evidence: {evidence}")

            st.markdown("---")

            # Major Themes
            st.subheader("Major Themes")
            themes = reduced.get('major_themes', [])
            if themes:
                theme_data = []
                for t in themes[:12]:
                    theme_data.append({
                        'Theme': t.get('theme', 'Unknown')[:50],
                        'Papers': t.get('paper_count_est', 0),
                        'Prevalence': t.get('prevalence', 'N/A'),
                    })
                theme_df = pd.DataFrame(theme_data)
                if not theme_df.empty and 'Papers' in theme_df.columns:
                    fig_th = px.bar(
                        theme_df, x='Theme', y='Papers',
                        title="Theme Prevalence",
                        color_discrete_sequence=['#8B5CF6']
                    )
                    fig_th.update_layout(height=300, xaxis_tickangle=-45)
                    st.plotly_chart(fig_th, width='stretch')

                for t in themes[:10]:
                    with st.expander(t.get('theme', 'Unknown')):
                        st.write(t.get('description', ''))
                        st.caption(f"Prevalence: {t.get('prevalence', 'N/A')} | "
                                   f"Est. papers: {t.get('paper_count_est', 'N/A')}")

            st.markdown("---")

            # Debates & Contradictions
            st.subheader("Unresolved Debates")
            debates = final.get('unresolved_debates', [])
            for d in debates[:5]:
                with st.expander(d.get('debate', 'Unknown debate')):
                    for side in d.get('sides', []):
                        st.write(f"- {side}")
                    st.caption(f"Evidence: {d.get('current_evidence', 'N/A')}")

            # Future Directions
            st.subheader("Future Research Directions")
            for fd in final.get('future_directions', [])[:5]:
                st.write(f"- {fd}")

            # Map-Reduce stats
            if chunk_summaries:
                st.markdown("---")
                st.caption(f"Synthesis processed {len(chunk_summaries)} chunks via Map-Reduce pipeline")
        else:
            st.info("No synthesis results available. Run a review first.")

    # ========== TAB: SCISPACE AI ==========
    with tab_scispace:
        st.subheader("SciSpace-Equivalent AI Capabilities")
        st.markdown("""
        These features replicate [SciSpace](https://scispace.com/) capabilities locally using your own LLMs.
        """)

        sci_col1, sci_col2 = st.columns(2)

        with sci_col1:
            st.markdown("#### Semantic Search")
            st.markdown("""
            Vector-based paper matching using sentence-transformers.
            Finds papers by *meaning*, not just keywords.
            """)
            semantic_query = st.text_input(
                "Semantic search query",
                placeholder="e.g., mechanisms of neural plasticity in aging",
                key="semantic_q"
            )
            if semantic_query and papers:
                try:
                    from core.semantic_search import SemanticSearchEngine
                    engine = SemanticSearchEngine()
                    engine.index_papers(papers)
                    sem_results = engine.search(semantic_query, top_k=10)
                    if sem_results:
                        st.success(f"Found {len(sem_results)} semantically similar papers")
                        for i, p in enumerate(sem_results, 1):
                            score = p.get('semantic_score', 0)
                            title = p.get('title', 'Unknown')[:80]
                            st.markdown(f"**{i}.** [{title}] (score: {score:.3f})")
                    else:
                        st.info("No papers matched above the similarity threshold.")
                except Exception as e:
                    st.warning(f"Semantic search error: {e}")

        with sci_col2:
            st.markdown("#### Paper Chat (Q&A)")
            st.markdown("""
            Ask questions about any paper in your corpus.
            Powered by DeepSeek/Gemini with full-text context.
            """)
            if papers:
                paper_titles = [f"{p.get('title', 'Unknown')[:70]}" for p in papers[:50]]
                selected_idx = st.selectbox(
                    "Select paper to chat with",
                    range(len(paper_titles)),
                    format_func=lambda i: paper_titles[i],
                    key="chat_paper_select"
                )
                chat_question = st.text_input(
                    "Ask a question about this paper",
                    placeholder="e.g., What methodology did they use?",
                    key="chat_q"
                )
                if chat_question and selected_idx is not None:
                    try:
                        from agents.tier3.scispace_agents import PaperChatAgent
                        chat_agent = PaperChatAgent()
                        result = chat_agent.execute({
                            'paper': papers[selected_idx],
                            'question': chat_question
                        })
                        st.markdown(f"**Answer:** {result.get('answer', 'No answer generated')}")
                        if result.get('confidence'):
                            st.caption(f"Confidence: {result['confidence']}")
                        follow_ups = result.get('follow_up_questions', [])
                        if follow_ups:
                            st.markdown("**Follow-up questions:**")
                            for fq in follow_ups[:3]:
                                st.markdown(f"- {fq}")
                    except Exception as e:
                        st.warning(f"Paper chat error: {e}")

        st.markdown("---")
        st.markdown("#### Deep Review (Multi-Pass Analysis)")
        st.markdown("""
        Iterative 5-pass literature review that progressively deepens analysis:
        Theme Scan -> Deep Dive -> Cross-Synthesis -> Evidence Assessment -> Gap Mapping
        """)
        dr_col1, dr_col2 = st.columns([3, 1])
        with dr_col1:
            deep_topic = st.text_input(
                "Deep review topic",
                value=results.get('query', ''),
                key="deep_review_topic"
            )
        with dr_col2:
            review_depth = st.slider("Depth (passes)", 1, 5, 3, key="review_depth")

        if st.button("Run Deep Review", key="run_deep_review"):
            if papers and deep_topic:
                with st.spinner(f"Running {review_depth}-pass deep review..."):
                    try:
                        from agents.tier3.scispace_agents import DeepReviewAgent
                        reviewer = DeepReviewAgent()
                        dr_result = reviewer.execute({
                            'papers': papers[:50],
                            'topic': deep_topic,
                            'depth': review_depth
                        })
                        st.success(f"Deep review complete! {dr_result.get('passes_completed', 0)} passes.")

                        # Show themes
                        themes = dr_result.get('themes', [])
                        if themes:
                            st.markdown("**Major Themes:**")
                            for t in themes:
                                if isinstance(t, dict):
                                    st.markdown(f"- **{t.get('theme', t.get('name', 'Unknown'))}**: "
                                                f"{t.get('description', t.get('summary', ''))[:120]}")
                                elif isinstance(t, str):
                                    st.markdown(f"- {t}")

                        # Show gaps
                        gaps = dr_result.get('knowledge_gaps', {})
                        if isinstance(gaps, dict):
                            gap_list = gaps.get('gaps', [])
                        elif isinstance(gaps, list):
                            gap_list = gaps
                        else:
                            gap_list = []
                        if gap_list:
                            st.markdown("**Knowledge Gaps:**")
                            for g in gap_list[:5]:
                                if isinstance(g, dict):
                                    st.markdown(f"- {g.get('gap', g.get('description', str(g)))[:120]}")
                                elif isinstance(g, str):
                                    st.markdown(f"- {g}")

                    except Exception as e:
                        st.error(f"Deep review error: {e}")
            else:
                st.warning("Need papers and a topic to run deep review.")

        st.markdown("---")
        st.markdown("#### Capabilities Summary")
        cap_c1, cap_c2, cap_c3, cap_c4 = st.columns(4)
        cap_c1.metric("Paper Sources", "7+", help="Semantic Scholar, OpenAlex, arXiv, PubMed, CrossRef, SSRN, CORE")
        cap_c2.metric("Open Access Papers", "140M+", help="Via CORE API")
        cap_c3.metric("AI Agents", "22", help="3 Tier-1 + 6 Tier-2 + 13 Tier-3")
        cap_c4.metric("Review Depth", "5 passes", help="Theme->Dive->Cross->Evidence->Gaps")

    # ========== TAB: AGENTS ==========
    with tab_agents:
        st.subheader("20-Agent Architecture")
        st.markdown("This review was powered by a **3-tier multi-agent system** "
                     "with 20 specialized agents.")

        # Tier 1
        st.markdown("#### Tier 1 - Scripted Executors (No LLM)")
        t1_data = [
            {"Agent": "DatabaseQueryAgent", "Role": "Execute searches across 6 academic databases",
             "Engine": "Scripted", "Phase": "Search, Pagination, Refinement"},
            {"Agent": "DeduplicationAgent", "Role": "6-strategy fuzzy matching deduplication",
             "Engine": "Scripted", "Phase": "Deduplication"},
            {"Agent": "PRISMAComplianceAgent", "Role": "PRISMA methodology enforcement",
             "Engine": "Scripted", "Phase": "Screening, Deduplication"},
        ]
        st.dataframe(pd.DataFrame(t1_data), width='stretch', hide_index=True)

        # Tier 2
        st.markdown("#### Tier 2 - Specialist Analysts (Gemini Primary)")
        t2_data = [
            {"Agent": "GapDetectionAgent", "Role": "Identify coverage gaps across dimensions",
             "Engine": "Gemini", "Phase": "Adaptive Rounds"},
            {"Agent": "QueryRefinementAgent", "Role": "Generate targeted refinement queries",
             "Engine": "Gemini", "Phase": "Adaptive Rounds"},
            {"Agent": "RelevanceFilterAgent", "Role": "LLM-based relevance scoring",
             "Engine": "Gemini", "Phase": "Adaptive Rounds"},
            {"Agent": "ScreeningAgent", "Role": "Inclusion/exclusion screening",
             "Engine": "Gemini", "Phase": "Screening"},
            {"Agent": "QualityTierAgent", "Role": "A/B/C evidence quality tiers",
             "Engine": "Gemini", "Phase": "Screening"},
            {"Agent": "ClusterThemingAgent", "Role": "Semantic cluster labeling",
             "Engine": "Gemini", "Phase": "Clustering"},
        ]
        st.dataframe(pd.DataFrame(t2_data), width='stretch', hide_index=True)

        # Tier 3
        st.markdown("#### Tier 3 - Strategic Council (DeepSeek Primary)")
        t3_data = [
            {"Agent": "ResearchStrategist", "Role": "Query decomposition and dimension analysis",
             "Engine": "DeepSeek", "Phase": "Query Decomposition"},
            {"Agent": "PatternSynthesizer", "Role": "Cross-cutting pattern identification",
             "Engine": "DeepSeek", "Phase": "Map/Reduce Synthesis"},
            {"Agent": "ContradictionAnalyzer", "Role": "Detect conflicting findings",
             "Engine": "DeepSeek", "Phase": "Deep Synthesis"},
            {"Agent": "TemporalEvolutionAnalyzer", "Role": "Track emerging/declining themes",
             "Engine": "DeepSeek", "Phase": "Deep Synthesis"},
            {"Agent": "CausalChainExtractor", "Role": "Map causal relationships",
             "Engine": "DeepSeek", "Phase": "Deep Synthesis"},
            {"Agent": "ConsensusQuantifier", "Role": "Measure agreement levels",
             "Engine": "DeepSeek", "Phase": "Deep Synthesis"},
            {"Agent": "PredictiveInsightsGenerator", "Role": "Forecast research directions",
             "Engine": "DeepSeek", "Phase": "Deep Synthesis"},
            {"Agent": "AdaptiveStoppingAgent", "Role": "Multi-factor search termination",
             "Engine": "DeepSeek", "Phase": "Adaptive Rounds"},
            {"Agent": "SynthesisCoordinatorAgent", "Role": "Plan which agents to activate",
             "Engine": "DeepSeek", "Phase": "Map Synthesis"},
            {"Agent": "ReportComposerAgent", "Role": "Generate titles, summaries, recommendations",
             "Engine": "DeepSeek", "Phase": "Report Generation"},
            {"Agent": "CitationCrawlStrategyAgent", "Role": "Optimize seed selection and crawl depth",
             "Engine": "DeepSeek", "Phase": "Citation Crawling"},
        ]
        st.dataframe(pd.DataFrame(t3_data), width='stretch', hide_index=True)

        # Summary metrics
        st.markdown("---")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Tier 1 Agents", "3", help="Scripted, no LLM")
        ac2.metric("Tier 2 Agents", "6", help="Gemini primary")
        ac3.metric("Tier 3 Agents", "11", help="DeepSeek primary")
        ac4.metric("Total Agents", "20")

    # ========== TAB: EXPORT ==========
    with tab_export:
        st.subheader("Export Results")

        ex1, ex2, ex3 = st.columns(3)

        # Export report as markdown
        with ex1:
            report_text = results.get('report', '')
            if report_text:
                st.download_button(
                    "Download Report (.md)",
                    data=report_text,
                    file_name=f"systematic_review_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    width='stretch'
                )

        # Export data as JSON
        with ex2:
            export_data = {
                'query': results.get('query', ''),
                'statistics': stats,
                'prisma_flow': prisma,
                'quality_tiers': quality,
                'synthesis': synthesis,
                'papers': papers[:200],
            }
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "Download Data (.json)",
                data=json_str,
                file_name=f"review_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                width='stretch'
            )

        # Export papers list as CSV
        with ex3:
            if papers:
                csv_df = pd.DataFrame([{
                    'Title': p.get('title', ''),
                    'Authors': ', '.join(p.get('authors', [])[:3]),
                    'Year': p.get('year', ''),
                    'Source': p.get('source', ''),
                    'Citations': p.get('citation_count', 0),
                    'DOI': p.get('doi', ''),
                    'URL': p.get('url', ''),
                } for p in papers])
                st.download_button(
                    "Download Papers (.csv)",
                    data=csv_df.to_csv(index=False),
                    file_name=f"papers_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width='stretch'
                )

        # Timeline
        st.markdown("---")
        st.subheader("Execution Timeline")
        timeline = results.get('progress_timeline', [])
        if timeline:
            tl_df = pd.DataFrame(timeline)
            st.dataframe(tl_df, width='stretch', height=250)

        # Cache stats
        cache_stats = stats.get('cache_stats', {})
        if cache_stats:
            st.subheader("Cache Statistics")
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Cached Papers", cache_stats.get('cached_papers', 0))
            cs2.metric("API Cache Entries", cache_stats.get('api_cache_entries', 0))
            cs3.metric("DB Size", f"{cache_stats.get('db_size_mb', 0):.1f} MB")

# ============================================================
# WELCOME STATE
# ============================================================
else:
    st.markdown("""
    ### Welcome to the Systematic Review Engine v2.0 + SciSpace AI

    This engine performs **PhD-level systematic literature reviews** at scale,
    processing up to **1000+ papers** with full PRISMA methodology.
    Now with **SciSpace-equivalent AI capabilities** built in.

    ---

    **13 Integrated Features:**

    | # | Feature | Description |
    |---|---------|-------------|
    | 1 | Parallel Search | 7 databases searched simultaneously |
    | 2 | Bulk Harvesting | Pagination fetches hundreds per source |
    | 3 | Map-Reduce Synthesis | Handles 1000+ papers via chunked analysis |
    | 4 | Topic Clustering | Groups papers by semantic similarity |
    | 5 | Smart Caching | SQLite cache avoids duplicate API calls |
    | 6 | Adaptive Rounds | Stops when coverage plateaus |
    | 7 | Citation Crawling | Snowball sampling via reference networks |
    | 8 | Full-Text Access | PDF extraction via CORE/Unpaywall/arXiv/PMC |
    | 9 | Progress Tracking | Real-time phase-by-phase progress |
    | 10 | PRISMA Flow | Full PRISMA 2020 compliance tracking |
    | 11 | Semantic Search | Vector-based paper matching (SciSpace) |
    | 12 | Paper Chat | Q&A over individual papers (SciSpace) |
    | 13 | Deep Review | 5-pass iterative literature analysis (SciSpace) |

    ---

    **22 AI Agents across 3 Tiers:**
    - **Tier 1** (3 agents) - Scripted executors (no LLM)
    - **Tier 2** (6 agents) - Specialist analysts (Gemini primary)
    - **Tier 3** (13 agents) - Strategic council + SciSpace AI (DeepSeek primary)

    **Supported LLM Providers:**
    - **Ollama** (local) - Free, unlimited, private. Best for bulk work.
    - **Gemini** (cloud) - 1M token context. Great for final synthesis.
    - **DeepSeek** (cloud) - Strong reasoning. Good for analysis.

    ---

    **Enter a research query in the sidebar to begin.**
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"Systematic Review Engine v2.0 + SciSpace AI | "
    f"22 Agents | 7 Sources | "
    f"Ollama {'online' if _ollama_ok else 'offline'} | "
    f"Powered by Ollama + DeepSeek + Gemini"
)
