"""
Multi-Agent Research Protocol Dashboard
Streamlit-based visualization for the research system.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Research Agent Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
    }
    .tier-badge-1 { background-color: #DBEAFE; color: #1E40AF; padding: 0.25rem 0.75rem; border-radius: 0.5rem; }
    .tier-badge-2 { background-color: #FEF3C7; color: #92400E; padding: 0.25rem 0.75rem; border-radius: 0.5rem; }
    .tier-badge-3 { background-color: #DCFCE7; color: #166534; padding: 0.25rem 0.75rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 Multi-Agent Research Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🚀 New Research")
    
    query = st.text_area(
        "Research Query",
        value="",
        placeholder="e.g., Cryptocurrency liquidation cascades in DeFi"
    )
    
    domain = st.selectbox(
        "Domain (optional)",
        ["auto", "economics", "computer_science", "medicine", "physics"]
    )
    
    max_rounds = st.slider("Max Refinement Rounds", 1, 5, 3)
    
    if st.button("🔍 Execute Research", type="primary"):
        if query:
            st.session_state['query'] = query
            st.session_state['running'] = True
        else:
            st.warning("Please enter a query")
    
    st.markdown("---")
    st.markdown("### Agent Status")
    st.markdown("""
    <span class="tier-badge-1">Tier 1</span> Database Executor<br>
    <span class="tier-badge-2">Tier 2</span> Specialist Analysts<br>
    <span class="tier-badge-3">Tier 3</span> Strategic Council
    """, unsafe_allow_html=True)

# Main content
if 'running' in st.session_state and st.session_state['running']:
    st.info(f"🔄 Executing research for: **{st.session_state['query']}**")
    
    with st.spinner("Running multi-agent protocol..."):
        try:
            from multi_agent_protocol import MultiAgentProtocol
            
            protocol = MultiAgentProtocol(max_refinement_rounds=max_rounds, verbose=False)
            results = protocol.execute(
                query=st.session_state['query'],
                domain=domain if domain != "auto" else None
            )
            
            st.session_state['results'] = results
            st.session_state['running'] = False
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state['running'] = False

elif 'results' in st.session_state:
    results = st.session_state['results']
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Papers", len(results['papers']))
    
    with col2:
        coverage = results.get('coverage', {}).get('coverage_score', 0)
        st.metric("📊 Coverage", f"{coverage:.0f}%" if coverage else "N/A")
    
    with col3:
        st.metric("🔍 Patterns Found", len(results.get('patterns', [])))
    
    with col4:
        stats = results.get('statistics', {})
        st.metric("⏱️ Time", f"{stats.get('elapsed_seconds', 0):.1f}s")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📚 Papers", "🔍 Patterns", "📊 Analysis"])
    
    with tab1:
        st.subheader("Found Papers")
        
        papers = results.get('papers', [])
        if papers:
            # Convert to DataFrame
            df = pd.DataFrame([{
                'Title': p.get('title', '')[:60] + '...',
                'Year': p.get('year', 'N/A'),
                'Source': p.get('source', 'Unknown'),
                'Score': p.get('relevance_score', 'N/A'),
                'Citations': p.get('citation_count', 0)
            } for p in papers[:50]])
            
            st.dataframe(df, use_container_width=True)
            
            # Year distribution
            years = [p.get('year') for p in papers if p.get('year')]
            if years:
                fig = px.histogram(x=years, nbins=15, title="Papers by Year")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No papers found")
    
    with tab2:
        st.subheader("Cross-Cutting Patterns")
        
        patterns = results.get('patterns', [])
        if patterns:
            for i, pattern in enumerate(patterns, 1):
                with st.expander(f"Pattern {i}: {pattern.get('name', 'Unknown')}"):
                    st.write(pattern.get('description', ''))
                    st.success(f"💡 **Insight:** {pattern.get('insight', '')}")
                    st.write(f"Confidence: {pattern.get('confidence', 0):.0%}")
        else:
            st.info("No patterns identified yet")
    
    with tab3:
        st.subheader("Query Decomposition")
        
        decomp = results.get('decomposition', {})
        if decomp:
            st.write("**Sub-Questions:**")
            for sq in decomp.get('sub_questions', []):
                priority = sq.get('priority', 'medium')
                icon = "🔴" if priority == "critical" else "🟡" if priority == "high" else "🟢"
                st.write(f"{icon} {sq.get('question', '')}")
            
            st.write("**Dimensions:**")
            dims = decomp.get('dimensions', {})
            if dims:
                for dim_name, values in dims.items():
                    st.write(f"• **{dim_name}**: {', '.join(values[:5])}")

else:
    st.markdown("""
    ### Welcome to the Multi-Agent Research Dashboard
    
    This dashboard provides visualization and control for the 3-tier research agent system:
    
    - **Tier 1** (Scripted): Database executors for Semantic Scholar, arXiv, CrossRef, PubMed, SSRN
    - **Tier 2** (Gemini): Gap detection, query refinement, relevance filtering
    - **Tier 3** (DeepSeek): Query decomposition, pattern synthesis
    
    👈 **Enter a research query in the sidebar to get started!**
    """)

# Footer
st.markdown("---")
st.caption("Multi-Agent Research Protocol Dashboard | Powered by DeepSeek + Gemini")
