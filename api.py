"""
FastAPI endpoint for React integration.

Run with: python api.py

Endpoints:
- GET /search - Regular search (rule-based)
- GET /llm-search - LLM-powered search (AI intelligence)
- GET /health - Health check
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os

from agent import ResearchAgent

# Initialize FastAPI app
app = FastAPI(
    title="Academic Research Agent API",
    description="Domain-agnostic academic paper search with optional LLM intelligence",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
agent = ResearchAgent()
llm_agent = None  # Lazy load

def get_llm_agent():
    """Lazy load LLM agent (requires API keys)"""
    global llm_agent
    if llm_agent is None:
        try:
            from llm_agent import LLMPoweredAgent
            llm_agent = LLMPoweredAgent(primary_llm="gemini", fallback_llm="deepseek")
        except Exception as e:
            print(f"⚠️ LLM Agent not available: {e}")
            return None
    return llm_agent


# Multi-agent protocol (lazy load)
multi_agent_protocol = None

def get_multi_agent_protocol():
    """Lazy load Multi-Agent Protocol (3-tier system)"""
    global multi_agent_protocol
    if multi_agent_protocol is None:
        try:
            from multi_agent_protocol import MultiAgentProtocol
            multi_agent_protocol = MultiAgentProtocol(max_refinement_rounds=3, verbose=False)
        except Exception as e:
            print(f"⚠️ Multi-Agent Protocol not available: {e}")
            return None
    return multi_agent_protocol


@app.get("/")
async def root():
    """Root endpoint with API info"""
    llm_available = os.getenv("GEMINI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    return {
        "name": "Academic Research Agent API",
        "version": "2.0.0",
        "llm_available": bool(llm_available),
        "endpoints": {
            "/search": "Regular search (rule-based, fast)",
            "/llm-search": "LLM-powered search (AI intelligence, slower)",
            "/multi-agent-search": "Multi-agent search (3-tier agent system)",
            "/health": "Health check",
            "/analyze": "Analyze query structure"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    llm_available = bool(os.getenv("GEMINI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    return {
        "status": "healthy",
        "agent": "initialized",
        "llm_available": llm_available
    }


@app.get("/search")
async def search_papers(
    query: str = Query(..., description="Research query"),
    limit: int = Query(50, ge=1, le=200, description="Maximum papers"),
    min_quality: float = Query(0.3, ge=0, le=1, description="Minimum quality score")
):
    """
    Regular search (rule-based) - Fast but less intelligent.
    """
    try:
        result = agent.search(
            query=query,
            limit=limit,
            min_quality=min_quality
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm-search")
async def llm_search_papers(
    query: str = Query(..., description="Research query"),
    limit: int = Query(30, ge=1, le=100, description="Maximum papers"),
    min_relevance: int = Query(30, ge=0, le=100, description="Minimum LLM relevance score")
):
    """
    LLM-powered search - Uses AI for domain detection, filtering, and scoring.
    Requires GEMINI_API_KEY or DEEPSEEK_API_KEY in environment.
    """
    llm = get_llm_agent()
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM agent not available. Set GEMINI_API_KEY or DEEPSEEK_API_KEY in .env"
        )
    
    try:
        result = llm.search(
            query=query,
            max_papers=limit,
            min_relevance=min_relevance,
            verbose=True
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/multi-agent-search")
async def multi_agent_search(
    query: str = Query(..., description="Research query"),
    max_rounds: int = Query(3, ge=1, le=5, description="Maximum refinement rounds"),
    domain: str = Query(None, description="Research domain (optional)")
):
    """
    Multi-Agent search - Uses 3-tier agent system:
    - Tier 1: Database executors (scripted)
    - Tier 2: Specialist analysts (Gemini)
    - Tier 3: Strategic council (DeepSeek)
    
    Returns papers with cross-cutting patterns and coverage analysis.
    """
    protocol = get_multi_agent_protocol()
    if protocol is None:
        raise HTTPException(
            status_code=503,
            detail="Multi-agent protocol not available. Check API keys."
        )
    
    try:
        # Update max rounds
        protocol.max_rounds = max_rounds
        
        # Execute
        result = protocol.execute(query=query, domain=domain)
        
        return {
            "query": query,
            "papers": result['papers'],
            "patterns": result['patterns'],
            "statistics": result['statistics'],
            "decomposition": result.get('decomposition', {}),
            "coverage": result.get('coverage', {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze")
async def analyze_query(
    query: str = Query(..., description="Research query to analyze"),
    use_llm: bool = Query(False, description="Use LLM for analysis")
):
    """
    Analyze query without searching.
    Optionally use LLM for smarter analysis.
    """
    try:
        if use_llm:
            llm = get_llm_agent()
            if llm:
                classification = llm.domain_classifier.classify(query)
                filters = llm.filter_generator.generate_filters(query, classification)
                return {
                    "query": query,
                    "method": "llm",
                    "classification": classification,
                    "filters": filters
                }
        
        # Fallback to rule-based
        from core.query_analyzer import QueryAnalyzer
        from core.search_strategy import SearchStrategy
        
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze(query)
        strategy = SearchStrategy(analysis)
        search_plan = strategy.generate()
        
        return {
            "query": query,
            "method": "rule-based",
            "analysis": analysis,
            "search_plan": search_plan
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/format")
async def format_for_llm(
    query: str = Query(..., description="Research query"),
    limit: int = Query(30, description="Maximum papers"),
    use_llm: bool = Query(False, description="Use LLM-powered search")
):
    """
    Search and return results formatted for LLM consumption.
    """
    try:
        if use_llm:
            llm = get_llm_agent()
            if llm:
                result = llm.search(query=query, max_papers=limit)
                formatted = llm.format_for_llm(result)
                return {
                    "query": query,
                    "method": "llm",
                    "paper_count": len(result['papers']),
                    "formatted_text": formatted
                }
        
        # Fallback to regular
        result = agent.search(query=query, limit=limit)
        formatted = agent.format_for_llm(result)
        
        return {
            "query": query,
            "method": "rule-based",
            "paper_count": len(result['papers']),
            "formatted_text": formatted
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Academic Research Agent API v2.0")
    print("="*60)
    print("\nEndpoints:")
    print("  • http://localhost:8000/search      (rule-based)")
    print("  • http://localhost:8000/llm-search  (AI-powered)")
    print("  • http://localhost:8000/docs        (documentation)")
    print("\n" + "="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

