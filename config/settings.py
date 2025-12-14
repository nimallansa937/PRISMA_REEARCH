from pydantic_settings import BaseSettings
from typing import Dict, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Global configuration for the Academic Research Agent"""
    
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    CACHE_DIR: Path = BASE_DIR / ".cache"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # API Keys (from .env)
    SEMANTIC_SCHOLAR_API_KEY: str = ""  # Optional, increases rate limit
    SERP_API_KEY: str = ""              # For Google Scholar
    NCBI_API_KEY: str = ""              # For PubMed
    
    # Rate limits (requests per minute)
    SEMANTIC_SCHOLAR_RATE_LIMIT: int = 100
    ARXIV_RATE_LIMIT: int = 20
    PUBMED_RATE_LIMIT: int = 10
    CROSSREF_RATE_LIMIT: int = 50
    
    # Search parameters
    MIN_PAPERS_REQUIRED: int = 50
    MAX_PAPERS_PER_SOURCE: int = 100
    MIN_CITATION_COUNT: int = 0  # Minimum citations for inclusion
    MIN_YEAR: int = 2000          # Only papers from this year onwards
    
    # Quality thresholds
    MIN_QUALITY_SCORE: float = 0.3  # 0-1 scale
    REQUIRE_DOI: bool = False        # Strict mode requires DOI
    
    # Caching
    ENABLE_CACHE: bool = True
    CACHE_EXPIRY_DAYS: int = 7
    
    # Iterative search settings
    MAX_SEARCH_ATTEMPTS: int = 4
    BROADEN_THRESHOLD: int = 20  # Broaden search if fewer than this
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Create singleton instance
settings = Settings()

# Ensure directories exist
settings.CACHE_DIR.mkdir(exist_ok=True)
settings.LOGS_DIR.mkdir(exist_ok=True)
