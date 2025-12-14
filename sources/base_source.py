"""
Base class for academic data sources.
All source implementations should inherit from this.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """Standardized paper representation across all sources"""
    
    # Core identifiers
    paper_id: str
    title: str
    
    # Authors
    authors: List[str] = field(default_factory=list)
    
    # Publication info
    year: int = 0
    venue: str = ""
    venue_type: str = ""  # journal, conference, preprint
    
    # Content
    abstract: str = ""
    
    # Identifiers
    doi: str = ""
    arxiv_id: str = ""
    pubmed_id: str = ""
    url: str = ""
    
    # Metrics
    citation_count: int = 0
    influential_citation_count: int = 0
    
    # Quality indicators
    quality_score: float = 0.0
    verified: bool = False
    
    # Source tracking
    source: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'venue_type': self.venue_type,
            'abstract': self.abstract,
            'doi': self.doi,
            'arxiv_id': self.arxiv_id,
            'pubmed_id': self.pubmed_id,
            'url': self.url,
            'citation_count': self.citation_count,
            'influential_citation_count': self.influential_citation_count,
            'quality_score': self.quality_score,
            'verified': self.verified,
            'source': self.source,
            'retrieved_at': self.retrieved_at
        }
    
    def format_apa(self) -> str:
        """Format as APA citation"""
        if len(self.authors) == 0:
            author_str = "Unknown Author"
        elif len(self.authors) > 3:
            author_str = f"{self.authors[0]} et al."
        else:
            author_str = ", ".join(self.authors)
        
        year = self.year if self.year else "n.d."
        venue = f" {self.venue}." if self.venue else ""
        doi = f" https://doi.org/{self.doi}" if self.doi else ""
        
        return f"{author_str} ({year}). {self.title}.{venue}{doi}"


class BaseSource(ABC):
    """Abstract base class for academic data sources"""
    
    def __init__(self, rate_limit: int = 10):
        """
        Args:
            rate_limit: Maximum requests per minute
        """
        self.rate_limit = rate_limit
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> List[Paper]:
        """
        Search for papers matching query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of Paper objects
        """
        pass
    
    @abstractmethod
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Get a specific paper by ID.
        
        Args:
            paper_id: Source-specific paper identifier
            
        Returns:
            Paper object or None if not found
        """
        pass
    
    def _validate_response(self, response) -> bool:
        """Validate API response"""
        return response is not None and response.status_code == 200
