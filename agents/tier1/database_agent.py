"""
Tier 1 Database Query Agent - Scripted executor (no LLM).
Executes searches across multiple academic databases.
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import Tier1Agent
from sources.semantic_scholar import SemanticScholar
from sources.arxiv import ArXiv
from sources.crossref import CrossRef
from sources.pubmed import PubMed
from sources.ssrn import SSRN
from sources.openalex import OpenAlex
from sources.core_api import COREApi


class DatabaseQueryAgent(Tier1Agent):
    """
    Executes searches across academic databases.
    No LLM - purely scripted execution.
    """

    def __init__(self):
        super().__init__(
            name="DatabaseQuery",
            description="Searches academic databases"
        )

        # Initialize data sources (7 sources now!)
        self.sources = {
            'semantic_scholar': SemanticScholar(),
            'arxiv': ArXiv(),
            'crossref': CrossRef(),
            'pubmed': PubMed(),
            'ssrn': SSRN(),
            'openalex': OpenAlex(),
            'core': COREApi(),
        }
        
        self.search_history: List[Dict] = []
    
    def execute(self, input_data: Dict) -> Dict:
        """Execute search across databases"""
        query = input_data.get('query', '')
        databases = input_data.get('databases', ['semantic_scholar', 'arxiv'])
        limit_per_source = input_data.get('limit', 25)
        
        papers = self.search_all(query, databases, limit_per_source)
        
        return {
            'papers': papers,
            'total_found': len(papers),
            'databases_searched': databases
        }
    
    def search_all(
        self,
        query: str,
        databases: List[str] = None,
        limit_per_source: int = 25
    ) -> List[Dict]:
        """Search multiple databases and aggregate results"""
        
        if databases is None:
            databases = ['semantic_scholar', 'arxiv']
        
        all_papers = []
        
        for db_name in databases:
            if db_name not in self.sources:
                print(f"⚠️ Unknown database: {db_name}")
                continue
            
            try:
                source = self.sources[db_name]
                papers = source.search(query, limit=limit_per_source)
                
                # Convert to dicts
                paper_dicts = [p.to_dict() for p in papers]
                all_papers.extend(paper_dicts)
                
                print(f"✓ {db_name}: Found {len(papers)} papers")
                
            except Exception as e:
                print(f"⚠️ {db_name} error: {e}")
        
        # Deduplicate by DOI or title
        unique_papers = self._deduplicate(all_papers)
        
        self.search_history.append({
            'query': query,
            'databases': databases,
            'raw_count': len(all_papers),
            'unique_count': len(unique_papers)
        })
        
        return unique_papers
    
    def _deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers by DOI or title"""
        seen = set()
        unique = []
        
        for paper in papers:
            identifier = paper.get('doi') or paper.get('title', '').lower().strip()[:100]
            
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique.append(paper)
        
        return unique
    
    def search_with_queries(
        self,
        queries: List[Dict],
        limit_per_query: int = 20
    ) -> List[Dict]:
        """Execute multiple refinement queries"""
        
        all_papers = []
        
        for query_info in queries:
            query_text = query_info.get('query', '')
            databases = query_info.get('databases', ['semantic_scholar', 'arxiv'])
            
            papers = self.search_all(query_text, databases, limit_per_query)
            all_papers.extend(papers)
        
        # Final dedup
        return self._deduplicate(all_papers)
