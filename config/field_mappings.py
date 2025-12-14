"""
Field-specific vocabularies and mappings for domain detection.
These help the agent identify the research field and recommend appropriate sources.
"""

# Academic field indicators (keywords → field mapping)
FIELD_INDICATORS = {
    'computer_science': [
        'algorithm', 'neural network', 'machine learning', 'AI', 'artificial intelligence',
        'software', 'database', 'programming', 'blockchain', 'cryptocurrency', 'DeFi',
        'deep learning', 'NLP', 'computer vision', 'reinforcement learning', 'GPU',
        'distributed systems', 'cybersecurity', 'data mining', 'big data'
    ],
    'medicine': [
        'patient', 'clinical', 'disease', 'treatment', 'drug', 'therapy', 'pharmaceutical',
        'diagnosis', 'medical', 'health', 'cancer', 'tumor', 'surgery', 'vaccine',
        'immunotherapy', 'oncology', 'cardiology', 'neurology', 'psychiatry'
    ],
    'biology': [
        'gene', 'protein', 'cell', 'DNA', 'RNA', 'evolution', 'species', 'organism',
        'genome', 'CRISPR', 'molecular', 'cellular', 'microbiology', 'ecology',
        'biodiversity', 'phylogenetic', 'transcription', 'mutation'
    ],
    'physics': [
        'quantum', 'particle', 'energy', 'force', 'wave', 'relativity', 'photon',
        'electron', 'magnetic', 'gravitational', 'thermodynamics', 'optics',
        'condensed matter', 'string theory', 'cosmology'
    ],
    'chemistry': [
        'molecule', 'reaction', 'compound', 'synthesis', 'catalyst', 'polymer',
        'organic', 'inorganic', 'electrochemistry', 'spectroscopy', 'crystallography'
    ],
    'economics': [
        'market', 'economy', 'trade', 'GDP', 'inflation', 'financial', 'monetary',
        'fiscal', 'macroeconomic', 'microeconomic', 'econometric', 'investment',
        'liquidity', 'leverage', 'derivatives', 'bonds', 'equities'
    ],
    'psychology': [
        'behavior', 'cognitive', 'mental', 'brain', 'emotion', 'therapy', 'anxiety',
        'depression', 'personality', 'memory', 'perception', 'consciousness',
        'developmental', 'social psychology', 'neuropsychology'
    ],
    'sociology': [
        'society', 'culture', 'social', 'community', 'demographics', 'inequality',
        'stratification', 'institutions', 'norms', 'collective behavior',
        'social networks', 'urbanization', 'migration'
    ],
    'engineering': [
        'design', 'system', 'optimization', 'manufacturing', 'circuit', 'control',
        'robotics', 'mechanical', 'electrical', 'civil', 'aerospace', 'materials',
        'structural', 'thermal', 'fluid dynamics'
    ],
    'environmental_science': [
        'climate', 'environment', 'pollution', 'sustainability', 'ecosystem',
        'carbon', 'emissions', 'renewable', 'conservation', 'biodiversity',
        'global warming', 'deforestation', 'ocean acidification'
    ],
    'mathematics': [
        'theorem', 'proof', 'algebra', 'topology', 'calculus', 'differential',
        'integral', 'probability', 'statistics', 'optimization', 'graph theory',
        'number theory', 'combinatorics'
    ]
}

# Research type indicators
RESEARCH_TYPES = {
    'review': ['review', 'survey', 'systematic review', 'meta-analysis', 'literature review'],
    'empirical': ['empirical', 'experimental', 'study', 'analysis', 'investigation', 'trial'],
    'theoretical': ['theoretical', 'model', 'framework', 'theory', 'hypothesis'],
    'methodology': ['method', 'technique', 'approach', 'methodology', 'algorithm', 'protocol'],
    'case_study': ['case study', 'case report', 'observational'],
}

# Field-specific database recommendations (ONLY IMPLEMENTED SOURCES)
SOURCE_RECOMMENDATIONS = {
    'computer_science': ['semantic_scholar', 'arxiv', 'crossref'],
    'medicine': ['pubmed', 'semantic_scholar', 'crossref'],
    'biology': ['pubmed', 'arxiv', 'semantic_scholar'],
    'physics': ['arxiv', 'semantic_scholar', 'crossref'],
    'chemistry': ['pubmed', 'semantic_scholar', 'crossref'],
    'economics': ['semantic_scholar', 'crossref', 'arxiv'],
    'psychology': ['pubmed', 'semantic_scholar', 'crossref'],
    'sociology': ['semantic_scholar', 'crossref'],
    'engineering': ['semantic_scholar', 'arxiv', 'crossref'],
    'environmental_science': ['semantic_scholar', 'arxiv', 'pubmed'],
    'mathematics': ['arxiv', 'semantic_scholar', 'crossref'],
    'general': ['semantic_scholar', 'arxiv', 'crossref']
}

# Common research synonyms for query expansion
RESEARCH_SYNONYMS = {
    # Action words
    'liquidation': ['deleveraging', 'forced selling', 'margin call', 'position closure'],
    'cascade': ['contagion', 'spillover', 'domino effect', 'systemic failure', 'chain reaction'],
    'impact': ['effect', 'consequence', 'outcome', 'result', 'influence'],
    'analysis': ['study', 'research', 'investigation', 'examination', 'assessment'],
    
    # Domain-neutral terms
    'mechanisms': ['drivers', 'causes', 'factors', 'dynamics', 'processes'],
    'risk': ['vulnerability', 'exposure', 'hazard', 'threat', 'uncertainty'],
    'model': ['framework', 'theory', 'approach', 'methodology', 'paradigm'],
    'system': ['network', 'structure', 'architecture', 'infrastructure'],
    'performance': ['efficiency', 'effectiveness', 'accuracy', 'quality'],
    
    # Crypto/Finance specific
    'cryptocurrency': ['crypto', 'digital asset', 'blockchain', 'DeFi', 'decentralized finance'],
    'market': ['exchange', 'trading', 'financial market'],
    'protocol': ['smart contract', 'platform', 'dApp'],
    
    # Medical specific
    'treatment': ['therapy', 'intervention', 'management', 'regimen'],
    'patient': ['subject', 'participant', 'individual', 'case'],
    'disease': ['condition', 'disorder', 'illness', 'pathology'],
    
    # Technical specific
    'algorithm': ['method', 'technique', 'procedure', 'approach'],
    'optimization': ['improvement', 'enhancement', 'tuning', 'refinement'],
}
