"""
Semantic Search Engine - Vector-based paper matching.

Replicates SciSpace's semantic search: natural language queries
matched against paper embeddings instead of keyword matching.

Uses sentence-transformers (all-MiniLM-L6-v2) for fast embedding.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class SemanticSearchEngine:
    """
    Vector-based semantic search over paper corpus.
    Encodes queries and paper abstracts as embeddings,
    then ranks by cosine similarity.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = '.cache/embeddings'):
        self.model_name = model_name
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Paper index
        self.paper_embeddings: Optional[np.ndarray] = None
        self.paper_ids: List[str] = []
        self.paper_texts: List[str] = []
        self.papers_index: Dict[str, Dict] = {}

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"  Semantic search: loaded {self.model_name}")
        except ImportError:
            print("  sentence-transformers not installed, using TF-IDF fallback")
            self.model = None

    def index_papers(self, papers: List[Dict]):
        """
        Build vector index from papers.
        Each paper is embedded using title + abstract.
        """
        self._load_model()

        if not papers:
            return

        # Build text representations
        texts = []
        ids = []
        for p in papers:
            title = p.get('title', '')
            abstract = p.get('abstract', '')
            text = f"{title}. {abstract}".strip()
            if len(text) < 10:
                continue

            paper_id = p.get('paper_id', p.get('doi', str(hash(title))))
            texts.append(text)
            ids.append(paper_id)
            self.papers_index[paper_id] = p

        if not texts:
            return

        self.paper_texts = texts
        self.paper_ids = ids

        # Encode papers
        if self.model:
            self.paper_embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True
            )
        else:
            # TF-IDF fallback
            self.paper_embeddings = self._tfidf_encode(texts)

        print(f"  Semantic index: {len(ids)} papers indexed")

    def search(self, query: str, top_k: int = 20,
               min_score: float = 0.3) -> List[Dict]:
        """
        Semantic search: find papers most similar to query.

        Returns papers ranked by cosine similarity with scores.
        """
        if self.paper_embeddings is None or len(self.paper_ids) == 0:
            return []

        self._load_model()

        # Encode query
        if self.model:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True
            )
        else:
            query_embedding = self._tfidf_encode([query])

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        similarities = np.dot(self.paper_embeddings, query_embedding.T).flatten()

        # Rank by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break

            paper_id = self.paper_ids[idx]
            paper = self.papers_index.get(paper_id, {})
            result = {**paper}
            result['semantic_score'] = round(score, 4)
            result['semantic_rank'] = len(results) + 1
            results.append(result)

        return results

    def find_similar_papers(self, paper_id: str, top_k: int = 10) -> List[Dict]:
        """Find papers most similar to a given paper."""
        if paper_id not in self.papers_index:
            return []

        idx = self.paper_ids.index(paper_id)
        paper_embedding = self.paper_embeddings[idx:idx + 1]

        similarities = np.dot(self.paper_embeddings, paper_embedding.T).flatten()

        # Exclude the paper itself
        similarities[idx] = -1

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_indices:
            score = float(similarities[i])
            if score < 0.2:
                break

            pid = self.paper_ids[i]
            paper = self.papers_index.get(pid, {})
            result = {**paper}
            result['similarity_score'] = round(score, 4)
            results.append(result)

        return results

    def cluster_by_similarity(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """Quick clustering using embedding similarity."""
        if self.paper_embeddings is None or len(self.paper_ids) < n_clusters:
            return {}

        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.paper_embeddings)

            clusters = {}
            for idx, label in enumerate(labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(self.paper_ids[idx])

            return clusters
        except ImportError:
            return {}

    def rerank_with_query(self, papers: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank a list of papers by semantic similarity to query.
        Useful for re-ranking keyword search results with semantic relevance.
        """
        if not papers or not query:
            return papers

        self._load_model()

        # Encode query and papers
        texts = [f"{p.get('title', '')}. {p.get('abstract', '')}".strip() for p in papers]

        if self.model:
            query_emb = self.model.encode([query], normalize_embeddings=True)
            paper_embs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        else:
            all_texts = [query] + texts
            all_embs = self._tfidf_encode(all_texts)
            query_emb = all_embs[:1]
            paper_embs = all_embs[1:]

        similarities = np.dot(paper_embs, query_emb.T).flatten()

        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1]

        ranked_papers = []
        for idx in ranked_indices:
            paper = {**papers[idx]}
            paper['semantic_score'] = round(float(similarities[idx]), 4)
            ranked_papers.append(paper)

        return ranked_papers

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF encoding when sentence-transformers unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
            matrix = vectorizer.fit_transform(texts).toarray()
            # Normalize
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return matrix / norms
        except ImportError:
            # Ultimate fallback: random embeddings (useless but won't crash)
            return np.random.randn(len(texts), 128).astype(np.float32)
