"""
Semantic Clustering Engine - Upgrade #4.
Groups papers by topic similarity using sentence embeddings.
Enables organized thematic analysis of 1000+ papers.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class SemanticClusterer:
    """
    Clusters papers by abstract similarity using sentence-transformers.
    Falls back to TF-IDF if sentence-transformers unavailable.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.use_tfidf = False
        self._load_model()

    def _load_model(self):
        """Load sentence-transformers or fall back to TF-IDF."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"  [Clustering] Loaded {self.model_name}")
        except ImportError:
            print("  [Clustering] sentence-transformers not available, using TF-IDF fallback")
            self.use_tfidf = True

    def cluster_papers(self, papers: List[Dict], n_clusters: int = None,
                       min_cluster_size: int = 5) -> Dict:
        """
        Cluster papers by semantic similarity.

        Args:
            papers: List of paper dicts with 'abstract' and 'title' fields
            n_clusters: Number of clusters (auto-detected if None)
            min_cluster_size: Minimum papers per cluster

        Returns:
            Dict with clusters, labels, and metadata
        """
        if len(papers) < min_cluster_size:
            return {
                'clusters': [{'cluster_id': 0, 'papers': list(range(len(papers))),
                              'label': 'All Papers', 'size': len(papers)}],
                'n_clusters': 1,
                'embeddings': None
            }

        # Generate embeddings
        texts = [self._get_text(p) for p in papers]
        embeddings = self._embed(texts)

        # Auto-detect number of clusters
        if n_clusters is None:
            n_clusters = self._auto_n_clusters(len(papers))

        # Cluster
        labels = self._cluster(embeddings, n_clusters, min_cluster_size)

        # Build cluster info
        clusters = self._build_clusters(papers, labels, embeddings)

        return {
            'clusters': clusters,
            'n_clusters': len(clusters),
            'paper_labels': labels.tolist() if hasattr(labels, 'tolist') else list(labels),
            'embeddings': embeddings
        }

    def find_similar(self, query_text: str, papers: List[Dict],
                     embeddings: np.ndarray = None, top_k: int = 20) -> List[Tuple[int, float]]:
        """Find most similar papers to a query text."""
        if embeddings is None:
            texts = [self._get_text(p) for p in papers]
            embeddings = self._embed(texts)

        query_emb = self._embed([query_text])
        similarities = self._cosine_similarity(query_emb, embeddings)[0]

        # Top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _get_text(self, paper: Dict) -> str:
        """Extract text for embedding from a paper."""
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        tldr = paper.get('tldr', '') or ''

        if tldr:
            return f"{title}. {tldr}"
        elif abstract:
            return f"{title}. {abstract[:500]}"
        else:
            return title

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using model or TF-IDF."""
        if self.use_tfidf:
            return self._tfidf_embed(texts)
        else:
            return self.model.encode(texts, show_progress_bar=False, batch_size=64)

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """TF-IDF fallback embedding."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Reduce to dense 128-dim
        n_components = min(128, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components)
        embeddings = svd.fit_transform(tfidf_matrix)

        return embeddings

    def _cluster(self, embeddings: np.ndarray, n_clusters: int,
                 min_cluster_size: int) -> np.ndarray:
        """Cluster embeddings using HDBSCAN or KMeans."""
        try:
            # Try HDBSCAN first (better for irregular clusters)
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings)
            # Check if HDBSCAN found reasonable clusters
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            if n_found >= 2:
                return labels
        except ImportError:
            pass

        # Fallback to KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)

    def _auto_n_clusters(self, n_papers: int) -> int:
        """Auto-detect appropriate number of clusters."""
        if n_papers < 20:
            return 3
        elif n_papers < 50:
            return 5
        elif n_papers < 100:
            return 8
        elif n_papers < 300:
            return 12
        elif n_papers < 500:
            return 15
        elif n_papers < 1000:
            return 20
        else:
            return 25

    def _build_clusters(self, papers: List[Dict], labels: np.ndarray,
                        embeddings: np.ndarray) -> List[Dict]:
        """Build structured cluster information."""
        cluster_papers = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_papers[int(label)].append(i)

        clusters = []
        for cluster_id, paper_indices in sorted(cluster_papers.items()):
            if cluster_id == -1:  # HDBSCAN noise
                continue

            cluster_docs = [papers[i] for i in paper_indices]

            # Generate cluster label from common terms
            label = self._generate_cluster_label(cluster_docs)

            # Get centroid paper (closest to cluster center)
            cluster_embs = embeddings[paper_indices]
            centroid = cluster_embs.mean(axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            centroid_idx = paper_indices[np.argmin(dists)]

            # Year distribution
            years = [p.get('year', 0) for p in cluster_docs if p.get('year')]
            year_range = f"{min(years)}-{max(years)}" if years else "N/A"

            # Top cited in cluster
            sorted_by_cites = sorted(
                paper_indices,
                key=lambda i: papers[i].get('citation_count', 0) or 0,
                reverse=True
            )

            clusters.append({
                'cluster_id': cluster_id,
                'label': label,
                'size': len(paper_indices),
                'papers': paper_indices,
                'centroid_paper': centroid_idx,
                'year_range': year_range,
                'avg_citations': np.mean([papers[i].get('citation_count', 0) or 0
                                          for i in paper_indices]),
                'top_papers': sorted_by_cites[:5]
            })

        # Sort by size descending
        clusters.sort(key=lambda c: c['size'], reverse=True)
        return clusters

    def _generate_cluster_label(self, papers: List[Dict], max_words: int = 5) -> str:
        """Generate a descriptive label for a cluster."""
        # Combine titles and extract common terms
        all_text = ' '.join(
            (p.get('title', '') or '').lower() for p in papers
        )

        # Simple word frequency (skip common words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'on',
            'with', 'by', 'from', 'at', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'has', 'have', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'can', 'not', 'no', 'nor', 'but', 'yet', 'so', 'as', 'if',
            'then', 'than', 'too', 'very', 'just', 'about', 'above',
            'after', 'again', 'all', 'also', 'am', 'any', 'because',
            'before', 'between', 'both', 'during', 'each', 'few',
            'further', 'here', 'how', 'its', 'more', 'most', 'other',
            'our', 'out', 'over', 'own', 'same', 'some', 'such', 'that',
            'their', 'them', 'these', 'this', 'those', 'through', 'under',
            'until', 'up', 'we', 'what', 'when', 'where', 'which', 'while',
            'who', 'why', 'you', 'using', 'based', 'study', 'analysis',
            'approach', 'method', 'paper', 'research', 'results', 'new',
        }

        import re
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        word_counts = Counter(w for w in words if w not in stop_words)

        top_words = [w for w, _ in word_counts.most_common(max_words)]
        return ' '.join(top_words).title() if top_words else 'Miscellaneous'

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
