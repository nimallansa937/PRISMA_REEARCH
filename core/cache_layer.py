"""
Caching Layer - Upgrade #5.
SQLite-based cache for API responses, embeddings, and query results.
Prevents redundant API calls and enables incremental updates.
"""

import sqlite3
import json
import hashlib
import time
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class ResearchCache:
    """
    Multi-layer SQLite cache for the research agent.

    Layers:
    1. API Response Cache - raw results from each source
    2. Paper Cache - deduplicated paper metadata
    3. Query Cache - maps queries to paper sets
    4. Embedding Cache - precomputed paper embeddings
    """

    def __init__(self, db_path: str = None, ttl_days: int = None):
        self.db_path = db_path or str(settings.CACHE_DIR / "research_cache.db")
        self.ttl_days = ttl_days or settings.CACHE_EXPIRY_DAYS

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create cache tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                response_json TEXT NOT NULL,
                paper_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_cache (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                abstract TEXT,
                doi TEXT,
                venue TEXT,
                citation_count INTEGER DEFAULT 0,
                source TEXT,
                full_data TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                source TEXT,
                paper_count INTEGER,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                paper_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_api_source ON api_cache(source);
            CREATE INDEX IF NOT EXISTS idx_api_expires ON api_cache(expires_at);
            CREATE INDEX IF NOT EXISTS idx_paper_doi ON paper_cache(doi);
            CREATE INDEX IF NOT EXISTS idx_paper_title ON paper_cache(title);
            CREATE INDEX IF NOT EXISTS idx_query_log_query ON query_log(query);
        """)
        self.conn.commit()

    def _make_key(self, source: str, query: str, limit: int = 0, offset: int = 0) -> str:
        """Generate cache key from search parameters."""
        raw = f"{source}:{query}:{limit}:{offset}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # ==================== API Response Cache ====================

    def get_api_response(self, source: str, query: str,
                          limit: int = 0, offset: int = 0) -> Optional[List[Dict]]:
        """Retrieve cached API response."""
        key = self._make_key(source, query, limit, offset)
        now = datetime.now().isoformat()

        row = self.conn.execute(
            "SELECT response_json FROM api_cache WHERE cache_key = ? AND expires_at > ?",
            (key, now)
        ).fetchone()

        if row:
            return json.loads(row['response_json'])
        return None

    def set_api_response(self, source: str, query: str, papers: List[Dict],
                          limit: int = 0, offset: int = 0):
        """Cache an API response."""
        key = self._make_key(source, query, limit, offset)
        now = datetime.now()
        expires = now + timedelta(days=self.ttl_days)

        self.conn.execute(
            """INSERT OR REPLACE INTO api_cache
               (cache_key, source, query, response_json, paper_count, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (key, source, query, json.dumps(papers), len(papers),
             now.isoformat(), expires.isoformat())
        )
        self.conn.commit()

    # ==================== Paper Cache ====================

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Retrieve a cached paper."""
        row = self.conn.execute(
            "SELECT full_data FROM paper_cache WHERE paper_id = ?",
            (paper_id,)
        ).fetchone()

        if row:
            return json.loads(row['full_data'])
        return None

    def set_paper(self, paper: Dict):
        """Cache a paper."""
        paper_id = paper.get('paper_id', '') or paper.get('doi', '')
        if not paper_id:
            return

        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO paper_cache
               (paper_id, title, authors, year, abstract, doi, venue,
                citation_count, source, full_data, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (paper_id, paper.get('title', ''),
             json.dumps(paper.get('authors', [])),
             paper.get('year', 0), paper.get('abstract', ''),
             paper.get('doi', ''), paper.get('venue', ''),
             paper.get('citation_count', 0), paper.get('source', ''),
             json.dumps(paper), now, now)
        )
        self.conn.commit()

    def set_papers_batch(self, papers: List[Dict]):
        """Cache multiple papers in a single transaction."""
        now = datetime.now().isoformat()
        rows = []
        for paper in papers:
            paper_id = paper.get('paper_id', '') or paper.get('doi', '')
            if not paper_id:
                continue
            rows.append((
                paper_id, paper.get('title', ''),
                json.dumps(paper.get('authors', [])),
                paper.get('year', 0), paper.get('abstract', ''),
                paper.get('doi', ''), paper.get('venue', ''),
                paper.get('citation_count', 0), paper.get('source', ''),
                json.dumps(paper), now, now
            ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO paper_cache
               (paper_id, title, authors, year, abstract, doi, venue,
                citation_count, source, full_data, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        self.conn.commit()

    def search_papers_local(self, query: str, limit: int = 50) -> List[Dict]:
        """Search cached papers by title/abstract."""
        rows = self.conn.execute(
            """SELECT full_data FROM paper_cache
               WHERE title LIKE ? OR abstract LIKE ?
               ORDER BY citation_count DESC
               LIMIT ?""",
            (f'%{query}%', f'%{query}%', limit)
        ).fetchall()

        return [json.loads(row['full_data']) for row in rows]

    # ==================== Embedding Cache ====================

    def get_embeddings(self, paper_ids: List[str], model_name: str) -> Dict[str, Any]:
        """Retrieve cached embeddings."""
        import numpy as np
        results = {}
        placeholders = ','.join('?' * len(paper_ids))
        rows = self.conn.execute(
            f"""SELECT paper_id, embedding FROM embedding_cache
                WHERE paper_id IN ({placeholders}) AND model_name = ?""",
            paper_ids + [model_name]
        ).fetchall()

        for row in rows:
            results[row['paper_id']] = np.frombuffer(row['embedding'], dtype=np.float32)

        return results

    def set_embeddings(self, paper_ids: List[str], embeddings: Any,
                       model_name: str):
        """Cache embeddings."""
        import numpy as np
        now = datetime.now().isoformat()
        rows = []
        for pid, emb in zip(paper_ids, embeddings):
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()
            rows.append((pid, emb_bytes, model_name, now))

        self.conn.executemany(
            """INSERT OR REPLACE INTO embedding_cache
               (paper_id, embedding, model_name, created_at)
               VALUES (?, ?, ?, ?)""",
            rows
        )
        self.conn.commit()

    # ==================== Query Logging ====================

    def log_query(self, query: str, source: str, paper_count: int):
        """Log a query for analytics."""
        self.conn.execute(
            "INSERT INTO query_log (query, source, paper_count, timestamp) VALUES (?, ?, ?, ?)",
            (query, source, paper_count, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_query_history(self, limit: int = 50) -> List[Dict]:
        """Get recent query history."""
        rows = self.conn.execute(
            "SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    # ==================== Maintenance ====================

    def cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "DELETE FROM api_cache WHERE expires_at < ?", (now,)
        )
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        api_count = self.conn.execute("SELECT COUNT(*) as c FROM api_cache").fetchone()['c']
        paper_count = self.conn.execute("SELECT COUNT(*) as c FROM paper_cache").fetchone()['c']
        emb_count = self.conn.execute("SELECT COUNT(*) as c FROM embedding_cache").fetchone()['c']
        query_count = self.conn.execute("SELECT COUNT(*) as c FROM query_log").fetchone()['c']

        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        return {
            'api_cache_entries': api_count,
            'cached_papers': paper_count,
            'cached_embeddings': emb_count,
            'queries_logged': query_count,
            'db_size_mb': db_size / (1024 * 1024),
            'ttl_days': self.ttl_days
        }

    def clear_all(self):
        """Clear all caches."""
        self.conn.executescript("""
            DELETE FROM api_cache;
            DELETE FROM paper_cache;
            DELETE FROM embedding_cache;
            DELETE FROM query_log;
        """)
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()
