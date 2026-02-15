"""
RAG Engine - Retrieval-Augmented Generation for systematic reviews.

Indexes full-text paper chunks into ChromaDB, retrieves semantically
relevant evidence for any query, and generates grounded answers using
gpt-oss:120b-cloud (131K context) via Ollama.

Architecture:
  Papers -> Chunk (2000 words, 200 overlap) -> Embed (all-MiniLM-L6-v2)
  -> ChromaDB persistent store -> Semantic retrieval -> LLM generation
  with 30-50 evidence passages -> Grounded answer + citations
"""

import json
import hashlib
import re
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import requests as req

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with provenance."""
    text: str
    paper_title: str
    paper_authors: List[str]
    paper_year: int
    paper_doi: str
    section: str
    chunk_index: int
    similarity_score: float
    paper_id: str = ""

    def format_citation(self) -> str:
        """Format as inline citation."""
        first_author = self.paper_authors[0] if self.paper_authors else "Unknown"
        return f"[{first_author}, {self.paper_year}]"

    def format_evidence(self, max_words: int = 500) -> str:
        """Format as evidence block for LLM prompt."""
        words = self.text.split()
        truncated = ' '.join(words[:max_words])
        citation = self.format_citation()
        return f"--- Evidence from {citation} ({self.paper_title[:80]}) ---\n{truncated}\n"


class RAGEngine:
    """
    ChromaDB-backed RAG engine for systematic review corpus.

    Key methods:
    - ingest_papers(): Chunk + embed + store in ChromaDB
    - retrieve(): Semantic search for relevant chunks
    - generate(): RAG generation with LLM
    - generate_report_section(): Single section with dedicated retrieval
    """

    def __init__(self, persist_dir: str = None, collection_name: str = "research_corpus",
                 ollama_model: str = None):
        self.persist_dir = persist_dir or ".cache/rag_store"
        self.collection_name = collection_name
        self.chunk_size = 2000  # words per chunk
        self.chunk_overlap = 200

        # ChromaDB
        self._client = None
        self._collection = None

        # Embedding model (shared with SemanticSearchEngine)
        self._embed_model = None

        # LLM config - use user's model selection, or auto-detect best available
        self._ollama_url = "http://localhost:11434"
        self._llm_model = None
        self._llm_context_window = 4096  # default conservative
        self._detect_best_model(preferred_model=ollama_model)

        # Stats
        self.total_chunks = 0
        self.total_papers = 0
        self.ingested_paper_ids = set()

    # Known context windows for cloud/large models
    MODEL_CONTEXT_WINDOWS = {
        'gpt-oss:120b-cloud': 131072,
        'deepseek-v3.1:671b-cloud': 163840,
        'kimi-k2.5:cloud': 131072,
        'qwen3-coder:480b-cloud': 131072,
        'deepseek-v3.1': 163840,
        'kimi-k2.5': 131072,
    }

    def _detect_best_model(self, preferred_model: str = None):
        """
        Detect the best Ollama model for RAG generation.

        If preferred_model is given (from dashboard selection), use it.
        Otherwise auto-detect the best available cloud frontier model.
        """
        try:
            resp = req.get(f"{self._ollama_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return
            models = resp.json().get('models', [])
            available = {m['name']: m for m in models}

            if not available:
                return

            # --- If user selected a specific model, use it ---
            if preferred_model:
                matched = self._match_model(preferred_model, available)
                if matched:
                    self._llm_model = matched
                    self._llm_context_window = self._get_context_window(matched)
                    print(f"  RAG Engine: using {self._llm_model} ({self._llm_context_window//1024}K context) [user-selected]")
                    return
                else:
                    print(f"  RAG Engine: user model '{preferred_model}' not found, auto-selecting...")

            # --- Auto-detect: cloud frontier models first ---
            auto_priority = [
                'gpt-oss:120b-cloud',
                'deepseek-v3.1:671b-cloud',
                'kimi-k2.5:cloud',
                'qwen3-coder:480b-cloud',
                'qwen2.5:7b',
                'gemma2:2b',
            ]

            for model_name in auto_priority:
                matched = self._match_model(model_name, available)
                if matched:
                    self._llm_model = matched
                    self._llm_context_window = self._get_context_window(matched)
                    print(f"  RAG Engine: using {self._llm_model} ({self._llm_context_window//1024}K context) [auto-detected]")
                    return

            # Fallback to first available
            if models:
                self._llm_model = models[0]['name']
                self._llm_context_window = self._get_context_window(self._llm_model)
                print(f"  RAG Engine: fallback to {self._llm_model}")

        except Exception as e:
            print(f"  RAG Engine: Ollama detection failed: {e}")

    def _match_model(self, target: str, available: Dict) -> Optional[str]:
        """Match a model name against available Ollama models."""
        # Exact match
        if target in available:
            return target
        # Prefix match (e.g. 'deepseek-v3.1' matches 'deepseek-v3.1:671b-cloud')
        for name in available:
            if name.startswith(target) or target.startswith(name.split(':')[0]):
                if target in name or name in target:
                    return name
        return None

    def _get_context_window(self, model_name: str) -> int:
        """Get known context window for a model, or conservative default."""
        # Check known models
        for known, ctx in self.MODEL_CONTEXT_WINDOWS.items():
            if known in model_name or model_name in known:
                return ctx
        # Cloud models generally have large context
        if 'cloud' in model_name:
            return 131072
        # Local models default
        return 4096

    def _get_collection(self):
        """Lazy-init ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            self.total_chunks = self._collection.count()
            print(f"  RAG Store: {self.total_chunks} chunks in {self.persist_dir}")
            return self._collection

        except ImportError:
            print("  ChromaDB not installed. Run: pip install chromadb")
            return None

    def _get_embed_model(self):
        """Lazy-load sentence-transformers model."""
        if self._embed_model is not None:
            return self._embed_model

        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self._embed_model
        except ImportError:
            print("  sentence-transformers not installed")
            return None

    # ================================================================
    # INGESTION
    # ================================================================

    def ingest_papers(self, papers: List[Dict],
                      full_text_papers: List = None,
                      progress_callback=None) -> Dict:
        """
        Ingest papers into ChromaDB vector store.

        Args:
            papers: List of paper dicts (at minimum: title, abstract)
            full_text_papers: Optional list of FullTextPaper objects with full text + chunks
            progress_callback: Optional callable(message, progress_float)

        Returns:
            Stats dict with chunks_added, papers_processed, etc.
        """
        collection = self._get_collection()
        if collection is None:
            return {"error": "ChromaDB not available"}

        embed_model = self._get_embed_model()
        if embed_model is None:
            return {"error": "Embedding model not available"}

        # Build full-text lookup
        ft_lookup = {}
        if full_text_papers:
            for ftp in full_text_papers:
                key = ftp.paper_id or ftp.doi or ftp.title[:100]
                ft_lookup[key] = ftp

        chunks_added = 0
        papers_processed = 0
        total = len(papers)

        for i, paper in enumerate(papers):
            paper_id = paper.get('paper_id') or paper.get('doi') or (paper.get('title', '') or '')[:100]

            if paper_id in self.ingested_paper_ids:
                continue

            title = paper.get('title', '')
            authors = paper.get('authors', [])
            year = paper.get('year', 0)
            doi = paper.get('doi', '')
            abstract = paper.get('abstract', '')

            # Get full text chunks if available
            ft_key = paper.get('paper_id') or paper.get('doi') or title[:100]
            ft_paper = ft_lookup.get(ft_key)

            if ft_paper and ft_paper.chunks:
                # Use full-text chunks
                text_chunks = ft_paper.chunks
                source_type = f"fulltext_{ft_paper.source_type}"
            elif abstract and len(abstract) > 100:
                # Fall back to abstract chunking
                text_chunks = self._chunk_text(f"{title}\n\n{abstract}")
                source_type = "abstract"
            else:
                continue  # Skip papers with no content

            # Embed and store each chunk
            for ci, chunk_text in enumerate(text_chunks):
                chunk_id = f"{hashlib.md5(paper_id.encode()).hexdigest()[:12]}_c{ci}"

                # Check if already exists
                try:
                    existing = collection.get(ids=[chunk_id])
                    if existing and existing['ids']:
                        continue
                except Exception:
                    pass

                # Embed
                embedding = embed_model.encode(chunk_text, normalize_embeddings=True).tolist()

                # Store with metadata
                metadata = {
                    "paper_id": paper_id[:200],
                    "title": title[:200],
                    "authors": ', '.join(authors[:3])[:200],
                    "year": year or 0,
                    "doi": doi[:100],
                    "chunk_index": ci,
                    "source_type": source_type,
                    "word_count": len(chunk_text.split()),
                }

                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk_text[:10000]],  # Cap per chunk
                    metadatas=[metadata]
                )
                chunks_added += 1

            self.ingested_paper_ids.add(paper_id)
            papers_processed += 1

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(
                    f"Indexed {papers_processed}/{total} papers ({chunks_added} chunks)",
                    (i + 1) / total
                )

        self.total_chunks = collection.count()
        self.total_papers = len(self.ingested_paper_ids)

        stats = {
            "chunks_added": chunks_added,
            "papers_processed": papers_processed,
            "total_chunks": self.total_chunks,
            "total_papers": self.total_papers,
        }

        if progress_callback:
            progress_callback(f"RAG indexing complete: {self.total_chunks} chunks", 1.0)

        return stats

    # ================================================================
    # RETRIEVAL
    # ================================================================

    def retrieve(self, query: str, top_k: int = 20,
                 year_filter: Optional[Tuple[int, int]] = None) -> List[RetrievedChunk]:
        """
        Semantic retrieval of relevant chunks.

        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve
            year_filter: Optional (min_year, max_year) tuple

        Returns:
            List of RetrievedChunk objects ranked by relevance
        """
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return []

        embed_model = self._get_embed_model()
        if embed_model is None:
            return []

        # Embed query
        query_embedding = embed_model.encode(query, normalize_embeddings=True).tolist()

        # Build where filter
        where_filter = None
        if year_filter:
            where_filter = {
                "$and": [
                    {"year": {"$gte": year_filter[0]}},
                    {"year": {"$lte": year_filter[1]}}
                ]
            }

        # Query ChromaDB
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"  RAG retrieval error: {e}")
            return []

        if not results or not results['ids'] or not results['ids'][0]:
            return []

        # Convert to RetrievedChunk objects
        chunks = []
        for idx in range(len(results['ids'][0])):
            meta = results['metadatas'][0][idx] if results['metadatas'] else {}
            doc = results['documents'][0][idx] if results['documents'] else ""
            dist = results['distances'][0][idx] if results['distances'] else 1.0

            # ChromaDB returns distances (lower = more similar for cosine)
            similarity = 1.0 - dist  # Convert distance to similarity

            authors_str = meta.get('authors', '')
            authors_list = [a.strip() for a in authors_str.split(',')] if authors_str else []

            chunk = RetrievedChunk(
                text=doc,
                paper_title=meta.get('title', 'Unknown'),
                paper_authors=authors_list,
                paper_year=meta.get('year', 0),
                paper_doi=meta.get('doi', ''),
                section=meta.get('source_type', ''),
                chunk_index=meta.get('chunk_index', 0),
                similarity_score=similarity,
                paper_id=meta.get('paper_id', '')
            )
            chunks.append(chunk)

        # Sort by similarity (highest first)
        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return chunks

    # ================================================================
    # GENERATION
    # ================================================================

    def generate(self, query: str, chunks: List[RetrievedChunk] = None,
                 system_prompt: str = None, top_k: int = 30,
                 temperature: float = 0.3) -> Dict:
        """
        RAG generation: retrieve evidence + generate grounded answer.

        Args:
            query: User question
            chunks: Pre-retrieved chunks (if None, auto-retrieves)
            system_prompt: Custom system prompt
            top_k: Number of chunks to use
            temperature: LLM temperature

        Returns:
            Dict with 'answer', 'citations', 'chunks_used'
        """
        if not self._llm_model:
            return {"answer": "No LLM available for RAG generation.", "citations": [], "chunks_used": 0}

        # Retrieve if not provided
        if chunks is None:
            chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return {"answer": "No relevant evidence found in the corpus.", "citations": [], "chunks_used": 0}

        # Build evidence context within token budget
        evidence_budget = self._compute_evidence_budget()
        evidence_text, used_chunks = self._build_evidence_context(chunks, evidence_budget)

        # System prompt
        if system_prompt is None:
            system_prompt = """You are a systematic review research assistant with access to full-text academic papers.
Answer questions using ONLY the evidence provided below. For every claim, cite the source using [Author, Year] format.
If the evidence is insufficient, say so explicitly. Never fabricate information."""

        # User prompt
        user_prompt = f"""## Research Question
{query}

## Evidence from {len(used_chunks)} paper chunks (retrieved from corpus)

{evidence_text}

## Instructions
1. Answer the question comprehensively using the evidence above
2. Cite every claim with [Author, Year] format
3. Identify areas where evidence is strong vs. weak
4. Note any contradictions between sources
5. Be specific - use data, statistics, and concrete findings from the papers"""

        # Call LLM
        answer = self._call_llm(system_prompt, user_prompt, temperature)

        # Extract citations
        citations = self._extract_citations(answer, used_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": len(used_chunks),
            "model": self._llm_model,
        }

    def generate_report_section(self, section_name: str, section_prompt: str,
                                 query: str, top_k: int = 25) -> str:
        """Generate one report section with dedicated RAG retrieval."""
        # Section-specific retrieval query
        retrieval_query = f"{query} {section_name}"
        chunks = self.retrieve(retrieval_query, top_k=top_k)

        if not chunks:
            return f"*No evidence available for {section_name}.*"

        system_prompt = f"""You are writing the "{section_name}" section of a systematic review report.
Use ONLY the evidence provided. Cite sources as [Author, Year].
Be comprehensive, specific, and evidence-based. Include statistics and concrete findings."""

        result = self.generate(
            query=section_prompt,
            chunks=chunks,
            system_prompt=system_prompt,
            top_k=top_k
        )

        return result.get("answer", f"*{section_name} generation failed.*")

    # ================================================================
    # HELPERS
    # ================================================================

    def _compute_evidence_budget(self) -> int:
        """Compute token budget for evidence based on model context window."""
        # Reserve tokens for system prompt (~500) + generation (~30K for large, ~600 for small)
        if self._llm_context_window >= 100000:
            # Frontier model (gpt-oss:120b, deepseek-v3.1)
            return 90000  # ~90K tokens for evidence
        elif self._llm_context_window >= 30000:
            # Mid-range model
            return 20000
        else:
            # Small local model (qwen2.5:7b)
            return 3000

    def _build_evidence_context(self, chunks: List[RetrievedChunk],
                                 max_tokens: int) -> Tuple[str, List[RetrievedChunk]]:
        """Build evidence text within token budget."""
        # Rough estimate: 1 word ~= 1.3 tokens
        max_words = int(max_tokens / 1.3)
        current_words = 0
        evidence_parts = []
        used_chunks = []

        for chunk in chunks:
            chunk_words = len(chunk.text.split())
            if current_words + chunk_words > max_words:
                break

            evidence_parts.append(chunk.format_evidence(max_words=min(chunk_words, 800)))
            used_chunks.append(chunk)
            current_words += chunk_words

        return "\n".join(evidence_parts), used_chunks

    def _call_llm(self, system_prompt: str, user_prompt: str,
                   temperature: float = 0.3) -> str:
        """Call Ollama LLM for text generation."""
        if not self._llm_model:
            return "No LLM available."

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Estimate token count and set appropriate num_predict
        prompt_words = len(full_prompt.split())
        prompt_tokens_est = int(prompt_words * 1.3)
        max_generate = min(
            max(2000, self._llm_context_window - prompt_tokens_est - 500),
            32768
        )

        payload = {
            "model": self._llm_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_generate,
            }
        }

        try:
            resp = req.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=600  # 10 min for large generations
            )

            if resp.status_code != 200:
                return f"LLM error {resp.status_code}: {resp.text[:200]}"

            text = resp.json().get('response', '').strip()

            # Strip thinking tags
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            return text

        except Exception as e:
            return f"LLM call failed: {e}"

    def _extract_citations(self, answer: str,
                            chunks: List[RetrievedChunk]) -> List[Dict]:
        """Extract citation references from the generated answer."""
        citations = []
        seen = set()

        # Find [Author, Year] patterns
        pattern = r'\[([^\]]+?,\s*\d{4})\]'
        matches = re.findall(pattern, answer)

        for match in matches:
            if match in seen:
                continue
            seen.add(match)

            # Try to match to a chunk
            best_chunk = None
            for chunk in chunks:
                citation = chunk.format_citation().strip('[]')
                if citation.lower() in match.lower() or match.lower() in citation.lower():
                    best_chunk = chunk
                    break

            citations.append({
                "reference": f"[{match}]",
                "title": best_chunk.paper_title if best_chunk else "Unknown",
                "doi": best_chunk.paper_doi if best_chunk else "",
                "year": best_chunk.paper_year if best_chunk else 0,
            })

        return citations

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def get_stats(self) -> Dict:
        """Get RAG engine statistics."""
        collection = self._get_collection()
        count = collection.count() if collection else 0

        return {
            "total_chunks": count,
            "total_papers": self.total_papers,
            "persist_dir": self.persist_dir,
            "llm_model": self._llm_model or "none",
            "context_window": self._llm_context_window,
            "evidence_budget": self._compute_evidence_budget(),
        }

    def clear(self):
        """Clear the vector store."""
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            self.total_chunks = 0
            self.total_papers = 0
            self.ingested_paper_ids.clear()
            print("  RAG Store: cleared")
