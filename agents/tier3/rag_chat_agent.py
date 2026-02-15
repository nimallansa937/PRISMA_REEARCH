"""
RAG Chat Agent - Interactive Q&A over the entire research corpus.

Uses ChromaDB vector retrieval + gpt-oss:120b-cloud for grounded
answers with citations. Supports multi-turn conversation.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import Tier3Agent


class RAGChatAgent(Tier3Agent):
    """
    Interactive research assistant backed by RAG retrieval.

    Retrieves relevant full-text chunks from ChromaDB,
    feeds them to gpt-oss:120b-cloud, and produces grounded
    answers with inline [Author, Year] citations.
    """

    def __init__(self, rag_engine=None):
        super().__init__(
            name="RAGChatAgent",
            description="Grounded Q&A over research corpus via RAG"
        )
        self.rag_engine = rag_engine
        self.conversation_history: List[Dict] = []

    def set_rag_engine(self, rag_engine):
        """Set or update the RAG engine reference."""
        self.rag_engine = rag_engine

    def execute(self, input_data: Dict) -> Dict:
        """
        Answer a research question using RAG retrieval.

        Input:
            question: str - The user's question
            top_k: int - Number of chunks to retrieve (default 20)
            year_range: tuple - Optional (min_year, max_year)
            include_followups: bool - Generate follow-up questions

        Returns:
            answer: str - Grounded answer with citations
            citations: list - Source references
            chunks_used: int - Number of evidence chunks used
            follow_up_questions: list - Suggested follow-ups
        """
        if self.rag_engine is None:
            return {
                "answer": "RAG engine not initialized. Run a review first to index papers.",
                "citations": [],
                "chunks_used": 0,
                "follow_up_questions": []
            }

        question = input_data.get('question', '')
        top_k = input_data.get('top_k', 20)
        year_range = input_data.get('year_range')
        include_followups = input_data.get('include_followups', True)

        if not question:
            return {
                "answer": "Please provide a question.",
                "citations": [],
                "chunks_used": 0,
                "follow_up_questions": []
            }

        # Retrieve relevant chunks
        chunks = self.rag_engine.retrieve(
            query=question,
            top_k=top_k,
            year_filter=year_range
        )

        if not chunks:
            return {
                "answer": "No relevant evidence found in the indexed corpus. "
                          "Try rephrasing your question or running a new review.",
                "citations": [],
                "chunks_used": 0,
                "follow_up_questions": [
                    "What topics are covered in the corpus?",
                    "Can you summarize the main themes?",
                ]
            }

        # Build conversation context
        history_context = ""
        if self.conversation_history:
            recent = self.conversation_history[-3:]  # Last 3 turns
            for turn in recent:
                history_context += f"\nPrevious Q: {turn['question'][:200]}\n"
                history_context += f"Previous A: {turn['answer'][:300]}...\n"

        # Enhanced system prompt with conversation context
        system_prompt = """You are a research assistant with access to full-text academic papers indexed in a vector database.
Answer questions using ONLY the evidence provided. For every claim, cite the source using [Author, Year] format.
Be comprehensive and specific - use data, statistics, and concrete findings."""

        if history_context:
            system_prompt += f"\n\n## Previous conversation context:\n{history_context}"

        # Generate answer via RAG engine
        result = self.rag_engine.generate(
            query=question,
            chunks=chunks,
            system_prompt=system_prompt,
            top_k=top_k
        )

        answer = result.get('answer', 'Generation failed.')
        citations = result.get('citations', [])

        # Generate follow-up questions
        follow_ups = []
        if include_followups and chunks:
            follow_ups = self._generate_followups(question, answer, chunks)

        # Store in conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer[:500],
            "chunks_used": len(chunks),
        })

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": result.get('chunks_used', 0),
            "follow_up_questions": follow_ups,
            "model": result.get('model', 'unknown'),
        }

    def _generate_followups(self, question: str, answer: str,
                             chunks: List) -> List[str]:
        """Generate follow-up questions based on the answer and evidence."""
        # Extract unique topics from chunks for suggestions
        topics = set()
        years = set()
        for chunk in chunks[:10]:
            if chunk.paper_year:
                years.add(chunk.paper_year)
            # Extract key terms from title
            words = chunk.paper_title.lower().split()
            for w in words:
                if len(w) > 6 and w.isalpha():
                    topics.add(w)

        follow_ups = []

        # Methodological follow-up
        follow_ups.append(f"What research methods were used to study this?")

        # Temporal follow-up
        if years:
            min_y, max_y = min(years), max(years)
            follow_ups.append(f"How has research on this evolved from {min_y} to {max_y}?")

        # Contradiction follow-up
        follow_ups.append("Are there any conflicting findings in the literature?")

        # Practical follow-up
        follow_ups.append("What are the practical implications of these findings?")

        return follow_ups[:4]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
