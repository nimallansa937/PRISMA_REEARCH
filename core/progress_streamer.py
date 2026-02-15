"""
Progress Streamer - Upgrade #9.
Real-time progress tracking and event streaming for long-running research tasks.
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ResearchPhase(Enum):
    INITIALIZATION = "initialization"
    QUERY_DECOMPOSITION = "query_decomposition"
    BASELINE_SEARCH = "baseline_search"
    PAGINATION = "pagination"
    CITATION_CRAWLING = "citation_crawling"
    DEDUPLICATION = "deduplication"
    SCREENING = "screening"
    RELEVANCE_FILTERING = "relevance_filtering"
    CLUSTERING = "clustering"
    RAG_INDEXING = "rag_indexing"
    MAP_SYNTHESIS = "map_synthesis"
    REDUCE_SYNTHESIS = "reduce_synthesis"
    DEEP_SYNTHESIS = "deep_synthesis"
    REPORT_GENERATION = "report_generation"
    COMPLETE = "complete"


@dataclass
class ProgressEvent:
    """A single progress update."""
    phase: ResearchPhase
    message: str
    progress: float  # 0.0 to 1.0
    papers_count: int = 0
    detail: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ProgressStreamer:
    """
    Tracks and streams research progress in real-time.
    Supports both sync callbacks and async event streaming.
    """

    def __init__(self, callback: Optional[Callable] = None, verbose: bool = True):
        self.callback = callback
        self.verbose = verbose
        self.events: List[ProgressEvent] = []
        self.start_time = time.monotonic()
        self.current_phase = ResearchPhase.INITIALIZATION
        self._listeners: List[asyncio.Queue] = []

        # Phase weights for overall progress
        self._phase_weights = {
            ResearchPhase.INITIALIZATION: 0.02,
            ResearchPhase.QUERY_DECOMPOSITION: 0.03,
            ResearchPhase.BASELINE_SEARCH: 0.10,
            ResearchPhase.PAGINATION: 0.15,
            ResearchPhase.CITATION_CRAWLING: 0.15,
            ResearchPhase.DEDUPLICATION: 0.03,
            ResearchPhase.SCREENING: 0.07,
            ResearchPhase.RELEVANCE_FILTERING: 0.10,
            ResearchPhase.CLUSTERING: 0.05,
            ResearchPhase.RAG_INDEXING: 0.08,
            ResearchPhase.MAP_SYNTHESIS: 0.12,
            ResearchPhase.REDUCE_SYNTHESIS: 0.07,
            ResearchPhase.DEEP_SYNTHESIS: 0.05,
            ResearchPhase.REPORT_GENERATION: 0.03,
            ResearchPhase.COMPLETE: 0.0,
        }
        self._phase_progress: Dict[str, float] = {}

    def update(self, phase: ResearchPhase, message: str,
               progress: float = 0.0, papers_count: int = 0,
               detail: Dict = None):
        """Record a progress update."""
        event = ProgressEvent(
            phase=phase,
            message=message,
            progress=progress,
            papers_count=papers_count,
            detail=detail or {}
        )
        self.events.append(event)
        self.current_phase = phase
        self._phase_progress[phase.value] = progress

        if self.verbose:
            elapsed = time.monotonic() - self.start_time
            overall = self._compute_overall_progress()
            bar = self._progress_bar(overall)

            icon = self._phase_icon(phase)
            papers_str = f" [{papers_count} papers]" if papers_count else ""

            print(f"  {icon} {bar} {overall:.0%} | {message}{papers_str} ({elapsed:.1f}s)")

        if self.callback:
            self.callback(event)

        # Notify async listeners
        for queue in self._listeners:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time events (for WebSocket/SSE streaming)."""
        queue = asyncio.Queue(maxsize=100)
        self._listeners.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from events."""
        if queue in self._listeners:
            self._listeners.remove(queue)

    def _compute_overall_progress(self) -> float:
        """Compute weighted overall progress."""
        phases = list(ResearchPhase)
        current_idx = phases.index(self.current_phase)

        # Sum completed phases
        total = 0.0
        for i, phase in enumerate(phases):
            weight = self._phase_weights.get(phase, 0)
            if i < current_idx:
                total += weight
            elif i == current_idx:
                phase_progress = self._phase_progress.get(phase.value, 0)
                total += weight * phase_progress

        return min(total, 1.0)

    def _progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a text progress bar."""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f'[{bar}]'

    def _phase_icon(self, phase: ResearchPhase) -> str:
        """Get icon for each phase."""
        icons = {
            ResearchPhase.INITIALIZATION: '⚙️ ',
            ResearchPhase.QUERY_DECOMPOSITION: '🔍',
            ResearchPhase.BASELINE_SEARCH: '📚',
            ResearchPhase.PAGINATION: '📄',
            ResearchPhase.CITATION_CRAWLING: '🕸️ ',
            ResearchPhase.DEDUPLICATION: '🔄',
            ResearchPhase.SCREENING: '📋',
            ResearchPhase.RELEVANCE_FILTERING: '🎯',
            ResearchPhase.CLUSTERING: '🗂️ ',
            ResearchPhase.RAG_INDEXING: '🧠',
            ResearchPhase.MAP_SYNTHESIS: '🗺️ ',
            ResearchPhase.REDUCE_SYNTHESIS: '🔬',
            ResearchPhase.DEEP_SYNTHESIS: '🧬',
            ResearchPhase.REPORT_GENERATION: '📝',
            ResearchPhase.COMPLETE: '✅',
        }
        return icons.get(phase, '  ')

    def get_summary(self) -> Dict:
        """Get progress summary."""
        elapsed = time.monotonic() - self.start_time

        # Count papers at latest event
        latest_papers = 0
        for event in reversed(self.events):
            if event.papers_count:
                latest_papers = event.papers_count
                break

        return {
            'current_phase': self.current_phase.value,
            'overall_progress': self._compute_overall_progress(),
            'elapsed_seconds': elapsed,
            'papers_found': latest_papers,
            'events_count': len(self.events),
            'phases_completed': [
                e.phase.value for e in self.events
                if e.progress >= 1.0
            ]
        }

    def get_timeline(self) -> List[Dict]:
        """Get event timeline."""
        return [
            {
                'phase': e.phase.value,
                'message': e.message,
                'progress': e.progress,
                'papers': e.papers_count,
                'timestamp': e.timestamp
            }
            for e in self.events
        ]
