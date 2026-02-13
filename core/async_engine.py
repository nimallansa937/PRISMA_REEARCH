"""
Async Engine - Parallel database queries with rate limiting.
Upgrade #1: Replaces sequential requests with asyncio + aiohttp.
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RateLimiter:
    """Per-source async rate limiter using token bucket algorithm."""
    max_requests: int
    period: float = 60.0  # seconds
    _tokens: float = 0
    _last_refill: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        self._tokens = float(self.max_requests)

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self.max_requests,
                self._tokens + (elapsed * self.max_requests / self.period)
            )
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) * self.period / self.max_requests
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


@dataclass
class SearchTask:
    """A single search task to be executed asynchronously."""
    query: str
    source_name: str
    limit: int = 50
    offset: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from an async search task."""
    task: SearchTask
    papers: List[Dict] = field(default_factory=list)
    total_available: int = 0
    error: Optional[str] = None
    elapsed_ms: float = 0


class AsyncSearchEngine:
    """
    Executes search tasks in parallel across multiple sources.
    Handles rate limiting, retries, and connection pooling.
    """

    def __init__(self, timeout: int = 45, max_retries: int = 2):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    def set_rate_limit(self, source_name: str, max_requests: int, period: float = 60.0):
        self.rate_limiters[source_name] = RateLimiter(max_requests=max_requests, period=period)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def execute_search(
        self,
        task: SearchTask,
        search_fn: Callable,
    ) -> SearchResult:
        """Execute a single search with rate limiting and retries."""
        start = time.monotonic()

        # Apply rate limiting
        limiter = self.rate_limiters.get(task.source_name)
        if limiter:
            await limiter.acquire()

        for attempt in range(self.max_retries + 1):
            try:
                papers = await search_fn(task.query, task.limit, task.offset)
                elapsed = (time.monotonic() - start) * 1000

                return SearchResult(
                    task=task,
                    papers=papers,
                    total_available=len(papers),
                    elapsed_ms=elapsed
                )

            except Exception as e:
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                else:
                    elapsed = (time.monotonic() - start) * 1000
                    return SearchResult(
                        task=task,
                        error=str(e),
                        elapsed_ms=elapsed
                    )

    async def execute_parallel(
        self,
        tasks: List[SearchTask],
        search_fns: Dict[str, Callable],
        max_concurrent: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> List[SearchResult]:
        """Execute multiple search tasks in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        completed = 0

        async def _run_task(task: SearchTask) -> SearchResult:
            nonlocal completed
            async with semaphore:
                fn = search_fns.get(task.source_name)
                if fn is None:
                    return SearchResult(task=task, error=f"Unknown source: {task.source_name}")

                result = await self.execute_search(task, fn)
                completed += 1

                if progress_callback:
                    await progress_callback(completed, len(tasks), result)

                return result

        # Run all tasks concurrently
        coros = [_run_task(t) for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=False)

        return results

    async def execute_paginated(
        self,
        query: str,
        source_name: str,
        search_fn: Callable,
        target_count: int = 200,
        page_size: int = 50,
        max_pages: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Fetch multiple pages from a single source until target reached."""
        all_papers = []
        offset = 0

        for page in range(max_pages):
            if len(all_papers) >= target_count:
                break

            remaining = target_count - len(all_papers)
            limit = min(page_size, remaining)

            task = SearchTask(
                query=query,
                source_name=source_name,
                limit=limit,
                offset=offset
            )

            result = await self.execute_search(task, search_fn)

            if result.error or not result.papers:
                break

            all_papers.extend(result.papers)
            offset += len(result.papers)

            if progress_callback:
                await progress_callback(len(all_papers), target_count, result)

            # Stop if source returned fewer than requested (no more results)
            if len(result.papers) < limit:
                break

        return all_papers[:target_count]


def run_async(coro):
    """Helper to run async code from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing event loop (e.g., Jupyter)
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
