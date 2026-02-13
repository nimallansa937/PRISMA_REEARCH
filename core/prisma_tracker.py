"""
PRISMA Flow Tracker - Upgrade #10.
Tracks papers through each stage of systematic review methodology.
Provides quality tiers and generates PRISMA flow diagrams.

PRISMA stages:
  Identification → Screening → Eligibility → Included
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict


class PRISMAStage(Enum):
    IDENTIFIED = "identified"
    DEDUPLICATED = "deduplicated"
    SCREENED = "screened"
    ELIGIBLE = "eligible"
    INCLUDED = "included"


class ExclusionReason(Enum):
    DUPLICATE = "Duplicate paper"
    OFF_TOPIC = "Off-topic / irrelevant"
    LOW_QUALITY = "Below quality threshold"
    NO_ABSTRACT = "Missing abstract"
    TOO_OLD = "Outside date range"
    WRONG_TYPE = "Wrong publication type"
    LANGUAGE = "Non-English"
    RETRACTED = "Retracted paper"
    INSUFFICIENT_DATA = "Insufficient methodological data"


@dataclass
class PRISMARecord:
    """Tracks a single paper through PRISMA stages."""
    paper_id: str
    current_stage: PRISMAStage
    source: str
    exclusion_reason: Optional[ExclusionReason] = None
    exclusion_detail: str = ""
    quality_tier: str = ""  # "A", "B", "C"
    relevance_score: float = 0.0
    timestamps: Dict[str, str] = field(default_factory=dict)


class PRISMATracker:
    """
    Tracks the systematic review PRISMA flow.
    Provides counts at each stage and reasons for exclusion.
    """

    def __init__(self):
        self.records: Dict[str, PRISMARecord] = {}
        self.stage_counts: Dict[str, int] = defaultdict(int)
        self.exclusion_counts: Dict[str, int] = defaultdict(int)
        self.source_counts: Dict[str, int] = defaultdict(int)
        self.events: List[Dict] = []

    def add_identified(self, papers: List[Dict], source: str):
        """Stage 1: Record papers as identified from a source."""
        added = 0
        for p in papers:
            pid = p.get('paper_id') or p.get('doi') or p.get('title', '')[:100]
            if pid and pid not in self.records:
                self.records[pid] = PRISMARecord(
                    paper_id=pid,
                    current_stage=PRISMAStage.IDENTIFIED,
                    source=source,
                    timestamps={'identified': datetime.now().isoformat()}
                )
                added += 1
                self.source_counts[source] += 1

        self._log_event('identified', source, added, len(papers))

    def mark_deduplicated(self, removed_ids: List[str]):
        """Stage 2: Mark papers removed as duplicates."""
        for pid in removed_ids:
            if pid in self.records:
                self.records[pid].current_stage = PRISMAStage.DEDUPLICATED
                self.records[pid].exclusion_reason = ExclusionReason.DUPLICATE
                self.records[pid].timestamps['removed'] = datetime.now().isoformat()
                self.exclusion_counts['duplicate'] += 1

        surviving = [r for r in self.records.values()
                     if r.current_stage == PRISMAStage.IDENTIFIED]
        for r in surviving:
            r.current_stage = PRISMAStage.DEDUPLICATED

        self._log_event('deduplicated', '', len(removed_ids), 0)

    def mark_screened(self, paper_id: str, passed: bool,
                      reason: ExclusionReason = None, detail: str = ""):
        """Stage 3: Title/abstract screening result."""
        if paper_id in self.records:
            record = self.records[paper_id]
            if passed:
                record.current_stage = PRISMAStage.SCREENED
                record.timestamps['screened'] = datetime.now().isoformat()
            else:
                record.exclusion_reason = reason or ExclusionReason.OFF_TOPIC
                record.exclusion_detail = detail
                record.timestamps['excluded_screening'] = datetime.now().isoformat()
                self.exclusion_counts[f'screening:{record.exclusion_reason.value}'] += 1

    def mark_eligible(self, paper_id: str, passed: bool,
                      relevance_score: float = 0.0,
                      reason: ExclusionReason = None, detail: str = ""):
        """Stage 4: Full eligibility assessment."""
        if paper_id in self.records:
            record = self.records[paper_id]
            record.relevance_score = relevance_score
            if passed:
                record.current_stage = PRISMAStage.ELIGIBLE
                record.timestamps['eligible'] = datetime.now().isoformat()
            else:
                record.exclusion_reason = reason or ExclusionReason.LOW_QUALITY
                record.exclusion_detail = detail
                record.timestamps['excluded_eligibility'] = datetime.now().isoformat()
                self.exclusion_counts[f'eligibility:{record.exclusion_reason.value}'] += 1

    def mark_included(self, paper_id: str, quality_tier: str = "B"):
        """Stage 5: Paper included in final synthesis."""
        if paper_id in self.records:
            record = self.records[paper_id]
            record.current_stage = PRISMAStage.INCLUDED
            record.quality_tier = quality_tier
            record.timestamps['included'] = datetime.now().isoformat()

    def assign_quality_tiers(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Assign quality tiers to papers.

        Tier A: High quality (DOI + top venue + 20+ citations + recent)
        Tier B: Medium quality (DOI + some citations OR good venue)
        Tier C: Acceptable (meets minimum threshold)
        """
        tiers = {'A': [], 'B': [], 'C': []}

        for p in papers:
            pid = p.get('paper_id') or p.get('doi', '')
            tier = self._compute_tier(p)
            tiers[tier].append(p)

            if pid in self.records:
                self.records[pid].quality_tier = tier

        return tiers

    def _compute_tier(self, paper: Dict) -> str:
        """Determine quality tier for a paper."""
        score = 0
        if paper.get('doi'):
            score += 2
        cites = paper.get('citation_count', 0) or 0
        if cites >= 50:
            score += 3
        elif cites >= 20:
            score += 2
        elif cites >= 5:
            score += 1

        venue = (paper.get('venue') or '').lower()
        top_venues = ['nature', 'science', 'cell', 'lancet', 'nejm', 'ieee', 'acm',
                      'neurips', 'icml', 'iclr', 'cvpr', 'pnas', 'jama', 'bmj']
        if any(v in venue for v in top_venues):
            score += 3
        elif venue and venue != 'arxiv preprint':
            score += 1

        year = paper.get('year', 0) or 0
        if year >= datetime.now().year - 3:
            score += 1

        if paper.get('abstract') and len(paper.get('abstract', '')) >= 100:
            score += 1

        if score >= 7:
            return 'A'
        elif score >= 4:
            return 'B'
        else:
            return 'C'

    def get_flow_counts(self) -> Dict:
        """Get PRISMA flow counts at each stage."""
        total_identified = len(self.records)

        stage_counts = defaultdict(int)
        for r in self.records.values():
            stage_counts[r.current_stage.value] += 1

        # Count excluded at each stage (check key presence, not substring in timestamp)
        excluded_screening = sum(
            1 for r in self.records.values()
            if r.exclusion_reason and 'excluded_screening' in r.timestamps
        )
        excluded_eligibility = sum(
            1 for r in self.records.values()
            if r.exclusion_reason and 'excluded_eligibility' in r.timestamps
        )

        included = stage_counts.get('included', 0)
        tier_counts = defaultdict(int)
        for r in self.records.values():
            if r.current_stage == PRISMAStage.INCLUDED and r.quality_tier:
                tier_counts[r.quality_tier] += 1

        return {
            'identified': total_identified,
            'duplicates_removed': self.exclusion_counts.get('duplicate', 0),
            'after_dedup': total_identified - self.exclusion_counts.get('duplicate', 0),
            'screened': stage_counts.get('screened', 0) + stage_counts.get('eligible', 0) + included,
            'excluded_screening': excluded_screening,
            'eligible': stage_counts.get('eligible', 0) + included,
            'excluded_eligibility': excluded_eligibility,
            'included': included,
            'quality_tiers': dict(tier_counts),
            'by_source': dict(self.source_counts),
            'exclusion_reasons': dict(self.exclusion_counts)
        }

    def generate_prisma_text(self) -> str:
        """Generate a text-based PRISMA flow diagram."""
        counts = self.get_flow_counts()

        sources_text = ", ".join(
            f"{source}: {count}" for source, count in counts['by_source'].items()
        )

        diagram = f"""
╔══════════════════════════════════════════════════════════════╗
║                    PRISMA FLOW DIAGRAM                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  IDENTIFICATION                                              ║
║  ┌──────────────────────────────────────┐                   ║
║  │ Records identified: {counts['identified']:>6}            │                   ║
║  │ Sources: {sources_text:<30}│                   ║
║  └──────────────────────┬───────────────┘                   ║
║                         │                                    ║
║                         ▼                                    ║
║  ┌──────────────────────────────────────┐                   ║
║  │ Duplicates removed:  {counts['duplicates_removed']:>6}            │                   ║
║  └──────────────────────┬───────────────┘                   ║
║                         │                                    ║
║  SCREENING              ▼                                    ║
║  ┌──────────────────────────────────────┐  ┌─────────────┐ ║
║  │ Records screened:    {counts['screened']:>6}            │→│ Excluded:   │ ║
║  │                                      │  │ {counts['excluded_screening']:>6}        │ ║
║  └──────────────────────┬───────────────┘  └─────────────┘ ║
║                         │                                    ║
║  ELIGIBILITY            ▼                                    ║
║  ┌──────────────────────────────────────┐  ┌─────────────┐ ║
║  │ Full-text assessed:  {counts['eligible']:>6}            │→│ Excluded:   │ ║
║  │                                      │  │ {counts['excluded_eligibility']:>6}        │ ║
║  └──────────────────────┬───────────────┘  └─────────────┘ ║
║                         │                                    ║
║  INCLUDED               ▼                                    ║
║  ┌──────────────────────────────────────┐                   ║
║  │ Studies included:    {counts['included']:>6}            │                   ║
║  │   Tier A (High):     {counts['quality_tiers'].get('A', 0):>6}            │                   ║
║  │   Tier B (Medium):   {counts['quality_tiers'].get('B', 0):>6}            │                   ║
║  │   Tier C (Standard): {counts['quality_tiers'].get('C', 0):>6}            │                   ║
║  └──────────────────────────────────────┘                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        return diagram

    def _log_event(self, stage: str, source: str, count: int, total: int):
        """Log a PRISMA event."""
        self.events.append({
            'stage': stage,
            'source': source,
            'count': count,
            'total': total,
            'timestamp': datetime.now().isoformat()
        })

    def to_dict(self) -> Dict:
        """Export tracker state."""
        return {
            'flow_counts': self.get_flow_counts(),
            'events': self.events,
            'records_count': len(self.records)
        }
