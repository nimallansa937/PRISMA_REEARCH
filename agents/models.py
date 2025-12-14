"""
Data Models for Multi-Agent Research Protocol.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class GapType(Enum):
    METHODOLOGICAL = "methodological"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    OUTCOME = "outcome"
    ASSET_CLASS = "asset_class"


class GapSeverity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ResearchGap:
    """Identified gap in literature coverage"""
    gap_type: GapType
    description: str
    severity: GapSeverity
    suggested_query: str
    affected_dimensions: List[str] = field(default_factory=list)


@dataclass
class RefinementQuery:
    """Query generated to fill a specific gap"""
    query: str
    target_gap_description: str
    expected_yield: str
    database_priority: List[str] = field(default_factory=list)


@dataclass
class CoverageMatrix:
    """Tracks coverage across different dimensions"""
    dimensions: Dict[str, Dict[str, int]]
    total_papers: int
    coverage_score: float  # 0-100
    
    def to_dict(self) -> Dict:
        return {
            "dimensions": self.dimensions,
            "total_papers": self.total_papers,
            "coverage_score": self.coverage_score
        }


@dataclass
class CrossCuttingPattern:
    """Emergent pattern across multiple papers"""
    pattern_name: str
    description: str
    supporting_paper_indices: List[int]
    actionable_insight: str
    confidence: float
