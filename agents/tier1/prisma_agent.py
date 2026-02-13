"""
Tier 1 PRISMA Compliance Agent - Scripted executor (no LLM).
Ensures every paper is tracked through PRISMA stages with audit trail.

Responsibilities:
  - Validate PRISMA stage transitions
  - Generate compliance reports
  - Track exclusion reasons with counts
  - Produce audit-ready PRISMA flow data
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.base_agent import Tier1Agent


class PRISMAComplianceAgent(Tier1Agent):
    """
    Enforces PRISMA methodology compliance.
    No LLM - purely rule-based tracking and validation.
    """

    VALID_TRANSITIONS = {
        'identified': ['deduplicated', 'excluded'],
        'deduplicated': ['screened', 'excluded'],
        'screened': ['eligible', 'excluded'],
        'eligible': ['included', 'excluded'],
        'included': [],
        'excluded': [],
    }

    def __init__(self):
        super().__init__(
            name="PRISMACompliance",
            description="PRISMA methodology compliance tracking"
        )
        self.audit_log: List[Dict] = []
        self.violations: List[Dict] = []

    def execute(self, input_data: Dict) -> Dict:
        """Run compliance check on current PRISMA state."""
        prisma_tracker = input_data.get('prisma_tracker')
        papers = input_data.get('papers', [])
        stage = input_data.get('stage', '')

        report = self.compliance_check(prisma_tracker, papers, stage)
        return report

    def compliance_check(self, prisma_tracker, papers: List[Dict],
                         current_stage: str) -> Dict:
        """Validate PRISMA compliance at current stage."""
        issues = []
        warnings = []

        if prisma_tracker is None:
            return {
                'compliant': False,
                'issues': ['No PRISMA tracker provided'],
                'warnings': [],
                'stage': current_stage
            }

        flow = prisma_tracker.get_flow_counts()

        # Check 1: All papers must be identified before any other stage
        if flow.get('identified', 0) == 0 and current_stage != 'identification':
            issues.append("No papers identified - PRISMA flow not initialized")

        # Check 2: Numbers must be monotonically decreasing
        stages_ordered = ['identified', 'after_dedup', 'screened', 'eligible', 'included']
        prev_count = float('inf')
        for stage_name in stages_ordered:
            count = flow.get(stage_name, 0)
            if count > prev_count:
                warnings.append(
                    f"Stage '{stage_name}' ({count}) > previous stage ({prev_count})"
                )
            prev_count = count

        # Check 3: Exclusion reasons must be documented
        exclusion_reasons = flow.get('exclusion_reasons', {})
        total_excluded = (
            flow.get('excluded_screening', 0) +
            flow.get('excluded_eligibility', 0) +
            flow.get('duplicates_removed', 0)
        )
        total_documented = sum(exclusion_reasons.values())
        if total_excluded > 0 and total_documented < total_excluded * 0.5:
            warnings.append(
                f"Only {total_documented}/{total_excluded} exclusions have documented reasons"
            )

        # Check 4: Source diversity (should have 2+ sources)
        sources = flow.get('by_source', {})
        if len(sources) < 2:
            warnings.append(
                f"Only {len(sources)} source(s) used - PRISMA recommends 2+"
            )

        # Check 5: Inclusion rate sanity check
        identified = flow.get('identified', 1)
        included = flow.get('included', 0)
        inclusion_rate = included / max(identified, 1)
        if inclusion_rate > 0.95:
            warnings.append(
                f"Inclusion rate {inclusion_rate:.0%} is unusually high - "
                "review screening criteria"
            )
        elif inclusion_rate < 0.01 and identified > 100:
            warnings.append(
                f"Inclusion rate {inclusion_rate:.2%} is very low - "
                "review search strategy"
            )

        # Check 6: Papers without IDs
        papers_without_id = sum(
            1 for p in papers
            if not p.get('paper_id') and not p.get('doi')
        )
        if papers_without_id > 0:
            warnings.append(
                f"{papers_without_id} papers lack unique identifiers"
            )

        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'stage': current_stage,
            'issues': len(issues),
            'warnings': len(warnings),
            'flow_snapshot': flow
        })

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stage': current_stage,
            'flow_counts': flow,
            'source_diversity': len(sources),
            'inclusion_rate': inclusion_rate if identified > 0 else 0,
            'audit_entries': len(self.audit_log)
        }

    def validate_transition(self, paper_id: str, from_stage: str,
                           to_stage: str) -> bool:
        """Validate a PRISMA stage transition."""
        valid_next = self.VALID_TRANSITIONS.get(from_stage, [])
        if to_stage not in valid_next:
            self.violations.append({
                'paper_id': paper_id,
                'from': from_stage,
                'to': to_stage,
                'timestamp': datetime.now().isoformat(),
                'reason': f"Invalid transition: {from_stage} -> {to_stage}"
            })
            return False
        return True

    def generate_audit_report(self) -> str:
        """Generate a text audit report."""
        lines = [
            "=" * 60,
            "  PRISMA COMPLIANCE AUDIT REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"  Audit Entries: {len(self.audit_log)}",
            f"  Violations: {len(self.violations)}",
            "",
        ]

        for entry in self.audit_log:
            lines.append(f"  [{entry['timestamp']}] Stage: {entry['stage']}")
            lines.append(f"    Issues: {entry['issues']}, Warnings: {entry['warnings']}")

        if self.violations:
            lines.append("\n  VIOLATIONS:")
            for v in self.violations:
                lines.append(f"    - {v['paper_id']}: {v['reason']}")

        lines.append("=" * 60)
        return "\n".join(lines)
