"""
Finding Domain Models — A confirmed vulnerability match in a scanned project.

A Finding is produced when the CVE Analyst agent confirms that a specific
dependency version in a scanned repository is affected by a known CVE.

Architectural note:
    CVERecord is the raw vulnerability knowledge.
    Finding is the contextualised match — it knows WHICH repo,
    WHICH file, WHICH version triggered it.
    
    Keeping these separate lets us store CVE knowledge once
    and reference it across many findings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from vulnremedy.models.cve import CVERecord, Ecosystem, Severity    

class FindingStatus(str, Enum):
    """
    Lifecycle state of a finding.
    
    Tracks the finding from discovery through to resolution.
    Used by the Orchestrator to route findings and by the API
    to filter findings by state.
    """
    OPEN = "open"                           # Newly discovered, not yet actioned
    CONFIRMED = "confirmed"                 # Human or agent verified it is real
    FALSE_POSITIVE = "false_positive"       # Confirmed not exploitable in this context
    REMEDIATION_PLANNED = "remediation_planned"  # Fix is being prepared
    REMEDIATED = "remediated"               # Fix applied and verified
    ACCEPTED_RISK = "accepted_risk"         # Consciously accepted, documented


class DependencyType(str, Enum):
    """
    Whether the vulnerable dependency is pulled in directly
    or transitively through another library.
    
    This affects remediation strategy significantly:
    - Direct deps: fixed by updating version in manifest file
    - Transitive deps: may need dependency exclusions or BOM overrides
    """
    DIRECT = "direct"
    TRANSITIVE = "transitive"
    UNKNOWN = "unknown"

class AffectedDependency(BaseModel):
    """
    The specific dependency instance that triggered this finding.
    
    Tied to a file location in the scanned repository.
    This is what distinguishes a Finding from a CVERecord —
    we know exactly WHERE in the codebase the vulnerable dependency exists.
    """
    package_name: str
    group_id: Optional[str] = None
    current_version: str = Field(
        description="The version actually in use — the vulnerable one"
    )
    fixed_version: Optional[str] = Field(
        default=None,
        description="The version that resolves this CVE"
    )
    ecosystem: Ecosystem
    dependency_type: DependencyType = DependencyType.UNKNOWN
    manifest_file: str = Field(
        description="Path to file where dependency is declared e.g. 'pom.xml', 'package.json'"
    )
    manifest_line: Optional[int] = Field(
        default=None,
        description="Line number in manifest file — enables precise PR changes"
    )

    @property
    def fully_qualified_name(self) -> str:
        """
        Returns Maven-style fully qualified name where applicable.
        
        e.g. 'org.apache.logging.log4j:log4j-core'
        
        For other ecosystems returns package_name directly.
        """
        if self.group_id:
            return f"{self.group_id}:{self.package_name}"
        return self.package_name

class Finding(BaseModel):
    """
    A confirmed vulnerability match in a scanned repository.

    Architectural note:
        Finding.id is a UUID generated at creation time.
        This ID is used as a correlation key across:
        - the vector store (for pattern matching against past findings)
        - the audit log (for tracing every action taken on this finding)
        - the remediation plan (linking fix back to root finding)
        
        This makes every finding traceable end-to-end through the system.
    """
    id: UUID = Field(default_factory=uuid4)
    scan_id: UUID = Field(
        description="The scan that produced this finding — links back to ScanRecord"
    )
    repository: str = Field(
        description="Repository identifier e.g. 'github.com/org/payment-service'"
    )
    branch: str = Field(
        default="main",
        description="Branch that was scanned"
    )
    cve: CVERecord = Field(
        description="The full CVE record this finding is based on"
    )
    affected_dependency: AffectedDependency
    status: FindingStatus = FindingStatus.OPEN
    severity: Severity = Field(
        description="Severity at time of finding — cached from CVE for performance"
    )
    discovered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this finding was first discovered"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last status change timestamp"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Human or agent notes — e.g. 'Not exploitable - code path never reached'"
    )

    @property
    def is_critical_or_high(self) -> bool:
        """
        Priority routing decision helper.
        
        Used by Orchestrator to fast-track high-severity findings.
        """
        return self.severity in (Severity.CRITICAL, Severity.HIGH)

    @property
    def needs_human_review(self) -> bool:
        """
        Determines if this finding requires human approval before remediation.
        
        Business rule: All CRITICAL findings need human review.
        MEDIUM and below can be auto-remediated if confidence is high.
        """
        return self.severity == Severity.CRITICAL

    def mark_as_false_positive(self, reason: str) -> None:
        """
        Transition finding to FALSE_POSITIVE status.
        
        Updates status, adds reason to notes, and updates timestamp.
        This method ensures state transitions are auditable.
        """
        self.status = FindingStatus.FALSE_POSITIVE
        self.notes = f"False positive: {reason}"
        self.updated_at = datetime.utcnow()

    def mark_as_remediated(self, pr_url: str) -> None:
        """
        Transition finding to REMEDIATED status.
        
        Called by the PR Creator agent after successfully creating
        and merging a fix PR.
        """
        self.status = FindingStatus.REMEDIATED
        self.notes = f"Remediated via PR: {pr_url}"
        self.updated_at = datetime.utcnow()
