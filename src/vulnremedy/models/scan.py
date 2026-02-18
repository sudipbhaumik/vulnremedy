"""
Scan Domain Models — Top-level scan operation and results.

A Scan represents a complete vulnerability analysis run against
one or more repositories. It contains all findings discovered,
execution metrics, and lifecycle status.

Architectural note:
    ScanRecord is the root aggregate in DDD terms.
    All findings belong to a scan.
    Scans are created by the API layer and executed by agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from vulnremedy.models.finding import Finding

class ScanStatus(str, Enum):
    """
    Lifecycle state of a scan operation.
    
    Used by the API to track scan progress and by the worker
    to update scan state as it executes.
    """
    QUEUED = "queued"           # Scan requested, waiting for worker
    RUNNING = "running"         # Worker is actively scanning
    COMPLETED = "completed"     # Scan finished successfully
    FAILED = "failed"           # Scan encountered unrecoverable error
    CANCELLED = "cancelled"     # User cancelled the scan

class ScanRequest(BaseModel):
    """
    Input parameters that triggered a scan.
    
    Represents what the user or API caller requested.
    Immutable once the scan starts — captures intent at request time.
    """
    repositories: list[str] = Field(
        description="List of repository URLs to scan e.g. ['github.com/org/service-a']"
    )
    branch: str = Field(
        default="main",
        description="Branch to scan — defaults to main"
    )
    scan_transitive_deps: bool = Field(
        default=True,
        description="Whether to analyze transitive dependencies or direct only"
    )
    severity_threshold: Optional[str] = Field(
        default=None,
        description="Only report findings at or above this severity e.g. 'high' filters out medium/low"
    )
    auto_remediate: bool = Field(
        default=False,
        description="If true, automatically create PRs for findings without human approval (DANGEROUS)"
    )

class ScanRecord(BaseModel):
    """
    Complete scan operation with results and metrics.
    
    This is the root aggregate — contains all findings,
    execution metadata, and lifecycle state.
    
    Architectural note:
        ScanRecord.id is the correlation key for the entire scan.
        Every Finding.scan_id references this ID.
        Every audit log entry for this scan references this ID.
        This makes scans fully traceable end-to-end.
    """
    id: UUID = Field(default_factory=uuid4)
    request: ScanRequest = Field(
        description="The original request that triggered this scan"
    )
    status: ScanStatus = ScanStatus.QUEUED
    findings: list[Finding] = Field(
        default_factory=list,
        description="All findings discovered during this scan"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="When scan execution began (None if still queued)"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When scan finished (None if still running or failed)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if status is FAILED"
    )
    
    # Execution metrics
    total_dependencies_scanned: int = Field(
        default=0,
        description="Total number of dependencies analyzed"
    )
    total_cves_checked: int = Field(
        default=0,
        description="Total number of CVE database queries made"
    )
    execution_time_seconds: Optional[float] = Field(
        default=None,
        description="Total scan duration in seconds"
    )

    @property
    def critical_findings_count(self) -> int:
        """Count of CRITICAL severity findings."""
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_findings_count(self) -> int:
        """Count of HIGH severity findings."""
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def total_findings_count(self) -> int:
        """Total number of findings."""
        return len(self.findings)

    def mark_as_running(self) -> None:
        """
        Transition scan to RUNNING status.
        
        Called by worker when it picks up the scan.
        Sets started_at timestamp.
        """
        self.status = ScanStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_as_completed(self) -> None:
        """
        Transition scan to COMPLETED status.
        
        Called by worker when scan finishes successfully.
        Sets completed_at timestamp and calculates execution time.
        """
        self.status = ScanStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.execution_time_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    def mark_as_failed(self, error: str) -> None:
        """
        Transition scan to FAILED status.
        
        Called by worker if agents crash or external APIs fail.
        Records error message for debugging.
        """
        self.status = ScanStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        if self.started_at:
            self.execution_time_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()