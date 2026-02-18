"""
API Domain Models — HTTP request/response contracts.

These models define the FastAPI route schemas — what clients send
to the API and what the API returns.

Architectural note:
    API models are deliberately separate from domain models.
    
    Why?
    - API responses might simplify domain models for client consumption
    - API versioning can evolve independently of internal domain
    - API models can aggregate multiple domain models into single response
    
    The API layer translates between external API contracts
    and internal domain models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from vulnremedy.models.finding import FindingStatus
from vulnremedy.models.remediation import RemediationStatus
from vulnremedy.models.scan import ScanRequest, ScanStatus


# ============================================================================
# Scan API — Create and query scans
# ============================================================================

class CreateScanRequest(BaseModel):
    """
    API request to create a new scan.
    
    POST /api/v1/scans
    """
    repositories: list[str] = Field(
        min_length=1,
        description="List of repository URLs to scan — at least one required"
    )
    branch: str = Field(
        default="main",
        description="Branch to scan"
    )
    scan_transitive_deps: bool = Field(
        default=True,
        description="Include transitive dependencies in analysis"
    )
    severity_threshold: Optional[str] = Field(
        default=None,
        description="Minimum severity to report: 'low', 'medium', 'high', 'critical'"
    )
    auto_remediate: bool = Field(
        default=False,
        description="Auto-create PRs without human approval (use with caution)"
    )


class CreateScanResponse(BaseModel):
    """
    API response after creating a scan.
    
    Returns scan ID and status.
    """
    scan_id: UUID = Field(
        description="Unique ID for this scan — use to query status"
    )
    status: ScanStatus = Field(
        description="Initial status — always QUEUED"
    )
    message: str = Field(
        default="Scan queued successfully",
        description="Human-readable status message"
    )


class GetScanResponse(BaseModel):
    """
    API response for scan details.
    
    GET /api/v1/scans/{scan_id}
    """
    scan_id: UUID
    status: ScanStatus
    repositories: list[str]
    branch: str
    
    # Findings summary
    total_findings: int = Field(
        default=0,
        description="Total number of vulnerabilities found"
    )
    critical_findings: int = Field(default=0)
    high_findings: int = Field(default=0)
    medium_findings: int = Field(default=0)
    low_findings: int = Field(default=0)
    
    # Execution metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    
    error_message: Optional[str] = None


class ListScansResponse(BaseModel):
    """
    API response for listing scans.
    
    GET /api/v1/scans
    """
    scans: list[GetScanResponse] = Field(
        default_factory=list,
        description="List of scans, newest first"
    )
    total: int = Field(
        description="Total number of scans matching filter"
    )
    page: int = Field(
        default=1,
        description="Current page number"
    )
    page_size: int = Field(
        default=20,
        description="Number of results per page"
    )


# ============================================================================
# Finding API — Query and update findings
# ============================================================================

class FindingResponse(BaseModel):
    """
    API response for a single finding.
    
    GET /api/v1/findings/{finding_id}
    """
    finding_id: UUID
    scan_id: UUID
    repository: str
    branch: str
    
    # CVE details
    cve_id: str
    cve_description: str
    severity: str
    cvss_score: Optional[float] = None
    
    # Affected dependency
    package_name: str
    current_version: str
    fixed_version: Optional[str] = None
    ecosystem: str
    dependency_type: str
    manifest_file: str
    manifest_line: Optional[int] = None
    
    # Status
    status: FindingStatus
    discovered_at: datetime
    
    notes: Optional[str] = None


class ListFindingsResponse(BaseModel):
    """
    API response for listing findings.
    
    GET /api/v1/scans/{scan_id}/findings
    """
    findings: list[FindingResponse] = Field(
        default_factory=list
    )
    total: int
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class MarkFalsePositiveRequest(BaseModel):
    """
    API request to mark finding as false positive.
    
    POST /api/v1/findings/{finding_id}/false-positive
    """
    reason: str = Field(
        min_length=10,
        description="Why this is a false positive — minimum 10 characters"
    )


# ============================================================================
# Remediation API — Approve/reject remediation plans
# ============================================================================

class RemediationPlanResponse(BaseModel):
    """
    API response for a remediation plan.
    
    GET /api/v1/remediation-plans/{plan_id}
    """
    plan_id: UUID
    finding_id: UUID
    status: RemediationStatus
    strategy: str
    
    # Plan details
    summary: str
    detailed_steps: list[str] = Field(default_factory=list)
    breaking_changes: list[str] = Field(default_factory=list)
    testing_recommendations: list[str] = Field(default_factory=list)
    
    # Code changes preview
    files_changed: list[str] = Field(
        default_factory=list,
        description="List of file paths that will be modified"
    )
    
    # Approval workflow
    requires_human_approval: bool
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # PR tracking
    pr_url: Optional[str] = None
    
    created_at: datetime


class ApprovePlanRequest(BaseModel):
    """
    API request to approve a remediation plan.
    
    POST /api/v1/remediation-plans/{plan_id}/approve
    """
    approver: str = Field(
        description="Username or email of person approving"
    )
    comment: Optional[str] = Field(
        default=None,
        description="Optional approval comment"
    )


class RejectPlanRequest(BaseModel):
    """
    API request to reject a remediation plan.
    
    POST /api/v1/remediation-plans/{plan_id}/reject
    """
    reason: str = Field(
        min_length=10,
        description="Why this plan is rejected — minimum 10 characters"
    )


class ListRemediationPlansResponse(BaseModel):
    """
    API response for listing remediation plans.
    
    GET /api/v1/scans/{scan_id}/remediation-plans
    """
    plans: list[RemediationPlanResponse] = Field(
        default_factory=list
    )
    total: int
    pending_approval_count: int = Field(
        default=0,
        description="How many plans are waiting for approval"
    )


# ============================================================================
# Health and Status API
# ============================================================================

class HealthResponse(BaseModel):
    """
    API response for health check.
    
    GET /api/v1/health
    """
    status: str = Field(
        default="healthy",
        description="Overall system health: 'healthy', 'degraded', 'unhealthy'"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services: {service_name: status}"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response for all API errors.
    
    Returned for 4xx and 5xx responses.
    """
    error: str = Field(
        description="Error type e.g. 'ValidationError', 'NotFound', 'InternalServerError'"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error context for debugging"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )