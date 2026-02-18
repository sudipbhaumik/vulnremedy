"""
Tool Domain Models — MCP tool call contracts.

These models define the request/response schemas for every external tool
that agents can invoke via the MCP (Model Context Protocol) pattern.

Architectural note:
    Every tool has:
    - A Request model (typed input parameters)
    - A Response model (typed output with success/error states)
    - A ToolDefinition (metadata the LLM reads to decide when to use the tool)
    
    This triple ensures:
    - Agents can't pass malformed requests to tools
    - Tools can't return malformed responses to agents
    - LLMs have clear descriptions of what each tool does
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """
    Execution status of a tool call.
    
    Used by the resilience layer to decide retry/fallback behavior.
    """
    SUCCESS = "success"                 # Tool executed successfully
    FAILURE = "failure"                 # Tool failed but can retry
    FATAL_ERROR = "fatal_error"         # Unrecoverable error, do not retry
    RATE_LIMITED = "rate_limited"       # API rate limit hit, back off


class ToolRequest(BaseModel):
    """
    Base class for all tool requests.
    
    Every tool-specific request inherits from this.
    Provides common fields for tracking and correlation.
    """
    request_id: str = Field(
        description="Unique ID for this tool call — used for tracing and correlation"
    )
    timeout_seconds: int = Field(
        default=30,
        description="Maximum time to wait for tool response"
    )


class ToolResponse(BaseModel):
    """
    Base class for all tool responses.
    
    Every tool-specific response inherits from this.
    Standardizes success/error handling across all tools.
    """
    request_id: str = Field(
        description="Matches the request_id from ToolRequest — correlation key"
    )
    status: ToolStatus
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if status is FAILURE or FATAL_ERROR"
    )
    execution_time_ms: int = Field(
        description="How long the tool took to execute in milliseconds"
    )


# ============================================================================
# GitHub Tool — Fetch dependency manifests from repositories
# ============================================================================

class GitHubFetchRequest(ToolRequest):
    """
    Request to fetch a file from a GitHub repository.
    
    Used by Scanner agent to retrieve pom.xml, package.json, requirements.txt
    """
    repository: str = Field(
        description="Repository in format 'owner/repo' e.g. 'apache/logging-log4j2'"
    )
    file_path: str = Field(
        description="Path to file within repo e.g. 'pom.xml', 'package.json'"
    )
    branch: str = Field(
        default="main",
        description="Branch to fetch from"
    )


class GitHubFetchResponse(ToolResponse):
    """
    Response from GitHub file fetch.
    
    Contains file content if successful, error message if not found.
    """
    file_content: Optional[str] = Field(
        default=None,
        description="Raw file content as string"
    )
    file_size_bytes: Optional[int] = Field(
        default=None,
        description="File size for logging and metrics"
    )
    commit_sha: Optional[str] = Field(
        default=None,
        description="SHA of commit where this file was fetched — for audit trail"
    )


# ============================================================================
# NVD Tool — Query CVE details from National Vulnerability Database
# ============================================================================

class NVDQueryRequest(ToolRequest):
    """
    Request to query NVD for CVE details.
    
    Used by CVE Analyst agent to fetch full CVE records.
    """
    cve_id: str = Field(
        description="CVE identifier e.g. 'CVE-2021-44228'",
        pattern=r"^CVE-\d{4}-\d{4,}$"
    )


class NVDQueryResponse(ToolResponse):
    """
    Response from NVD CVE query.
    
    Contains raw NVD API response — will be parsed into CVERecord by ingestion layer.
    """
    cve_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Raw NVD API response as dict — to be parsed into CVERecord"
    )


# ============================================================================
# OSV Tool — Query vulnerabilities from OSV.dev database
# ============================================================================

class OSVQueryRequest(ToolRequest):
    """
    Request to query OSV.dev for package vulnerabilities.
    
    Used by CVE Analyst agent — OSV is often faster and has better
    ecosystem-specific data than NVD.
    """
    package_name: str = Field(
        description="Package name e.g. 'log4j-core', 'lodash'"
    )
    ecosystem: str = Field(
        description="Ecosystem e.g. 'Maven', 'npm', 'PyPI'"
    )
    version: str = Field(
        description="Specific version to check e.g. '2.14.1'"
    )


class OSVQueryResponse(ToolResponse):
    """
    Response from OSV.dev query.
    
    Contains list of vulnerabilities affecting the queried package+version.
    """
    vulnerabilities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of vulnerability records from OSV — to be parsed into CVERecords"
    )
    total_count: int = Field(
        default=0,
        description="Total number of vulnerabilities found"
    )


# ============================================================================
# deps.dev Tool — Analyze dependency graphs and transitive dependencies
# ============================================================================

class DepsDevQueryRequest(ToolRequest):
    """
    Request to query deps.dev for dependency graph.
    
    Used by Impact Assessor agent to trace blast radius —
    which services depend on the vulnerable package.
    """
    package_name: str = Field(
        description="Package name e.g. 'log4j-core'"
    )
    ecosystem: str = Field(
        description="Ecosystem e.g. 'Maven', 'npm'"
    )
    version: str = Field(
        description="Version e.g. '2.14.1'"
    )


class DepsDevQueryResponse(ToolResponse):
    """
    Response from deps.dev query.
    
    Contains dependency graph — who depends on this package.
    """
    dependencies: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of packages that depend on the queried package"
    )
    transitive_count: int = Field(
        default=0,
        description="Number of transitive dependencies found"
    )


# ============================================================================
# Code Analysis Tool — Generate code fixes via LLM
# ============================================================================

class CodeAnalysisRequest(ToolRequest):
    """
    Request to generate code fix for a vulnerability.
    
    Used by Remediation Planner agent when strategy requires code changes.
    This is an LLM-powered tool, not an external API.
    """
    original_code: str = Field(
        description="Original code snippet that needs fixing"
    )
    vulnerability_description: str = Field(
        description="What vulnerability exists in this code"
    )
    target_version: str = Field(
        description="Target library version after upgrade"
    )
    ecosystem: str = Field(
        description="Ecosystem for context e.g. 'Maven', 'npm'"
    )


class CodeAnalysisResponse(ToolResponse):
    """
    Response from code analysis tool.
    
    Contains suggested code fix and rationale.
    """
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Fixed code snippet"
    )
    changes_required: list[str] = Field(
        default_factory=list,
        description="List of changes made — for human review"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM's confidence in this fix (0.0 to 1.0)"
    )


# ============================================================================
# PR Creator Tool — Create GitHub pull request
# ============================================================================

class PRCreateRequest(ToolRequest):
    """
    Request to create a pull request on GitHub.
    
    Used by PR Creator agent after remediation plan is approved.
    """
    repository: str = Field(
        description="Repository in format 'owner/repo'"
    )
    branch_name: str = Field(
        description="Name for the fix branch e.g. 'fix/cve-2021-44228'"
    )
    title: str = Field(
        description="PR title e.g. 'Fix CVE-2021-44228 in log4j-core'"
    )
    body: str = Field(
        description="PR description with fix details and testing notes"
    )
    file_changes: list[dict[str, str]] = Field(
        description="List of file changes: [{path, content}, ...]"
    )


class PRCreateResponse(ToolResponse):
    """
    Response from PR creation.
    
    Contains PR URL if successful.
    """
    pr_url: Optional[str] = Field(
        default=None,
        description="URL of created PR e.g. 'https://github.com/org/repo/pull/123'"
    )
    pr_number: Optional[int] = Field(
        default=None,
        description="PR number for API operations"
    )