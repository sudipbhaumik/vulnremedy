"""
Agent Domain Models — Agent message contracts and state.

These models define the typed messages that agents exchange
when executing in the LangGraph orchestration flow.

Architectural note:
    LangGraph agents communicate by reading from and writing to
    a shared state object. These models define the schema of that state.
    
    Every agent receives state, performs work, and returns updated state.
    Type safety here prevents entire classes of inter-agent bugs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from vulnremedy.models.cve import CVERecord
from vulnremedy.models.finding import Finding
from vulnremedy.models.remediation import RemediationPlan
from vulnremedy.models.scan import ScanRequest


class AgentMessage(BaseModel):
    """
    Base class for all agent messages.
    
    Provides common fields for tracing and debugging agent workflows.
    """
    agent_name: str = Field(
        description="Which agent produced this message e.g. 'scanner', 'cve_analyst'"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this message was created"
    )
    correlation_id: UUID = Field(
        description="Scan ID — ties all messages in a workflow together"
    )


class ScannerOutput(AgentMessage):
    """
    Output from the Scanner agent.
    
    Contains all dependencies discovered in the target repositories.
    Passed to CVE Analyst agent for vulnerability checking.
    """
    repositories_scanned: list[str] = Field(
        description="List of repository URLs that were scanned"
    )
    dependencies_found: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of dependencies with {package, version, ecosystem, manifest_file, line_number}"
    )
    total_dependencies: int = Field(
        default=0,
        description="Total count of dependencies discovered"
    )
    scan_errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered during scanning"
    )


class CVEAnalystOutput(AgentMessage):
    """
    Output from the CVE Analyst agent.
    
    Contains all findings — confirmed vulnerability matches.
    Passed to Impact Assessor agent for blast radius analysis.
    """
    findings: list[Finding] = Field(
        default_factory=list,
        description="All vulnerability findings discovered"
    )
    cves_checked: int = Field(
        default=0,
        description="Total number of CVE database queries made"
    )
    critical_count: int = Field(
        default=0,
        description="Number of CRITICAL severity findings"
    )
    high_count: int = Field(
        default=0,
        description="Number of HIGH severity findings"
    )


class ImpactAssessorOutput(AgentMessage):
    """
    Output from the Impact Assessor agent.
    
    Contains blast radius analysis — which services are affected,
    how many users impacted, criticality assessment.
    Passed to Remediation Planner agent.
    """
    findings_with_impact: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Findings enriched with blast radius data"
    )
    high_priority_findings: list[UUID] = Field(
        default_factory=list,
        description="Finding IDs that need immediate remediation"
    )
    affected_services: list[str] = Field(
        default_factory=list,
        description="List of service names affected by vulnerabilities"
    )


class RemediationPlannerOutput(AgentMessage):
    """
    Output from the Remediation Planner agent.
    
    Contains remediation plans for each finding.
    Plans requiring approval go to human checkpoint.
    Auto-approved plans go directly to PR Creator agent.
    """
    remediation_plans: list[RemediationPlan] = Field(
        default_factory=list,
        description="All generated remediation plans"
    )
    plans_requiring_approval: list[UUID] = Field(
        default_factory=list,
        description="Plan IDs that need human review"
    )
    auto_approved_plans: list[UUID] = Field(
        default_factory=list,
        description="Plan IDs that can proceed automatically"
    )


class PRCreatorOutput(AgentMessage):
    """
    Output from the PR Creator agent.
    
    Contains results of PR creation attempts.
    This is the final agent in the workflow.
    """
    prs_created: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of created PRs with {plan_id, pr_url, repository}"
    )
    creation_failures: list[dict[str, str]] = Field(
        default_factory=list,
        description="Failed PR attempts with {plan_id, error_message}"
    )
    total_prs: int = Field(
        default=0,
        description="Total number of PRs created"
    )


class OrchestratorState(BaseModel):
    """
    The complete state object passed through the LangGraph workflow.
    
    This is the shared state that all agents read from and write to.
    LangGraph ensures each agent sees the cumulative updates from prior agents.
    
    Architectural note:
        This is the most important model in the agent layer.
        It defines the contract between all agents.
        Changing a field here affects every agent in the system.
    """
    # Input — set at workflow start
    scan_id: UUID = Field(
        description="Unique ID for this scan workflow"
    )
    scan_request: ScanRequest = Field(
        description="Original scan parameters from API"
    )
    
    # Scanner agent output
    scanner_output: Optional[ScannerOutput] = None
    
    # CVE Analyst agent output
    cve_analyst_output: Optional[CVEAnalystOutput] = None
    
    # Impact Assessor agent output
    impact_assessor_output: Optional[ImpactAssessorOutput] = None
    
    # Remediation Planner agent output
    remediation_planner_output: Optional[RemediationPlannerOutput] = None
    
    # Human approval decisions
    approved_plan_ids: list[UUID] = Field(
        default_factory=list,
        description="Plans that received human approval"
    )
    rejected_plan_ids: list[UUID] = Field(
        default_factory=list,
        description="Plans that were rejected by human"
    )
    
    # PR Creator agent output
    pr_creator_output: Optional[PRCreatorOutput] = None
    
    # Workflow metadata
    current_agent: str = Field(
        default="scanner",
        description="Which agent is currently executing"
    )
    workflow_status: str = Field(
        default="running",
        description="Overall workflow state: 'running', 'waiting_approval', 'completed', 'failed'"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if workflow failed"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    completed_at: Optional[datetime] = None

    def get_all_findings(self) -> list[Finding]:
        """
        Helper to get all findings from CVE Analyst output.
        
        Used by downstream agents that need to iterate over findings.
        """
        if self.cve_analyst_output:
            return self.cve_analyst_output.findings
        return []

    def get_approved_plans(self) -> list[RemediationPlan]:
        """
        Helper to get only approved remediation plans.
        
        Used by PR Creator to know which plans to execute.
        """
        if not self.remediation_planner_output:
            return []
        return [
            plan for plan in self.remediation_planner_output.remediation_plans
            if plan.id in self.approved_plan_ids
        ]