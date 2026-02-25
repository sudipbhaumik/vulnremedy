"""
VulnRemedy FastAPI Application.

Exposes the LangGraph orchestrator as an HTTP API.

Endpoints:
  POST  /scan                   — Start a new vulnerability scan
  GET   /scan/{scan_id}         — Poll scan status
  POST  /scan/{scan_id}/approve — Approve plans and resume PR creation

Human-approval workflow:
  1. POST /scan  {repo, branch, dry_run=true, auto_approve=false}
     → 202 Accepted  {scan_id, status="awaiting_approval", ...}

  2. GET /scan/{scan_id}
     → review remediation_plans in the response

  3. POST /scan/{scan_id}/approve
     → all DRAFT plans are approved; pr_creator resumes
     → 200 OK  {status="complete", prs_created=[...]}

Auto-approve workflow (CI/CD, testing):
  POST /scan  {repo, branch, dry_run=true, auto_approve=true}
  → 200 OK  {status="complete", prs_created=[...]}

Run locally:
  uv run uvicorn vulnremedy.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from vulnremedy.agents.orchestrator.orchestrator import VulnRemedyOrchestrator
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


# ─── Application lifecycle ────────────────────────────────────────────────────


_orchestrator: Optional[VulnRemedyOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared orchestrator at startup; clean up at shutdown."""
    global _orchestrator
    logger.info("VulnRemedy API starting — initialising orchestrator")
    _orchestrator = VulnRemedyOrchestrator()
    logger.info("Orchestrator ready")
    yield
    _orchestrator = None
    logger.info("VulnRemedy API shutting down")


app = FastAPI(
    title="VulnRemedy API",
    description=(
        "AI-powered vulnerability scanning and automated PR remediation. "
        "Uses LangGraph for stateful multi-agent orchestration."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _get_orchestrator() -> VulnRemedyOrchestrator:
    if _orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized",
        )
    return _orchestrator


# ─── Request / Response models ────────────────────────────────────────────────


class ScanRequest(BaseModel):
    """Request body for POST /scan."""

    repo: str = Field(
        description="Repository in 'owner/repo' format or full GitHub URL.",
        examples=["apache/log4j"],
    )
    branch: str = Field(
        default="main",
        description="Branch to scan.",
    )
    dry_run: bool = Field(
        default=True,
        description=(
            "If true, PR Creator simulates GitHub API calls without opening real PRs. "
            "Safe for testing without a GitHub token."
        ),
    )
    auto_approve: bool = Field(
        default=False,
        description=(
            "If true, all DRAFT remediation plans are automatically approved "
            "and PR creation runs immediately. "
            "If false, the pipeline pauses for human review (status='awaiting_approval')."
        ),
    )


class ScanSummary(BaseModel):
    """Headline counts from a scan run."""

    dependencies_found: int
    vulnerabilities_found: int
    impact_reports: int
    actionable_reports: int
    plans_created: int
    plans_approved: int
    prs_created: int
    errors_total: int


class ScanResponse(BaseModel):
    """
    Response body for POST /scan, GET /scan/{scan_id},
    and POST /scan/{scan_id}/approve.
    """

    scan_id: str
    status: str = Field(
        description=(
            "'complete'           — pipeline finished (PRs created or short-circuited). "
            "'awaiting_approval'  — paused before PR creation; call /approve to resume. "
            "'not_found'          — no scan with this ID."
        )
    )
    repository: str
    branch: str
    summary: ScanSummary
    errors: list[str]


class ApproveResponse(BaseModel):
    """Response body for POST /scan/{scan_id}/approve."""

    scan_id: str
    status: str
    summary: ScanSummary
    prs_created: list[dict[str, Any]]
    errors: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post(
    "/scan",
    response_model=ScanResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a vulnerability scan",
    description=(
        "Scans the given repository for vulnerable dependencies, assesses impact, "
        "and generates (or auto-creates) remediation PRs. "
        "The pipeline runs synchronously inside a thread pool so the event loop "
        "is not blocked."
    ),
)
async def start_scan(request: ScanRequest) -> ScanResponse:
    """POST /scan — launch the full LangGraph pipeline."""
    orch = _get_orchestrator()

    logger.info(
        "API: POST /scan",
        repo=request.repo,
        branch=request.branch,
        dry_run=request.dry_run,
        auto_approve=request.auto_approve,
    )

    result: dict[str, Any] = await run_in_threadpool(
        orch.run,
        request.repo,
        request.branch,
        request.dry_run,
        request.auto_approve,
    )

    return ScanResponse(
        scan_id=result["scan_id"],
        status=result["status"],
        repository=result["repository"],
        branch=result["branch"],
        summary=ScanSummary(**result["summary"]),
        errors=result["errors"],
    )


@app.get(
    "/scan/{scan_id}",
    response_model=ScanResponse,
    summary="Get scan status",
    description=(
        "Returns the current state of a scan. "
        "Poll this after POST /scan to check progress or retrieve the "
        "remediation plan list for human review."
    ),
)
async def get_scan(scan_id: str) -> ScanResponse:
    """GET /scan/{scan_id} — read current graph state from checkpointer."""
    orch = _get_orchestrator()

    result = await run_in_threadpool(orch.get_status, scan_id)

    if result.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("error", f"Scan {scan_id!r} not found"),
        )

    return ScanResponse(
        scan_id=result["scan_id"],
        status=result["status"],
        repository=result["repository"],
        branch=result["branch"],
        summary=ScanSummary(**result["summary"]),
        errors=result["errors"],
    )


@app.post(
    "/scan/{scan_id}/approve",
    response_model=ApproveResponse,
    summary="Approve plans and resume PR creation",
    description=(
        "Approves all DRAFT remediation plans for this scan and resumes the "
        "LangGraph pipeline from the pr_creator node. "
        "Only valid when scan status is 'awaiting_approval'."
    ),
)
async def approve_scan(scan_id: str) -> ApproveResponse:
    """POST /scan/{scan_id}/approve — resume graph after human review."""
    orch = _get_orchestrator()

    # Verify the scan exists and is in the right state
    current = await run_in_threadpool(orch.get_status, scan_id)
    if current.get("status") == "not_found":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan {scan_id!r} not found",
        )
    if current.get("status") == "complete":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scan {scan_id!r} is already complete — PRs have been processed.",
        )

    logger.info("API: POST /scan/{scan_id}/approve", scan_id=scan_id)

    result: dict[str, Any] = await run_in_threadpool(
        orch.approve_and_resume, scan_id
    )

    return ApproveResponse(
        scan_id=result["scan_id"],
        status=result["status"],
        summary=ScanSummary(**result["summary"]),
        prs_created=result.get("prs_created", []),
        errors=result["errors"],
    )


# ─── Health check ─────────────────────────────────────────────────────────────


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
