"""
VulnRemedy Pipeline — Sequential multi-agent orchestration.

Runs the five agents in order, threading each stage's output into the next.
Short-circuits cleanly when there is nothing for downstream agents to process.

Stage order:
  1. Scanner        → list[Dependency]
  2. CVE Analyst    → list[Finding]
  3. Impact Assessor → list[ImpactReport]  (CRITICAL / HIGH only routed forward)
  4. Remediation Planner → list[RemediationPlan]
  5. PR Creator     → list[dict]  (prs_created)

Short-circuit conditions (pipeline stops early, no error):
  - Scanner returns 0 dependencies  → stage "no_dependencies"
  - CVE Analyst returns 0 findings  → stage "no_findings"
  - Impact Assessor returns 0 reports with CRITICAL/HIGH priority → stage "no_actionable_reports"
  - Remediation Planner returns 0 plans → stage "no_plans"

Human-approval model:
  All plans default to DRAFT status.  The Remediation Planner sets
  requires_human_approval=True for CRITICAL findings and complex strategies.

  auto_approve=True (default False) calls plan.approve() on every DRAFT plan
  before PR creation — useful for dry-run integration tests where no human
  is in the loop.

  In production: plans requiring approval are surfaced via the API; a human
  calls plan.approve(), then the PR Creator is called separately.

Output: dict with top-level keys:
  - success:       bool
  - scan_id:       str
  - repository:    str
  - branch:        str
  - short_circuit: str | None  (populated when pipeline stopped early)
  - summary:       dict        (headline counts)
  - stages:        dict        (raw output of each stage that ran)
  - errors:        list[str]   (accumulated across all stages)
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID, uuid4

from vulnremedy.agents.cve_analyst.agent import CVEAnalystAgent
from vulnremedy.agents.impact_assessor.agent import ImpactAssessorAgent
from vulnremedy.agents.pr_creator.agent import PRCreatorAgent
from vulnremedy.agents.remediation.agent import RemediationPlannerAgent
from vulnremedy.agents.scanner.agent import ScannerAgent
from vulnremedy.models.impact import Priority
from vulnremedy.models.remediation import RemediationStatus
from vulnremedy.utils.logging import logger


class VulnRemedyPipeline:
    """
    Sequential five-stage vulnerability remediation pipeline.

    Accepts optional agent instances for dependency injection (testing/mocking).
    When not supplied, each agent is instantiated with default settings.

    Usage:
        pipeline = VulnRemedyPipeline()
        result = pipeline.run(
            repo="apache/log4j",
            branch="main",
            dry_run=True,
            auto_approve=True,
        )
        print(result["summary"])
    """

    def __init__(
        self,
        scanner: Optional[Any] = None,
        analyst: Optional[Any] = None,
        assessor: Optional[Any] = None,
        planner: Optional[Any] = None,
        pr_creator: Optional[Any] = None,
    ) -> None:
        """
        Initialize the pipeline.

        All agent parameters are optional — pass custom instances to mock
        individual stages during testing.
        """
        self.scanner = scanner or ScannerAgent()
        self.analyst = analyst or CVEAnalystAgent()
        self.assessor = assessor or ImpactAssessorAgent()
        self.planner = planner or RemediationPlannerAgent()
        self.pr_creator = pr_creator or PRCreatorAgent()

        logger.info("VulnRemedy Pipeline initialized")

    # ─── Public entry point ───────────────────────────────────────────────────

    def run(
        self,
        repo: str,
        branch: str = "main",
        dry_run: bool = True,
        auto_approve: bool = False,
        scan_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """
        Run the full vulnerability remediation pipeline.

        Args:
            repo:         Repository slug "owner/repo" or full URL.
            branch:       Branch to scan (default: "main").
            dry_run:      If True, PR Creator does not call GitHub API.
            auto_approve: If True, auto-approves all DRAFT plans so PR Creator
                          can process them (useful with dry_run=True for testing).
            scan_id:      Optional UUID to correlate this run in logs.

        Returns:
            dict with keys: success, scan_id, repository, branch,
            short_circuit, summary, stages, errors.
        """
        scan_id = scan_id or uuid4()
        errors: list[str] = []
        stages: dict[str, Any] = {}
        short_circuit: Optional[str] = None

        logger.info(
            "Pipeline run starting",
            repository=repo,
            branch=branch,
            scan_id=str(scan_id),
            dry_run=dry_run,
            auto_approve=auto_approve,
        )

        # ── Stage 1: Scan ─────────────────────────────────────────────────────
        scan = self.scanner.scan(repo, branch)
        stages["scan"] = scan
        errors.extend(scan.get("errors", []))

        if not scan.get("success") or not scan.get("dependencies"):
            short_circuit = "no_dependencies"
            logger.info(
                "Pipeline short-circuiting: no dependencies found",
                repo=repo,
                scan_errors=scan.get("errors", []),
            )
            return self._build_result(
                scan_id, repo, branch, short_circuit, stages, errors
            )

        dependencies = scan["dependencies"]
        logger.info("Stage 1 complete: Scanner", dependencies=len(dependencies))

        # ── Stage 2: CVE Analysis ─────────────────────────────────────────────
        cve = self.analyst.analyze(
            dependencies=dependencies,
            scan_id=scan_id,
            repository=repo,
            branch=branch,
        )
        stages["cve_analysis"] = cve
        errors.extend(cve.get("errors", []))

        if not cve.get("findings"):
            short_circuit = "no_findings"
            logger.info(
                "Pipeline short-circuiting: no CVE findings",
                dependencies_analyzed=cve.get("dependencies_analyzed", 0),
            )
            return self._build_result(
                scan_id, repo, branch, short_circuit, stages, errors
            )

        findings = cve["findings"]
        logger.info("Stage 2 complete: CVE Analyst", findings=len(findings))

        # ── Stage 3: Impact Assessment ────────────────────────────────────────
        impact = self.assessor.assess(findings=findings, scan_id=scan_id)
        stages["impact"] = impact
        errors.extend(impact.get("errors", []))

        actionable = [
            r for r in impact.get("impact_reports", [])
            if r.priority in (Priority.CRITICAL, Priority.HIGH)
        ]
        if not actionable:
            short_circuit = "no_actionable_reports"
            logger.info(
                "Pipeline short-circuiting: no CRITICAL/HIGH impact reports",
                total_reports=len(impact.get("impact_reports", [])),
            )
            return self._build_result(
                scan_id, repo, branch, short_circuit, stages, errors
            )

        logger.info(
            "Stage 3 complete: Impact Assessor",
            total_reports=len(impact["impact_reports"]),
            actionable=len(actionable),
        )

        # ── Stage 4: Remediation Planning ─────────────────────────────────────
        remediation = self.planner.plan(
            impact_reports=impact["impact_reports"],
            scan_id=scan_id,
        )
        stages["remediation"] = remediation
        errors.extend(remediation.get("errors", []))

        plans = remediation.get("plans", [])
        if not plans:
            short_circuit = "no_plans"
            logger.info("Pipeline short-circuiting: no remediation plans generated")
            return self._build_result(
                scan_id, repo, branch, short_circuit, stages, errors
            )

        logger.info("Stage 4 complete: Remediation Planner", plans=len(plans))

        # ── Auto-approval ─────────────────────────────────────────────────────
        if auto_approve:
            approved_count = 0
            for plan in plans:
                if plan.status == RemediationStatus.DRAFT:
                    plan.approve(approver="pipeline-auto-approve")
                    approved_count += 1
            if approved_count:
                logger.info(
                    "Auto-approved remediation plans",
                    approved=approved_count,
                    total=len(plans),
                )

        # ── Stage 5: PR Creation ──────────────────────────────────────────────
        prs = self.pr_creator.create_prs(plans=plans, dry_run=dry_run)
        stages["prs"] = prs
        errors.extend(prs.get("errors", []))

        logger.info(
            "Stage 5 complete: PR Creator",
            prs_created=len(prs.get("prs_created", [])),
            skipped=len(prs.get("skipped", [])),
            dry_run=dry_run,
        )

        return self._build_result(
            scan_id, repo, branch, short_circuit, stages, errors
        )

    # ─── Result assembly ─────────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        scan_id: UUID,
        repo: str,
        branch: str,
        short_circuit: Optional[str],
        stages: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any]:
        """Assemble the final output dict from accumulated stage data."""
        scan_stage = stages.get("scan", {})
        cve_stage = stages.get("cve_analysis", {})
        impact_stage = stages.get("impact", {})
        remediation_stage = stages.get("remediation", {})
        prs_stage = stages.get("prs", {})

        # Count actionable (CRITICAL/HIGH) impact reports
        actionable_reports = sum(
            1
            for r in impact_stage.get("impact_reports", [])
            if r.priority in (Priority.CRITICAL, Priority.HIGH)
        )

        summary = {
            "dependencies_found": len(scan_stage.get("dependencies", [])),
            "vulnerabilities_found": len(cve_stage.get("findings", [])),
            "impact_reports": len(impact_stage.get("impact_reports", [])),
            "actionable_reports": actionable_reports,
            "plans_created": len(remediation_stage.get("plans", [])),
            "prs_created": len(prs_stage.get("prs_created", [])),
            "errors_total": len(errors),
        }

        logger.info(
            "Pipeline complete",
            scan_id=str(scan_id),
            short_circuit=short_circuit or "none",
            **summary,
        )

        return {
            "success": True,
            "scan_id": str(scan_id),
            "repository": repo,
            "branch": branch,
            "short_circuit": short_circuit,
            "summary": summary,
            "stages": stages,
            "errors": errors,
        }
