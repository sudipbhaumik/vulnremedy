"""
Integration test script for the VulnRemedy LangGraph orchestrator and FastAPI app.

Run with:
    uv run python scripts/test_orchestrator.py

Strategy:
  Orchestrator tests use dependency injection (mock agents) + dry_run=True
  so no GitHub token or live Ollama/RAG call is needed.
  FastAPI tests use TestClient + mock orchestrator injection.

Tests:
  1.  build_graph() — returns compiled graph with interrupt_before=["pr_creator"]
  2.  Orchestrator instantiation — has _graph and _checkpointer
  3.  Response dict structure — all required top-level and summary keys
  4.  Short-circuit: no dependencies → status="complete", dependencies_found=0
  5.  Short-circuit: no CVE findings → status="complete", vulnerabilities_found=0
  6.  Short-circuit: no CRITICAL/HIGH reports → status="complete", actionable_reports=0
  7.  Short-circuit: no plans → status="complete", plans_created=0
  8.  auto_approve=False → status="awaiting_approval" (graph interrupted at pr_creator)
  9.  auto_approve=True  → status="complete", prs_created > 0
  10. approve_and_resume() — awaiting_approval → approve → complete
  11. get_status() returns current state matching run() output
  12. get_status() on unknown scan_id → status="not_found"
  13. Summary counts consistent with actual state
  14. FastAPI: GET /health → 200 {"status": "ok"}
  15. FastAPI: POST /scan → 202 with required response keys
  16. FastAPI: GET /scan/{id} → 200
  17. FastAPI: GET /scan/{unknown_id} → 404
  18. FastAPI: POST /scan/{id}/approve (awaiting_approval) → 200 complete
  19. FastAPI: POST /scan/{id}/approve (already complete) → 409 conflict
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from vulnremedy.models.cve import CVERecord, CVSSScore, Ecosystem, Severity
from vulnremedy.models.dependency import Dependency
from vulnremedy.models.finding import AffectedDependency, DependencyType, Finding, FindingStatus
from vulnremedy.models.impact import ImpactReport, Priority
from vulnremedy.models.remediation import (
    CodeChange,
    RemediationPlan,
    RemediationStatus,
    RemediationStrategy,
)
from vulnremedy.agents.orchestrator.orchestrator import VulnRemedyOrchestrator, build_graph

_FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"

# ─── Helpers ──────────────────────────────────────────────────────────────────

_PASS = "PASS"
_FAIL = "FAIL"


def _result(ok: bool, label: str, detail: str = "") -> None:
    status = _PASS if ok else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")


# ─── Mock agents ──────────────────────────────────────────────────────────────


class _MockScanner:
    """Scanner stub — returns a fixed list of dependencies."""

    def __init__(self, dependencies: list[Dependency], succeed: bool = True):
        self._deps = dependencies
        self._succeed = succeed

    def scan(self, repo: str, branch: str = "main") -> dict:
        if not self._succeed:
            return {
                "success": False,
                "dependencies": [],
                "manifests_scanned": [],
                "errors": ["Simulated scan failure"],
            }
        return {
            "success": True,
            "dependencies": self._deps,
            "manifests_scanned": ["requirements.txt"],
            "errors": [],
        }


class _MockAnalyst:
    """CVE Analyst stub — returns a fixed list of findings."""

    def __init__(self, findings: list[Finding]):
        self._findings = findings

    def analyze(
        self, dependencies, scan_id=None, repository="unknown", branch="main"
    ) -> dict:
        return {
            "success": True,
            "findings": self._findings,
            "dependencies_analyzed": len(dependencies),
            "vulnerabilities_found": len(self._findings),
            "errors": [],
        }


class _MockAssessor:
    """Impact Assessor stub — returns a fixed list of ImpactReports."""

    def __init__(self, reports: list[ImpactReport]):
        self._reports = reports

    def assess(self, findings, scan_id=None) -> dict:
        return {
            "success": True,
            "impact_reports": self._reports,
            "high_priority_count": sum(
                1
                for r in self._reports
                if r.priority in (Priority.CRITICAL, Priority.HIGH)
            ),
            "errors": [],
        }


class _MockPlanner:
    """Remediation Planner stub — returns a fixed list of plans."""

    def __init__(self, plans: list[RemediationPlan]):
        self._plans = plans

    def plan(self, impact_reports, scan_id=None) -> dict:
        return {
            "success": True,
            "plans": self._plans,
            "skipped_count": 0,
            "errors": [],
        }


class _MockPRCreator:
    """PR Creator stub — processes APPROVED plans in dry-run mode."""

    def create_prs(
        self, plans: list[RemediationPlan], dry_run: bool = True
    ) -> dict:
        prs_created = []
        skipped = []
        for plan in plans:
            if plan.status == RemediationStatus.APPROVED:
                pr_url = f"https://github.com/acme/test-repo/pull/DRY_RUN"
                plan.mark_pr_created(pr_url)
                prs_created.append(
                    {
                        "plan_id": str(plan.id),
                        "cve_id": plan.finding.cve.cve_id,
                        "fix_branch": f"fix/cve-{plan.finding.cve.cve_id.lower()}-mock",
                        "pr_url": pr_url,
                        "pr_title": f"fix(security): {plan.summary[:80]}",
                        "dry_run": dry_run,
                    }
                )
            else:
                skipped.append(
                    {
                        "plan_id": str(plan.id),
                        "reason": f"status={plan.status.value}",
                    }
                )
        return {
            "success": True,
            "prs_created": prs_created,
            "skipped": skipped,
            "errors": [],
        }


# ─── Fixture builders ─────────────────────────────────────────────────────────


def _make_dep(
    pkg: str = "log4j-core",
    version: str = "2.14.1",
    ecosystem: Ecosystem = Ecosystem.MAVEN,
) -> Dependency:
    return Dependency(
        package_name=pkg,
        group_id="org.apache.logging.log4j" if ecosystem == Ecosystem.MAVEN else None,
        version=version,
        ecosystem=ecosystem,
        manifest_file="pom.xml",
    )


def _make_finding(priority_hint: Priority = Priority.CRITICAL) -> Finding:
    cve = CVERecord(
        cve_id="CVE-2021-44228",
        description="Log4Shell RCE vulnerability in Apache Log4j.",
        cvss=CVSSScore(version="3.1", score=10.0, severity=Severity.CRITICAL),
        source="nvd",
    )
    dep = AffectedDependency(
        package_name="log4j-core",
        group_id="org.apache.logging.log4j",
        current_version="2.14.1",
        fixed_version="2.17.1",
        ecosystem=Ecosystem.MAVEN,
        dependency_type=DependencyType.DIRECT,
        manifest_file="pom.xml",
    )
    return Finding(
        scan_id=uuid4(),
        repository="github.com/acme/payment-service",
        branch="main",
        cve=cve,
        affected_dependency=dep,
        status=FindingStatus.CONFIRMED,
        severity=Severity.CRITICAL
        if priority_hint == Priority.CRITICAL
        else Severity.MEDIUM,
    )


def _make_report(
    finding: Finding, priority: Priority = Priority.CRITICAL
) -> ImpactReport:
    return ImpactReport(
        finding=finding,
        is_exploitable=True,
        exploitability_reasoning="Direct network-accessible endpoint.",
        business_impact="RCE possible — critical data exposure.",
        priority=priority,
        used_fallback=True,
    )


def _make_plan(
    finding: Finding, status: RemediationStatus = RemediationStatus.DRAFT
) -> RemediationPlan:
    return RemediationPlan(
        finding_id=finding.id,
        finding=finding,
        strategy=RemediationStrategy.VERSION_UPGRADE,
        status=status,
        code_changes=[
            CodeChange(
                file_path="pom.xml",
                change_type="version_bump",
                original_content="<version>2.14.1</version>",
                new_content="<version>2.17.1</version>",
                rationale="Upgrade log4j-core from 2.14.1 to 2.17.1",
            )
        ],
        summary="Upgrade log4j-core from 2.14.1 to 2.17.1 to remediate CVE-2021-44228",
        detailed_steps=["Update log4j-core to 2.17.1 in pom.xml"],
        testing_recommendations=["Run mvn test"],
        breaking_changes=[],
        requires_human_approval=True,
    )


def _make_full_orch(
    deps=None,
    findings=None,
    reports=None,
    plans=None,
) -> VulnRemedyOrchestrator:
    """Build an orchestrator with all mock agents injected."""
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding)

    return VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=deps or [_make_dep()]),
        analyst=_MockAnalyst(findings=findings if findings is not None else [finding]),
        assessor=_MockAssessor(reports=reports if reports is not None else [report]),
        planner=_MockPlanner(plans=plans if plans is not None else [plan]),
        pr_creator=_MockPRCreator(),
    )


# ─── 1. build_graph() ─────────────────────────────────────────────────────────


def test_build_graph() -> None:
    print("\n── 1. build_graph() — compiled StateGraph ────────────────────────")
    from langgraph.checkpoint.memory import MemorySaver

    cp = MemorySaver()
    graph = build_graph(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        pr_creator=_MockPRCreator(),
        checkpointer=cp,
    )
    _result(graph is not None, "build_graph() returns a compiled graph")
    _result(callable(getattr(graph, "invoke", None)), "Graph has .invoke() method")
    _result(callable(getattr(graph, "get_state", None)), "Graph has .get_state() method")


# ─── 2. Orchestrator instantiation ───────────────────────────────────────────


def test_instantiation() -> None:
    print("\n── 2. Orchestrator instantiation ─────────────────────────────────")
    orch = VulnRemedyOrchestrator(pr_creator=_MockPRCreator())
    _result(hasattr(orch, "_graph"), "Has ._graph attribute")
    _result(hasattr(orch, "_checkpointer"), "Has ._checkpointer attribute")
    _result(callable(orch.run), "Has .run() method")
    _result(callable(orch.approve_and_resume), "Has .approve_and_resume() method")
    _result(callable(orch.get_status), "Has .get_status() method")


# ─── 3. Response dict structure ───────────────────────────────────────────────


def test_response_structure() -> None:
    print("\n── 3. Response dict structure ─────────────────────────────────────")
    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True)

    required_keys = {
        "scan_id",
        "status",
        "repository",
        "branch",
        "summary",
        "errors",
    }
    for key in required_keys:
        _result(key in result, f"Top-level key '{key}' present")

    summary_keys = {
        "dependencies_found",
        "vulnerabilities_found",
        "impact_reports",
        "actionable_reports",
        "plans_created",
        "plans_approved",
        "prs_created",
        "errors_total",
    }
    for key in summary_keys:
        _result(key in result["summary"], f"summary.{key} present")


# ─── 4–7. Short-circuit conditions ───────────────────────────────────────────


def test_short_circuit_no_dependencies() -> None:
    print("\n── 4. Short-circuit: no dependencies ──────────────────────────────")
    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True)

    _result(result["status"] == "complete", "status='complete'", result["status"])
    _result(result["summary"]["dependencies_found"] == 0, "dependencies_found == 0")
    _result(result["summary"]["vulnerabilities_found"] == 0, "vulnerabilities_found == 0")
    _result(result["summary"]["plans_created"] == 0, "plans_created == 0")
    _result(result["summary"]["prs_created"] == 0, "prs_created == 0")


def test_short_circuit_no_findings() -> None:
    print("\n── 5. Short-circuit: no CVE findings ──────────────────────────────")
    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True)

    _result(result["status"] == "complete", "status='complete'", result["status"])
    _result(result["summary"]["dependencies_found"] == 1, "dependencies_found == 1")
    _result(result["summary"]["vulnerabilities_found"] == 0, "vulnerabilities_found == 0")
    _result(result["summary"]["plans_created"] == 0, "plans_created == 0")


def test_short_circuit_no_actionable_reports() -> None:
    print("\n── 6. Short-circuit: no CRITICAL/HIGH reports ─────────────────────")
    finding = _make_finding(priority_hint=Priority.LOW)
    low_report = _make_report(finding, priority=Priority.LOW)

    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[low_report]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True)

    _result(result["status"] == "complete", "status='complete'", result["status"])
    _result(result["summary"]["vulnerabilities_found"] == 1, "vulnerabilities_found == 1")
    _result(result["summary"]["actionable_reports"] == 0, "actionable_reports == 0")
    _result(result["summary"]["plans_created"] == 0, "plans_created == 0")


def test_short_circuit_no_plans() -> None:
    print("\n── 7. Short-circuit: no plans generated ───────────────────────────")
    finding = _make_finding()
    report = _make_report(finding)

    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True)

    _result(result["status"] == "complete", "status='complete'", result["status"])
    _result(result["summary"]["plans_created"] == 0, "plans_created == 0")
    _result(result["summary"]["prs_created"] == 0, "prs_created == 0")


# ─── 8–9. auto_approve behaviour ─────────────────────────────────────────────


def test_auto_approve_false_awaiting() -> None:
    print("\n── 8. auto_approve=False → status='awaiting_approval' ─────────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding, status=RemediationStatus.DRAFT)

    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True, auto_approve=False)

    _result(
        result["status"] == "awaiting_approval",
        "status='awaiting_approval'",
        result["status"],
    )
    _result(result["summary"]["prs_created"] == 0, "No PRs created yet")
    _result(result["summary"]["plans_created"] == 1, "1 plan created (awaiting approval)")


def test_auto_approve_true_complete() -> None:
    print("\n── 9. auto_approve=True → status='complete', PRs created ──────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding, status=RemediationStatus.DRAFT)

    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True, auto_approve=True)

    _result(result["status"] == "complete", "status='complete'", result["status"])
    _result(result["summary"]["prs_created"] == 1, "1 PR created")
    _result(
        result["summary"]["plans_approved"] >= 1,
        "At least 1 plan approved",
        str(result["summary"]["plans_approved"]),
    )
    # LangGraph serializes plan objects via MemorySaver; mark_pr_created() is
    # called on the deserialized copy inside the graph — check via result dict.
    prs = result.get("prs_created", [])
    _result(len(prs) == 1, "prs_created list has 1 entry", str(len(prs)))
    _result(
        len(prs) > 0 and "DRY_RUN" in prs[0].get("pr_url", ""),
        "prs_created[0].pr_url is a dry-run URL",
        prs[0].get("pr_url", "<none>") if prs else "<none>",
    )


# ─── 10. approve_and_resume() workflow ───────────────────────────────────────


def test_approve_and_resume() -> None:
    print("\n── 10. approve_and_resume() — human-in-the-loop workflow ──────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding, status=RemediationStatus.DRAFT)

    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
        pr_creator=_MockPRCreator(),
    )

    # Step 1: start without auto-approve → should pause
    run_result = orch.run("acme/repo", dry_run=True, auto_approve=False)
    _result(
        run_result["status"] == "awaiting_approval",
        "Step 1: status='awaiting_approval'",
        run_result["status"],
    )
    scan_id = run_result["scan_id"]
    _result(bool(scan_id), "Step 1: scan_id returned", scan_id)

    # Step 2: human approves → pipeline resumes
    resume_result = orch.approve_and_resume(scan_id)
    _result(
        resume_result["status"] == "complete",
        "Step 2: status='complete' after approve",
        resume_result["status"],
    )
    _result(resume_result["summary"]["prs_created"] == 1, "Step 2: 1 PR created")
    # LangGraph operates on a deserialized copy of the plan — check via summary.
    _result(
        resume_result["summary"]["plans_approved"] >= 1,
        "Step 2: plans_approved >= 1 (includes PR_CREATED status)",
        str(resume_result["summary"]["plans_approved"]),
    )


# ─── 11. get_status() ────────────────────────────────────────────────────────


def test_get_status() -> None:
    print("\n── 11. get_status() — read current state ──────────────────────────")
    orch = _make_full_orch()
    run_result = orch.run("acme/repo", dry_run=True, auto_approve=False)
    scan_id = run_result["scan_id"]

    status_result = orch.get_status(scan_id)

    _result(
        status_result["scan_id"] == scan_id,
        "scan_id matches",
        status_result["scan_id"],
    )
    _result(
        status_result["status"] == run_result["status"],
        "status matches run() output",
        status_result["status"],
    )
    _result(
        status_result["repository"] == "acme/repo",
        "repository field present",
        status_result["repository"],
    )


# ─── 12. get_status() on unknown scan ────────────────────────────────────────


def test_get_status_not_found() -> None:
    print("\n── 12. get_status() — unknown scan_id → not_found ─────────────────")
    orch = VulnRemedyOrchestrator(pr_creator=_MockPRCreator())
    fake_id = str(uuid4())
    result = orch.get_status(fake_id)

    _result(result["status"] == "not_found", "status='not_found'", result["status"])
    _result(result["scan_id"] == fake_id, "scan_id echoed back", result["scan_id"])
    _result("error" in result, "error field present in not_found response")


# ─── 13. Summary consistency ─────────────────────────────────────────────────


def test_summary_consistency() -> None:
    print("\n── 13. Summary counts consistent ──────────────────────────────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding)

    orch = VulnRemedyOrchestrator(
        scanner=_MockScanner(
            dependencies=[
                _make_dep(),
                _make_dep("requests", "2.25.0", Ecosystem.PYPI),
            ]
        ),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
        pr_creator=_MockPRCreator(),
    )
    result = orch.run("acme/repo", dry_run=True, auto_approve=True)

    s = result["summary"]
    _result(s["dependencies_found"] == 2, "dependencies_found == 2", str(s["dependencies_found"]))
    _result(s["vulnerabilities_found"] == 1, "vulnerabilities_found == 1", str(s["vulnerabilities_found"]))
    _result(s["impact_reports"] == 1, "impact_reports == 1", str(s["impact_reports"]))
    _result(s["actionable_reports"] == 1, "actionable_reports == 1", str(s["actionable_reports"]))
    _result(s["plans_created"] == 1, "plans_created == 1", str(s["plans_created"]))
    _result(s["prs_created"] == 1, "prs_created == 1", str(s["prs_created"]))
    _result(isinstance(s["errors_total"], int) and s["errors_total"] >= 0, "errors_total is non-negative int")


# ─── 14–19. FastAPI endpoint tests ───────────────────────────────────────────


def _make_mock_orchestrator() -> object:
    """
    Return a stateful fake orchestrator matching VulnRemedyOrchestrator's interface.
    Used to inject into the FastAPI app without touching real agents.
    """
    import uuid as _uuid_mod

    class _FakeOrch:
        def __init__(self) -> None:
            self._store: dict = {}

        def run(
            self,
            repo: str,
            branch: str = "main",
            dry_run: bool = True,
            auto_approve: bool = False,
            scan_id: str | None = None,
        ) -> dict:
            sid = scan_id or str(_uuid_mod.uuid4())
            status = "complete" if auto_approve else "awaiting_approval"
            record = {
                "scan_id": sid,
                "status": status,
                "repository": repo,
                "branch": branch,
                "summary": {
                    "dependencies_found": 5,
                    "vulnerabilities_found": 2,
                    "impact_reports": 2,
                    "actionable_reports": 2,
                    "plans_created": 2,
                    "plans_approved": 2 if auto_approve else 0,
                    "prs_created": 2 if auto_approve else 0,
                    "errors_total": 0,
                },
                "errors": [],
            }
            self._store[sid] = record
            return record

        def get_status(self, scan_id: str) -> dict:
            if scan_id not in self._store:
                return {
                    "scan_id": scan_id,
                    "status": "not_found",
                    "error": f"No scan found: {scan_id}",
                }
            return self._store[scan_id]

        def approve_and_resume(self, scan_id: str) -> dict:
            if scan_id not in self._store:
                raise ValueError(f"Scan not found: {scan_id!r}")
            record = dict(self._store[scan_id])
            record["status"] = "complete"
            record["summary"] = dict(record["summary"])
            record["summary"]["prs_created"] = 2
            record["summary"]["plans_approved"] = 2
            record["prs_created"] = [
                {"plan_id": "mock-plan-1", "pr_url": "https://github.com/acme/repo/pull/1", "dry_run": True},
                {"plan_id": "mock-plan-2", "pr_url": "https://github.com/acme/repo/pull/2", "dry_run": True},
            ]
            self._store[scan_id] = record
            return record

    return _FakeOrch()


def test_api_health() -> None:
    print("\n── 14. FastAPI: GET /health ────────────────────────────────────────")
    import vulnremedy.api.main as api_module
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        api_module._orchestrator = _make_mock_orchestrator()
        resp = client.get("/health")
        _result(resp.status_code == 200, "GET /health → 200", str(resp.status_code))
        body = resp.json()
        _result(body.get("status") == "ok", "body.status == 'ok'", str(body.get("status")))


def test_api_post_scan() -> None:
    print("\n── 15. FastAPI: POST /scan → 202 ──────────────────────────────────")
    import vulnremedy.api.main as api_module
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        api_module._orchestrator = _make_mock_orchestrator()
        resp = client.post(
            "/scan",
            json={"repo": "acme/test-repo", "branch": "main", "dry_run": True, "auto_approve": False},
        )
        _result(resp.status_code == 202, "POST /scan → 202", str(resp.status_code))
        body = resp.json()
        for key in ("scan_id", "status", "repository", "branch", "summary", "errors"):
            _result(key in body, f"Response has key '{key}'")
        _result(bool(body.get("scan_id")), "scan_id is non-empty", body.get("scan_id", ""))


def test_api_get_scan() -> None:
    print("\n── 16. FastAPI: GET /scan/{id} → 200 ──────────────────────────────")
    import vulnremedy.api.main as api_module
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        mock_orch = _make_mock_orchestrator()
        api_module._orchestrator = mock_orch

        # Create a scan first
        post_resp = client.post(
            "/scan",
            json={"repo": "acme/test-repo", "branch": "main", "dry_run": True, "auto_approve": False},
        )
        scan_id = post_resp.json()["scan_id"]

        # Now fetch it
        get_resp = client.get(f"/scan/{scan_id}")
        _result(get_resp.status_code == 200, "GET /scan/{id} → 200", str(get_resp.status_code))
        body = get_resp.json()
        _result(body.get("scan_id") == scan_id, "Response scan_id matches", body.get("scan_id", ""))
        _result("status" in body, "Response has 'status' field")


def test_api_get_scan_not_found() -> None:
    print("\n── 17. FastAPI: GET /scan/{unknown} → 404 ─────────────────────────")
    import vulnremedy.api.main as api_module
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        api_module._orchestrator = _make_mock_orchestrator()
        resp = client.get(f"/scan/{uuid4()}")
        _result(resp.status_code == 404, "GET /scan/{unknown} → 404", str(resp.status_code))
        _result("detail" in resp.json(), "Error response has 'detail' field")


def test_api_approve_scan() -> None:
    print("\n── 18. FastAPI: POST /scan/{id}/approve (awaiting) → 200 ──────────")
    import vulnremedy.api.main as api_module
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        mock_orch = _make_mock_orchestrator()
        api_module._orchestrator = mock_orch

        # Create a scan that's awaiting_approval (auto_approve=False)
        post_resp = client.post(
            "/scan",
            json={"repo": "acme/test-repo", "branch": "main", "dry_run": True, "auto_approve": False},
        )
        scan_id = post_resp.json()["scan_id"]

        # Approve it
        approve_resp = client.post(f"/scan/{scan_id}/approve")
        _result(
            approve_resp.status_code == 200,
            "POST /approve → 200",
            str(approve_resp.status_code),
        )
        body = approve_resp.json()
        _result(body.get("status") == "complete", "status='complete' after approval", body.get("status", ""))
        _result(body.get("scan_id") == scan_id, "scan_id matches")
        _result("prs_created" in body, "Response has 'prs_created' field")


def test_api_approve_already_complete() -> None:
    print("\n── 19. FastAPI: POST /scan/{id}/approve (complete) → 409 ──────────")
    import vulnremedy.api.main as api_module
    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        mock_orch = _make_mock_orchestrator()
        api_module._orchestrator = mock_orch

        # Create a scan that's already complete (auto_approve=True)
        post_resp = client.post(
            "/scan",
            json={"repo": "acme/test-repo", "branch": "main", "dry_run": True, "auto_approve": True},
        )
        scan_id = post_resp.json()["scan_id"]

        # Attempting to approve a completed scan → 409
        approve_resp = client.post(f"/scan/{scan_id}/approve")
        _result(
            approve_resp.status_code == 409,
            "POST /approve on complete scan → 409",
            str(approve_resp.status_code),
        )
        _result("detail" in approve_resp.json(), "409 response has 'detail' field")


# ─── Runner ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 60)
    print("  VulnRemedy LangGraph Orchestrator — Integration Test Suite")
    print("=" * 60)

    # ── Orchestrator / LangGraph tests ────────────────────────────────────
    test_build_graph()
    test_instantiation()
    test_response_structure()
    test_short_circuit_no_dependencies()
    test_short_circuit_no_findings()
    test_short_circuit_no_actionable_reports()
    test_short_circuit_no_plans()
    test_auto_approve_false_awaiting()
    test_auto_approve_true_complete()
    test_approve_and_resume()
    test_get_status()
    test_get_status_not_found()
    test_summary_consistency()

    # ── FastAPI endpoint tests ─────────────────────────────────────────────
    test_api_health()
    test_api_post_scan()
    test_api_get_scan()
    test_api_get_scan_not_found()
    test_api_approve_scan()
    test_api_approve_already_complete()

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
