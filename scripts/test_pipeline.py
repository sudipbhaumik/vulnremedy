"""
Integration test script for the VulnRemedy sequential pipeline.

Run with:
    uv run python scripts/test_pipeline.py

Strategy:
  All tests use dependency injection (mock scanner) + dry_run=True
  so no GitHub token or live Ollama response is required for most tests.
  Stages that do call Ollama / RAG (CVE Analyst, Impact Assessor, Planner)
  have deterministic fallbacks that activate automatically when Ollama is
  unavailable.

Tests:
  1.  Pipeline instantiation — all 5 agents present
  2.  Result dict structure — all required top-level keys present
  3.  Short-circuit: scanner returns no dependencies → stage "no_dependencies"
  4.  Short-circuit: no CVE findings → stage "no_findings"
  5.  Short-circuit: no CRITICAL/HIGH reports → stage "no_actionable_reports"
  6.  Short-circuit: no plans generated → stage "no_plans"
  7.  auto_approve=False → plans stay DRAFT, pr_creator skips them
  8.  auto_approve=True  → plans become APPROVED, pr_creator processes them
  9.  dry_run=True: prs_created entries are present, no real GitHub calls
  10. Stage errors are accumulated into result["errors"] (not raised)
  11. Multi-stage live run: fixture requirements.txt parsed → CVE Analyst → rest of pipeline
  12. Summary counts are consistent with stage outputs
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from vulnremedy.pipeline.pipeline import VulnRemedyPipeline
from vulnremedy.agents.scanner.parsers.pip_parser import PipParser
from vulnremedy.agents.scanner.parsers.npm_parser import NpmParser
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
            return {"success": False, "dependencies": [], "manifests_scanned": [], "errors": ["Simulated scan failure"]}
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

    def analyze(self, dependencies, scan_id=None, repository="unknown", branch="main") -> dict:
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
            "high_priority_count": sum(1 for r in self._reports if r.priority in (Priority.CRITICAL, Priority.HIGH)),
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


# ─── Fixture builders ─────────────────────────────────────────────────────────

def _make_dep(pkg="log4j-core", version="2.14.1", ecosystem=Ecosystem.MAVEN) -> Dependency:
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
        description="Log4Shell RCE vulnerability.",
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
        severity=Severity.CRITICAL if priority_hint == Priority.CRITICAL else Severity.MEDIUM,
    )


def _make_report(finding: Finding, priority: Priority = Priority.CRITICAL) -> ImpactReport:
    return ImpactReport(
        finding=finding,
        is_exploitable=True,
        exploitability_reasoning="Direct network-accessible endpoint.",
        business_impact="RCE possible.",
        priority=priority,
        used_fallback=True,
    )


def _make_plan(finding: Finding, status=RemediationStatus.DRAFT) -> RemediationPlan:
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


# ─── 1. Pipeline instantiation ───────────────────────────────────────────────


def test_instantiation() -> None:
    print("\n── 1. Pipeline instantiation ──────────────────────────────────────")
    pipeline = VulnRemedyPipeline()
    _result(hasattr(pipeline, "scanner"), "Has scanner agent")
    _result(hasattr(pipeline, "analyst"), "Has CVE analyst agent")
    _result(hasattr(pipeline, "assessor"), "Has impact assessor agent")
    _result(hasattr(pipeline, "planner"), "Has remediation planner agent")
    _result(hasattr(pipeline, "pr_creator"), "Has PR creator agent")
    _result(callable(pipeline.run), "Has .run() method")


# ─── 2. Result dict structure ────────────────────────────────────────────────


def test_result_structure() -> None:
    print("\n── 2. Result dict structure ───────────────────────────────────────")
    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[]),
    )
    result = pipeline.run("acme/repo", dry_run=True)

    required_keys = {"success", "scan_id", "repository", "branch", "short_circuit", "summary", "stages", "errors"}
    for key in required_keys:
        _result(key in result, f"Top-level key '{key}' present")

    summary_keys = {"dependencies_found", "vulnerabilities_found", "impact_reports",
                    "actionable_reports", "plans_created", "prs_created", "errors_total"}
    for key in summary_keys:
        _result(key in result["summary"], f"summary.{key} present")


# ─── 3–6. Short-circuit conditions ───────────────────────────────────────────


def test_short_circuit_no_dependencies() -> None:
    print("\n── 3. Short-circuit: no dependencies ──────────────────────────────")
    pipeline = VulnRemedyPipeline(scanner=_MockScanner(dependencies=[]))
    result = pipeline.run("acme/repo", dry_run=True)

    _result(result["short_circuit"] == "no_dependencies", "short_circuit = 'no_dependencies'", result["short_circuit"])
    _result("cve_analysis" not in result["stages"], "CVE Analyst did NOT run")
    _result(result["summary"]["dependencies_found"] == 0, "dependencies_found == 0")


def test_short_circuit_no_findings() -> None:
    print("\n── 4. Short-circuit: no CVE findings ──────────────────────────────")
    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[]),
    )
    result = pipeline.run("acme/repo", dry_run=True)

    _result(result["short_circuit"] == "no_findings", "short_circuit = 'no_findings'", result["short_circuit"])
    _result("impact" not in result["stages"], "Impact Assessor did NOT run")
    _result(result["summary"]["vulnerabilities_found"] == 0, "vulnerabilities_found == 0")


def test_short_circuit_no_actionable_reports() -> None:
    print("\n── 5. Short-circuit: no CRITICAL/HIGH reports ─────────────────────")
    finding = _make_finding(priority_hint=Priority.LOW)
    low_report = _make_report(finding, priority=Priority.LOW)

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[low_report]),
    )
    result = pipeline.run("acme/repo", dry_run=True)

    _result(result["short_circuit"] == "no_actionable_reports", "short_circuit = 'no_actionable_reports'", result["short_circuit"])
    _result("remediation" not in result["stages"], "Planner did NOT run")
    _result(result["summary"]["actionable_reports"] == 0, "actionable_reports == 0")


def test_short_circuit_no_plans() -> None:
    print("\n── 6. Short-circuit: no plans generated ───────────────────────────")
    finding = _make_finding()
    report = _make_report(finding)

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[]),
    )
    result = pipeline.run("acme/repo", dry_run=True)

    _result(result["short_circuit"] == "no_plans", "short_circuit = 'no_plans'", result["short_circuit"])
    _result("prs" not in result["stages"], "PR Creator did NOT run")
    _result(result["summary"]["plans_created"] == 0, "plans_created == 0")


# ─── 7–8. auto_approve ───────────────────────────────────────────────────────


def test_auto_approve_false() -> None:
    print("\n── 7. auto_approve=False → plans stay DRAFT ───────────────────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding, status=RemediationStatus.DRAFT)

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
    )
    result = pipeline.run("acme/repo", dry_run=True, auto_approve=False)

    _result(result["short_circuit"] is None, "Pipeline ran to completion (no short-circuit)")
    _result(plan.status == RemediationStatus.DRAFT, "Plan stays DRAFT (not approved)")
    _result(result["summary"]["prs_created"] == 0, "No PRs created (plan was DRAFT, not APPROVED)")
    _result(len(result["stages"]["prs"]["skipped"]) == 1, "Plan counted as skipped by PR Creator")


def test_auto_approve_true() -> None:
    print("\n── 8. auto_approve=True → plans approved + PRs created ────────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding, status=RemediationStatus.DRAFT)

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
    )
    result = pipeline.run("acme/repo", dry_run=True, auto_approve=True)

    _result(plan.status == RemediationStatus.PR_CREATED, "Plan status → PR_CREATED", plan.status.value)
    _result(result["summary"]["prs_created"] == 1, "prs_created == 1")
    _result(plan.pr_url is not None and "DRY_RUN" in plan.pr_url, "pr_url is synthetic dry-run URL")


# ─── 9. dry_run PR entries ───────────────────────────────────────────────────


def test_dry_run_pr_entries() -> None:
    print("\n── 9. dry_run=True: PR entries structure ──────────────────────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding)

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
    )
    result = pipeline.run("acme/repo", dry_run=True, auto_approve=True)

    prs = result["stages"]["prs"]["prs_created"]
    _result(len(prs) == 1, "One PR entry in stages.prs.prs_created")
    pr = prs[0]
    for key in ("plan_id", "cve_id", "fix_branch", "pr_url", "pr_title", "dry_run"):
        _result(key in pr, f"PR entry has key '{key}'")
    _result(pr["dry_run"] is True, "PR entry.dry_run is True")


# ─── 10. Stage errors accumulated ────────────────────────────────────────────


def test_errors_accumulated() -> None:
    print("\n── 10. Stage errors accumulated ───────────────────────────────────")

    class _ErrorAnalyst:
        def analyze(self, dependencies, scan_id=None, repository="", branch="main"):
            return {
                "success": True,
                "findings": [],
                "dependencies_analyzed": len(dependencies),
                "vulnerabilities_found": 0,
                "errors": ["Simulated analyst error: RAG timeout"],
            }

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep()]),
        analyst=_ErrorAnalyst(),
    )
    result = pipeline.run("acme/repo", dry_run=True)

    _result(
        any("RAG timeout" in e for e in result["errors"]),
        "Analyst error propagated to result['errors']",
    )
    _result(result["success"] is True, "Pipeline still reports success=True (non-fatal error)")


# ─── 11. Live multi-stage: fixture requirements.txt ──────────────────────────


def test_live_multistage_pip_fixtures() -> None:
    print("\n── 11. Live multi-stage: fixture requirements.txt → full pipeline ─")

    # Parse fixture file with real pip parser (no GitHub needed)
    fixture_path = _FIXTURES / "requirements.txt"
    content = fixture_path.read_text(encoding="utf-8")
    parser = PipParser()
    dependencies = parser.parse(content)

    _result(len(dependencies) > 0, f"Parsed {len(dependencies)} dependencies from fixture")

    # Inject at scanner stage, let all downstream agents run for real (with fallbacks)
    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=dependencies),
    )
    result = pipeline.run(
        repo="acme/test-service",
        branch="main",
        dry_run=True,
        auto_approve=True,
    )

    _result(result["success"] is True, "Pipeline run succeeds")
    _result(result["summary"]["dependencies_found"] == len(dependencies), "Dependencies passed through")

    sc = result["short_circuit"]
    _result(True, f"Short-circuit (or none): {sc or 'pipeline ran to end'}")

    # Regardless of how far it got, summary counts should be non-negative integers
    for key, val in result["summary"].items():
        _result(isinstance(val, int) and val >= 0, f"summary.{key} is non-negative int", str(val))


# ─── 12. Summary consistency ─────────────────────────────────────────────────


def test_summary_consistency() -> None:
    print("\n── 12. Summary counts consistent with stage outputs ───────────────")
    finding = _make_finding()
    report = _make_report(finding)
    plan = _make_plan(finding)

    pipeline = VulnRemedyPipeline(
        scanner=_MockScanner(dependencies=[_make_dep(), _make_dep("requests", "2.25.0", Ecosystem.PYPI)]),
        analyst=_MockAnalyst(findings=[finding]),
        assessor=_MockAssessor(reports=[report]),
        planner=_MockPlanner(plans=[plan]),
    )
    result = pipeline.run("acme/repo", dry_run=True, auto_approve=True)

    s = result["summary"]
    stages = result["stages"]

    _result(s["dependencies_found"] == 2, "dependencies_found == 2", str(s["dependencies_found"]))
    _result(s["vulnerabilities_found"] == len(stages["cve_analysis"]["findings"]), "vulnerabilities_found matches findings list")
    _result(s["impact_reports"] == len(stages["impact"]["impact_reports"]), "impact_reports count matches")
    _result(s["plans_created"] == len(stages["remediation"]["plans"]), "plans_created count matches")
    _result(s["prs_created"] == len(stages["prs"]["prs_created"]), "prs_created count matches")
    _result(s["errors_total"] == len(result["errors"]), "errors_total matches len(errors)")


# ─── Runner ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 60)
    print("  VulnRemedy Pipeline — Integration Test Suite")
    print("=" * 60)

    test_instantiation()
    test_result_structure()
    test_short_circuit_no_dependencies()
    test_short_circuit_no_findings()
    test_short_circuit_no_actionable_reports()
    test_short_circuit_no_plans()
    test_auto_approve_false()
    test_auto_approve_true()
    test_dry_run_pr_entries()
    test_errors_accumulated()
    test_live_multistage_pip_fixtures()
    test_summary_consistency()

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
