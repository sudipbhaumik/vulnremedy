"""
Impact Assessor Agent — LLM-powered exploitability analysis and prioritisation.

Pipeline per Finding:
  1. Query RAG for exploit details (attack vectors, PoC context, CVSS breakdown)
  2. Sanitise RAG context through guardrails (injection detection, length cap)
  3. Build dependency path (project → manifest → vulnerable package)
  4. Check circuit breaker + apply rate limit via guardrails
  5. Call Ollama LLM with structured prompt loaded from prompts/ directory
  6. Validate LLM response through guardrails (type coercion, field truncation)
  7. Parse validated response into ImpactReport
  8. Graceful fallback to deterministic scoring if LLM is unavailable or circuit open

Output: list[ImpactReport] — each Finding enriched with:
  - is_exploitable (LLM-reasoned)
  - exploitability_reasoning (LLM-generated explanation)
  - business_impact (LLM-generated consequence description)
  - priority (CRITICAL / HIGH / MEDIUM / LOW)
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

import httpx

from vulnremedy.agents.impact_assessor.guardrails import ImpactAssessorGuardrails
from vulnremedy.models.cve import Severity
from vulnremedy.models.finding import DependencyType, Finding
from vulnremedy.models.impact import ImpactReport, Priority
from vulnremedy.rag.embeddings.embedding_service import EmbeddingService
from vulnremedy.rag.retrieval.retriever import HybridRetriever
from vulnremedy.rag.retrieval.vector_store import VectorStore
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


# ─── Fallback scoring tables ──────────────────────────────────────────────────
# Used when the LLM is unavailable. Mirrors the exploitability heuristics
# a human analyst would apply based solely on severity + dependency type.

_SEVERITY_TO_PRIORITY: dict[Severity, Priority] = {
    Severity.CRITICAL: Priority.CRITICAL,
    Severity.HIGH: Priority.HIGH,
    Severity.MEDIUM: Priority.MEDIUM,
    Severity.LOW: Priority.LOW,
    Severity.NONE: Priority.LOW,
    Severity.UNKNOWN: Priority.MEDIUM,
}

_EXPLOITABLE_BY_DEFAULT = {Severity.CRITICAL, Severity.HIGH}


class ImpactAssessorAgent:
    """
    Assesses the real-world exploitability and business impact of each Finding.

    Uses a RAG → LLM pipeline:
      - RAG retrieves exploit details and attack context for the CVE
      - LLM reasons about whether the vulnerability is actually reachable
        given the dependency path and service context

    Falls back to deterministic severity-based scoring if Ollama is unavailable.

    Usage:
        assessor = ImpactAssessorAgent()
        result = assessor.assess(
            findings=cve_result["findings"],
            scan_id=my_scan_id,
        )
        for report in result["impact_reports"]:
            print(report.priority, report.exploitability_reasoning)
    """

    def __init__(self) -> None:
        # RAG components — same pattern as CVE Analyst
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service)

        # Guardrails — injection detection, circuit breaker, rate limiter, response validation
        self.guardrails = ImpactAssessorGuardrails()

        # Load prompt template from prompts/ directory once at startup
        self.prompt_template = settings.load_prompt("impact_assessor_exploitability.txt")

        # Ollama HTTP client
        self._http = httpx.Client(timeout=120.0)

        logger.info(
            "Impact Assessor Agent initialized",
            llm_model=settings.llm_model,
            ollama_url=settings.ollama_base_url,
        )

    def assess(
        self,
        findings: list[Finding],
        scan_id: Optional[UUID] = None,
    ) -> dict:
        """
        Assess impact for all findings.

        Args:
            findings: list[Finding] from the CVE Analyst agent.
            scan_id:  Scan UUID for correlation logging.

        Returns:
            Dictionary with:
                - success:              bool
                - impact_reports:       list[ImpactReport]
                - high_priority_count:  int
                - errors:               list[str]
        """
        logger.info(
            "Starting impact assessment",
            total_findings=len(findings),
            scan_id=str(scan_id) if scan_id else "unknown",
        )

        reports: list[ImpactReport] = []
        errors: list[str] = []

        for finding in findings:
            try:
                report = self._assess_finding(finding)
                reports.append(report)
                logger.info(
                    "Finding assessed",
                    cve_id=finding.cve.cve_id,
                    package=finding.affected_dependency.fully_qualified_name,
                    priority=report.priority.value,
                    is_exploitable=report.is_exploitable,
                    used_fallback=report.used_fallback,
                )
            except Exception as exc:
                logger.error(
                    "Error assessing finding",
                    cve_id=finding.cve.cve_id,
                    package=finding.affected_dependency.fully_qualified_name,
                    error=str(exc),
                )
                errors.append(
                    f"Error assessing {finding.cve.cve_id} "
                    f"({finding.affected_dependency.fully_qualified_name}): {exc}"
                )

        high_priority_count = sum(
            1 for r in reports if r.priority in (Priority.CRITICAL, Priority.HIGH)
        )

        logger.info(
            "Impact assessment complete",
            total=len(findings),
            reports=len(reports),
            high_priority=high_priority_count,
            errors=len(errors),
        )

        return {
            "success": True,
            "impact_reports": reports,
            "high_priority_count": high_priority_count,
            "errors": errors,
        }

    # ─── Per-finding pipeline ───────────────────────────────────────────────────

    def _assess_finding(self, finding: Finding) -> ImpactReport:
        """Run the full RAG → guardrails → LLM pipeline for one Finding."""
        # Step 1: Query RAG for exploit context
        exploit_chunks = self._query_exploit_details(finding)

        # Step 2: Sanitise context through guardrails (injection detection + length cap)
        exploit_context = self.guardrails.sanitize_exploit_context(exploit_chunks)

        # Step 3: Build dependency path
        dependency_path = self._build_dependency_path(finding)

        # Step 4: Call LLM (circuit breaker + rate limit + response validation via guardrails)
        prompt = self._build_exploitability_prompt(
            finding=finding,
            exploit_context=exploit_context,
            dependency_path=dependency_path,
        )
        llm_result, used_fallback = self._call_llm_with_fallback(finding, prompt)

        # Step 5: Determine affected services from repository name
        affected_services = (
            [finding.repository] if finding.repository != "unknown" else []
        )

        return ImpactReport(
            finding=finding,
            affected_services=affected_services,
            dependency_path=dependency_path,
            is_exploitable=llm_result["is_exploitable"],
            exploitability_reasoning=llm_result["reasoning"],
            business_impact=llm_result["business_impact"],
            priority=Priority(llm_result["recommended_priority"]),
            llm_model_used=None if used_fallback else settings.llm_model,
            used_fallback=used_fallback,
        )

    # ─── RAG query ──────────────────────────────────────────────────────────────

    def _query_exploit_details(self, finding: Finding) -> list[dict]:
        """
        Query RAG for exploit details, attack vectors, and CVE context.

        Targets a more specific query than the CVE Analyst — we want
        attack vector and impact details, not just CVE presence.
        """
        dep = finding.affected_dependency
        cve_id = finding.cve.cve_id
        query = (
            f"{cve_id} {dep.fully_qualified_name} "
            f"exploit attack vector vulnerability details"
        )

        try:
            results = self.retriever.retrieve(
                query=query,
                top_k=3,
                filters={"ecosystems": [dep.ecosystem.value]},
            )
        except Exception as e:
            logger.warning(
                f"RAG exploit query with filter failed, retrying without filter: {e}"
            )
            try:
                results = self.retriever.retrieve(query=query, top_k=3)
            except Exception as e2:
                logger.warning(f"RAG exploit query failed entirely: {e2}")
                results = []

        logger.debug(
            "RAG exploit query complete",
            cve_id=cve_id,
            package=dep.fully_qualified_name,
            results=len(results),
        )
        return results

    # ─── Dependency path ────────────────────────────────────────────────────────

    @staticmethod
    def _build_dependency_path(finding: Finding) -> list[str]:
        """
        Build a human-readable dependency chain from project root to the
        vulnerable package.

        DIRECT:     repo-name → manifest-file → package@version
        TRANSITIVE: repo-name → manifest-file → (transitive) → package@version

        A production implementation would parse the build tool's lock file
        to show the full intermediate dependency chain.
        """
        dep = finding.affected_dependency

        # Extract a short repo name from the full repository identifier
        repo_name = (
            finding.repository.split("/")[-1]
            if "/" in finding.repository
            else finding.repository
        )

        package_node = f"{dep.fully_qualified_name}@{dep.current_version}"

        if dep.dependency_type == DependencyType.DIRECT:
            return [repo_name, dep.manifest_file, package_node]

        if dep.dependency_type == DependencyType.TRANSITIVE:
            return [repo_name, dep.manifest_file, "(transitive dependency)", package_node]

        return [repo_name, dep.manifest_file, package_node]

    # ─── LLM integration ────────────────────────────────────────────────────────

    def _build_exploitability_prompt(
        self,
        finding: Finding,
        exploit_context: str,
        dependency_path: list[str],
    ) -> str:
        """
        Build the structured prompt by filling the externalized template.

        The template is loaded from prompts/impact_assessor_exploitability.txt
        at agent initialisation time so prompt changes require no code edits.
        """
        dep = finding.affected_dependency
        return self.prompt_template.format(
            cve_id=finding.cve.cve_id,
            severity=finding.severity.value.upper(),
            package=dep.fully_qualified_name,
            current_version=dep.current_version,
            dependency_type=dep.dependency_type.value,
            manifest_file=dep.manifest_file,
            repository=finding.repository,
            fixed_version=dep.fixed_version or "unknown",
            exploit_context=exploit_context,
            dependency_path=" → ".join(dependency_path),
        )

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """
        Call Ollama /api/generate and parse + validate the JSON response.

        Uses format=json to request structured output.
        Temperature 0.1 keeps analysis consistent and factual.
        Response is passed through guardrails for type coercion and field sanitisation.
        """
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        logger.debug("Calling Ollama LLM", model=settings.llm_model)

        response = self._http.post(url, json=payload)
        response.raise_for_status()

        raw_text = response.json().get("response", "")
        parsed = json.loads(raw_text)

        # Validate all required fields are present before guardrails processing
        required = {"is_exploitable", "reasoning", "business_impact", "recommended_priority"}
        missing = required - parsed.keys()
        if missing:
            raise ValueError(f"LLM response missing required fields: {missing}")

        # Delegate type coercion, truncation, and priority normalisation to guardrails
        return self.guardrails.validate_llm_response(parsed)

    def _call_llm_with_fallback(
        self,
        finding: Finding,
        prompt: str,
    ) -> tuple[dict[str, Any], bool]:
        """
        Attempt LLM call; return deterministic fallback on any failure or open circuit.

        Order of operations:
          1. Circuit breaker check — skip LLM if too many consecutive failures.
          2. Rate limit enforcement — sleep if needed between calls.
          3. LLM call + guardrail validation.
          4. Record success (resets circuit) or failure (increments circuit counter).

        Returns:
            (result_dict, used_fallback)  —  used_fallback=True when LLM failed or circuit open.
        """
        # Circuit breaker check
        allowed, reason = self.guardrails.should_allow_llm_call()
        if not allowed:
            logger.warning(
                "Circuit breaker prevented LLM call — using deterministic fallback",
                cve_id=finding.cve.cve_id,
                reason=reason,
            )
            return self._deterministic_fallback(finding), True

        # Rate limit enforcement
        self.guardrails.apply_rate_limit()

        try:
            result = self._call_llm(prompt)
            self.guardrails.record_success()
            return result, False
        except Exception as exc:
            self.guardrails.record_failure()
            logger.warning(
                "LLM call failed — using deterministic fallback",
                cve_id=finding.cve.cve_id,
                package=finding.affected_dependency.fully_qualified_name,
                error=str(exc),
            )
            return self._deterministic_fallback(finding), True

    @staticmethod
    def _deterministic_fallback(finding: Finding) -> dict[str, Any]:
        """
        Severity-based fallback assessment when the LLM is unavailable.

        Conservative: assumes exploitability for CRITICAL/HIGH.
        Downgrades CRITICAL to HIGH for transitive dependencies
        (harder to reach directly).
        """
        dep = finding.affected_dependency
        is_exploitable = finding.severity in _EXPLOITABLE_BY_DEFAULT
        priority = _SEVERITY_TO_PRIORITY.get(finding.severity, Priority.MEDIUM)

        # Transitive CRITICAL findings are serious but less directly reachable
        if dep.dependency_type == DependencyType.TRANSITIVE and priority == Priority.CRITICAL:
            priority = Priority.HIGH

        reasoning = (
            f"LLM unavailable — deterministic fallback based on severity "
            f"({finding.severity.value.upper()}). "
            f"{dep.fully_qualified_name} is a {dep.dependency_type.value} dependency "
            f"declared in {dep.manifest_file}. "
            + (
                f"Fixed version {dep.fixed_version} is available."
                if dep.fixed_version
                else "No fixed version identified."
            )
        )

        business_impact = (
            f"A {finding.severity.value.upper()} severity vulnerability "
            f"({finding.cve.cve_id}) in {dep.fully_qualified_name} "
            f"could compromise the {finding.repository} service. "
            "Manual review required to confirm exploitability in this context."
        )

        return {
            "is_exploitable": is_exploitable,
            "reasoning": reasoning,
            "business_impact": business_impact,
            "recommended_priority": priority.value,
        }
