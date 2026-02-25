"""
Remediation Planner Agent — LLM-powered upgrade strategy generation.

Pipeline per ImpactReport (CRITICAL / HIGH priority only):
  1. Query RAG for migration guides and fix guidance for the CVE
  2. Sanitise RAG context through guardrails (injection detection, length cap)
  3. Build structured prompt from externalized template
  4. Check circuit breaker + apply rate limit via guardrails
  5. Call Ollama LLM → JSON strategy response
  6. Validate response through guardrails (strategy enum, truncation, list caps)
  7. Generate CodeChange objects via _generate_code_changes():
       - VERSION_UPGRADE on direct deps → deterministic regex-based snippet (no LLM)
       - TRANSITIVE_OVERRIDE / BOM_OVERRIDE / CODE_REFACTOR → LLM codegen prompt
       - MANUAL_REVIEW → empty list (human must act)
  8. Assemble RemediationPlan with populated code_changes, status=DRAFT
  9. Graceful fallback to rule-based strategy if LLM unavailable or circuit open

Filtering:
  Only CRITICAL and HIGH priority ImpactReports are planned.
  MEDIUM / LOW are returned as skipped (not errored).

Output: dict with:
  - success:           bool
  - plans:             list[RemediationPlan]
  - skipped_count:     int  (MEDIUM/LOW reports, not errors)
  - errors:            list[str]
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional
from uuid import UUID

import httpx

from vulnremedy.agents.remediation.guardrails import RemediationGuardrails
from vulnremedy.models.cve import Ecosystem
from vulnremedy.models.finding import AffectedDependency, DependencyType
from vulnremedy.models.impact import ImpactReport, Priority
from vulnremedy.models.remediation import (
    CodeChange,
    RemediationPlan,
    RemediationStatus,
    RemediationStrategy,
)
from vulnremedy.rag.embeddings.embedding_service import EmbeddingService
from vulnremedy.rag.retrieval.retriever import HybridRetriever
from vulnremedy.rag.retrieval.vector_store import VectorStore
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


# ─── Rule-based fallback tables ───────────────────────────────────────────────
# Applied when the LLM is unavailable. Encodes the most common remediation
# pattern for each combination of dependency type and ecosystem.

_DIRECT_STRATEGY = RemediationStrategy.VERSION_UPGRADE
_TRANSITIVE_STRATEGY = RemediationStrategy.TRANSITIVE_OVERRIDE

_FALLBACK_TESTING_PLAN = [
    "Run the full unit test suite",
    "Run integration tests covering the affected dependency's functionality",
    "Check application startup for new deprecation warnings or errors",
    "Verify no regression in features that use the upgraded library",
]

_FALLBACK_ROLLBACK_PLAN = [
    "Revert the manifest change (restore the original version)",
    "Re-run tests to confirm the rollback is clean",
    "Re-open the vulnerability finding for re-assessment",
]


# ─── Ecosystem-specific version bump generators ───────────────────────────────
# Pure functions — no LLM, no side effects.
# Each accepts the AffectedDependency and optional current manifest content.
# Returns a CodeChange whose original_content/new_content act as a
# search-and-replace pair for the PR Creator.

def _generate_version_bump(
    dep: AffectedDependency,
    manifest_content: str,
) -> Optional[CodeChange]:
    """Dispatch to the right ecosystem generator."""
    if not dep.fixed_version:
        return None

    mf = dep.manifest_file.lower()

    if dep.ecosystem == Ecosystem.MAVEN or mf.endswith("pom.xml"):
        return _maven_version_bump(dep, manifest_content)
    if dep.ecosystem == Ecosystem.NPM or mf.endswith("package.json"):
        return _npm_version_bump(dep, manifest_content)
    if dep.ecosystem == Ecosystem.PYPI or re.search(r"requirements.*\.txt$", mf):
        return _pip_version_bump(dep, manifest_content)
    if dep.ecosystem == Ecosystem.GRADLE or mf.endswith((".gradle", ".gradle.kts")):
        return _gradle_version_bump(dep, manifest_content)

    # Generic fallback — just record the version tokens
    return CodeChange(
        file_path=dep.manifest_file,
        change_type="version_bump",
        original_content=dep.current_version,
        new_content=dep.fixed_version,
        rationale=(
            f"Upgrade {dep.fully_qualified_name} from "
            f"{dep.current_version} to {dep.fixed_version}"
        ),
    )


def _maven_version_bump(dep: AffectedDependency, manifest_content: str) -> CodeChange:
    """
    Generate a Maven pom.xml version bump CodeChange.

    If manifest_content is provided, extract the exact <dependency> block
    so original_content is a precise substring of the real file.
    Falls back to constructing the expected block from dep metadata.
    """
    group_id = dep.group_id or ""
    artifact_id = dep.package_name
    old_ver = dep.current_version
    new_ver = dep.fixed_version  # type: ignore[assignment]

    # Try to find the exact block inside the provided file content
    if manifest_content and group_id:
        pattern = (
            r"(<dependency>[^<]*"
            r"<groupId>" + re.escape(group_id) + r"</groupId>[^<]*"
            r"<artifactId>" + re.escape(artifact_id) + r"</artifactId>[^<]*"
            r"<version>)" + re.escape(old_ver) + r"(</version>[^<]*</dependency>)"
        )
        m = re.search(pattern, manifest_content, re.DOTALL)
        if m:
            old_block = m.group(0)
            new_block = m.group(1) + new_ver + m.group(2)
            return CodeChange(
                file_path=dep.manifest_file,
                change_type="version_bump",
                original_content=old_block,
                new_content=new_block,
                rationale=(
                    f"Upgrade {dep.fully_qualified_name} from {old_ver} to {new_ver} "
                    f"to remediate {dep.manifest_file}"
                ),
            )

    # Fallback: synthesise the canonical block from dep metadata
    indent = "        "
    if group_id:
        old_block = (
            f"<dependency>\n"
            f"{indent}    <groupId>{group_id}</groupId>\n"
            f"{indent}    <artifactId>{artifact_id}</artifactId>\n"
            f"{indent}    <version>{old_ver}</version>\n"
            f"{indent}</dependency>"
        )
    else:
        old_block = f"<version>{old_ver}</version>"
    new_block = old_block.replace(f"<version>{old_ver}</version>", f"<version>{new_ver}</version>")

    return CodeChange(
        file_path=dep.manifest_file,
        change_type="version_bump",
        original_content=old_block,
        new_content=new_block,
        rationale=f"Upgrade {dep.fully_qualified_name} from {old_ver} to {new_ver}",
    )


def _npm_version_bump(dep: AffectedDependency, manifest_content: str) -> CodeChange:
    """
    Generate a package.json version bump CodeChange.

    Matches  "package-name": "VERSION"  (any semver prefix: ^, ~, =, or none).
    """
    pkg = dep.package_name
    old_ver = dep.current_version
    new_ver = dep.fixed_version  # type: ignore[assignment]

    # Version specifier may have a prefix like ^ or ~
    prefix_pattern = r'[\^~>=<]+'
    if manifest_content:
        pattern = (
            r'("' + re.escape(pkg) + r'":\s*")'
            + r'(' + prefix_pattern + r')?'
            + re.escape(old_ver)
            + r'(")'
        )
        m = re.search(pattern, manifest_content)
        if m:
            old_snippet = m.group(0)
            prefix = m.group(2) or ""
            new_snippet = f'"{pkg}": "{prefix}{new_ver}"'
            return CodeChange(
                file_path=dep.manifest_file,
                change_type="version_bump",
                original_content=old_snippet,
                new_content=new_snippet,
                rationale=f"Upgrade {pkg} from {old_ver} to {new_ver}",
            )

    # Fallback: no prefix assumed
    return CodeChange(
        file_path=dep.manifest_file,
        change_type="version_bump",
        original_content=f'"{pkg}": "{old_ver}"',
        new_content=f'"{pkg}": "{new_ver}"',
        rationale=f"Upgrade {pkg} from {old_ver} to {new_ver}",
    )


def _pip_version_bump(dep: AffectedDependency, manifest_content: str) -> CodeChange:
    """
    Generate a requirements.txt version bump CodeChange.

    Handles  package==OLD  and  package==OLD ; marker  patterns.
    Case-insensitive on the package name (pip normalises to lowercase).
    """
    pkg = dep.package_name
    old_ver = dep.current_version
    new_ver = dep.fixed_version  # type: ignore[assignment]

    if manifest_content:
        # Match   Package==1.2.3   or   Package==1.2.3 ; marker
        pattern = r'(?i)(' + re.escape(pkg) + r'==)' + re.escape(old_ver) + r'(\b[^\n]*)?'
        m = re.search(pattern, manifest_content)
        if m:
            old_line = m.group(0)
            suffix = m.group(2) or ""
            new_line = f"{pkg}=={new_ver}{suffix}"
            return CodeChange(
                file_path=dep.manifest_file,
                change_type="version_bump",
                original_content=old_line,
                new_content=new_line,
                rationale=f"Upgrade {pkg} from {old_ver} to {new_ver}",
            )

    return CodeChange(
        file_path=dep.manifest_file,
        change_type="version_bump",
        original_content=f"{pkg}=={old_ver}",
        new_content=f"{pkg}=={new_ver}",
        rationale=f"Upgrade {pkg} from {old_ver} to {new_ver}",
    )


def _gradle_version_bump(dep: AffectedDependency, manifest_content: str) -> CodeChange:
    """
    Generate a Gradle version bump CodeChange (Groovy or Kotlin DSL).

    Matches both forms:
      Groovy:  implementation 'group:artifact:VERSION'
      Kotlin:  implementation("group:artifact:VERSION")
    """
    group_id = dep.group_id or ""
    artifact_id = dep.package_name
    old_ver = dep.current_version
    new_ver = dep.fixed_version  # type: ignore[assignment]

    coord = f"{group_id}:{artifact_id}" if group_id else artifact_id

    if manifest_content:
        # Match both quote styles and both with/without parens
        pattern = (
            r'([\w]+\s*[\(\s]["\'])' + re.escape(coord) + r':'
            + re.escape(old_ver) + r'(["\'][\)]?)'
        )
        m = re.search(pattern, manifest_content)
        if m:
            old_snippet = m.group(0)
            new_snippet = old_snippet.replace(f":{old_ver}", f":{new_ver}")
            return CodeChange(
                file_path=dep.manifest_file,
                change_type="version_bump",
                original_content=old_snippet,
                new_content=new_snippet,
                rationale=f"Upgrade {coord} from {old_ver} to {new_ver}",
            )

    # Fallback: Groovy-style single-quote form
    return CodeChange(
        file_path=dep.manifest_file,
        change_type="version_bump",
        original_content=f"'{coord}:{old_ver}'",
        new_content=f"'{coord}:{new_ver}'",
        rationale=f"Upgrade {coord} from {old_ver} to {new_ver}",
    )


class RemediationPlannerAgent:
    """
    Generates RemediationPlan objects for CRITICAL/HIGH priority findings.

    Uses a RAG → LLM pipeline:
      - RAG retrieves fix guidance and migration notes for the CVE
      - LLM selects a strategy and elaborates on upgrade approach,
        breaking changes, config changes, testing plan, and rollback plan

    Falls back to a deterministic rule-based strategy when Ollama is unavailable.
    code_changes is always [] — populated in a future Week 5 code generation step.

    Usage:
        planner = RemediationPlannerAgent()
        result = planner.plan(
            impact_reports=assessor_result["impact_reports"],
            scan_id=my_scan_id,
        )
        for plan in result["plans"]:
            print(plan.strategy, plan.summary)
    """

    def __init__(self) -> None:
        # RAG components
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service)

        # Guardrails
        self.guardrails = RemediationGuardrails()

        # Load prompt templates once at startup
        self.prompt_template = settings.load_prompt("remediation_planner_strategy.txt")
        self.codegen_prompt_template = settings.load_prompt("remediation_planner_codegen.txt")

        # Ollama HTTP client
        self._http = httpx.Client(timeout=120.0)

        logger.info(
            "Remediation Planner Agent initialized",
            llm_model=settings.llm_model,
            ollama_url=settings.ollama_base_url,
        )

    # ─── Public entry point ───────────────────────────────────────────────────

    def plan(
        self,
        impact_reports: list[ImpactReport],
        scan_id: Optional[UUID] = None,
    ) -> dict[str, Any]:
        """
        Generate remediation plans for CRITICAL/HIGH priority findings.

        Args:
            impact_reports: list[ImpactReport] from the Impact Assessor.
            scan_id:        Scan UUID for correlation logging.

        Returns:
            Dictionary with:
                - success:       bool
                - plans:         list[RemediationPlan]
                - skipped_count: int  (MEDIUM/LOW reports not planned)
                - errors:        list[str]
        """
        logger.info(
            "Starting remediation planning",
            total_reports=len(impact_reports),
            scan_id=str(scan_id) if scan_id else "unknown",
        )

        plans: list[RemediationPlan] = []
        errors: list[str] = []
        skipped_count = 0

        for report in impact_reports:
            dep = report.finding.affected_dependency
            cve_id = report.finding.cve.cve_id

            # Filter — only plan for actionable priorities
            if report.priority not in (Priority.CRITICAL, Priority.HIGH):
                logger.debug(
                    "Skipping MEDIUM/LOW report",
                    cve_id=cve_id,
                    package=dep.fully_qualified_name,
                    priority=report.priority.value,
                )
                skipped_count += 1
                continue

            try:
                plan = self._plan_for_report(report)
                plans.append(plan)
                logger.info(
                    "Remediation plan created",
                    cve_id=cve_id,
                    package=dep.fully_qualified_name,
                    strategy=plan.strategy.value,
                    priority=report.priority.value,
                    used_fallback=not plan.detailed_steps
                    or plan.detailed_steps[0].startswith("LLM unavailable"),
                )
            except Exception as exc:
                logger.error(
                    "Error creating remediation plan",
                    cve_id=cve_id,
                    package=dep.fully_qualified_name,
                    error=str(exc),
                )
                errors.append(
                    f"Error planning {cve_id} "
                    f"({dep.fully_qualified_name}): {exc}"
                )

        logger.info(
            "Remediation planning complete",
            total=len(impact_reports),
            planned=len(plans),
            skipped=skipped_count,
            errors=len(errors),
        )

        return {
            "success": True,
            "plans": plans,
            "skipped_count": skipped_count,
            "errors": errors,
        }

    # ─── Per-report pipeline ──────────────────────────────────────────────────

    def _plan_for_report(self, report: ImpactReport) -> RemediationPlan:
        """Run the full RAG → guardrails → LLM pipeline for one ImpactReport."""
        finding = report.finding
        dep = finding.affected_dependency

        # Step 1: Query RAG for migration guides / fix guidance
        migration_chunks = self._query_migration_guides(report)

        # Step 2: Sanitise context
        migration_context = self.guardrails.sanitize_migration_context(migration_chunks)

        # Step 3: Build prompt
        prompt = self._build_strategy_prompt(report, migration_context)

        # Step 4: Call LLM (circuit breaker + rate limit + guardrail validation)
        llm_result, used_fallback = self._call_llm_with_fallback(report, prompt)

        # Step 5: Assemble RemediationPlan
        strategy = RemediationStrategy(llm_result["strategy"])

        summary = (
            f"Upgrade {dep.fully_qualified_name} from "
            f"{dep.current_version} to {dep.fixed_version or 'latest safe version'} "
            f"to remediate {finding.cve.cve_id}"
        )

        detailed_steps = self._build_detailed_steps(llm_result, dep, used_fallback)

        # Step 6: Generate code changes (deterministic for simple upgrades, LLM for complex)
        code_changes = self._generate_code_changes(
            strategy=strategy,
            dep=dep,
            finding=finding,
        )

        return RemediationPlan(
            finding_id=finding.id,
            finding=finding,
            strategy=strategy,
            status=RemediationStatus.DRAFT,
            code_changes=code_changes,
            summary=summary,
            detailed_steps=detailed_steps,
            breaking_changes=llm_result["breaking_changes"],
            testing_recommendations=llm_result["testing_plan"],
            requires_human_approval=self._requires_approval(report, strategy),
        )

    # ─── RAG query ────────────────────────────────────────────────────────────

    def _query_migration_guides(self, report: ImpactReport) -> list[dict[str, Any]]:
        """
        Query RAG for fix guidance and migration notes for this CVE.

        Targets ``fix_guidance`` and ``migration`` chunk types specifically —
        the RAG corpus stores upgrade instructions separately from CVE descriptions.
        """
        dep = report.finding.affected_dependency
        cve_id = report.finding.cve.cve_id
        fixed = dep.fixed_version or ""
        query = (
            f"{cve_id} {dep.fully_qualified_name} "
            f"fix upgrade migration guide {fixed}"
        )

        try:
            results = self.retriever.retrieve(
                query=query,
                top_k=3,
                filters={"ecosystems": [dep.ecosystem.value]},
            )
        except Exception as e:
            logger.warning(
                f"RAG migration query with filter failed, retrying without filter: {e}"
            )
            try:
                results = self.retriever.retrieve(query=query, top_k=3)
            except Exception as e2:
                logger.warning(f"RAG migration query failed entirely: {e2}")
                results = []

        logger.debug(
            "RAG migration query complete",
            cve_id=cve_id,
            package=dep.fully_qualified_name,
            results=len(results),
        )
        return results

    # ─── Prompt building ─────────────────────────────────────────────────────

    def _build_strategy_prompt(
        self,
        report: ImpactReport,
        migration_context: str,
    ) -> str:
        """
        Fill the externalized prompt template with values from the ImpactReport.
        """
        finding = report.finding
        dep = finding.affected_dependency
        return self.prompt_template.format(
            cve_id=finding.cve.cve_id,
            severity=finding.severity.value.upper(),
            priority=report.priority.value,
            is_exploitable=report.is_exploitable,
            package=dep.fully_qualified_name,
            current_version=dep.current_version,
            fixed_version=dep.fixed_version or "unknown",
            dependency_type=dep.dependency_type.value,
            manifest_file=dep.manifest_file,
            ecosystem=dep.ecosystem.value,
            repository=finding.repository,
            dependency_path=" → ".join(report.dependency_path),
            exploitability_reasoning=report.exploitability_reasoning,
            business_impact=report.business_impact,
            migration_context=migration_context,
        )

    # ─── LLM integration ──────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call Ollama and return the validated JSON strategy response."""
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 768,
            },
        }

        logger.debug("Calling Ollama LLM for remediation strategy", model=settings.llm_model)

        response = self._http.post(url, json=payload)
        response.raise_for_status()

        raw_text = response.json().get("response", "")
        parsed = json.loads(raw_text)

        # Require at minimum strategy and upgrade_approach
        required = {"strategy", "upgrade_approach", "testing_plan", "rollback_plan"}
        missing = required - parsed.keys()
        if missing:
            raise ValueError(f"LLM response missing required fields: {missing}")

        # Default optional list fields to empty list if absent
        for field in ("breaking_changes", "config_changes"):
            parsed.setdefault(field, [])

        return self.guardrails.validate_llm_response(parsed)

    def _call_llm_with_fallback(
        self,
        report: ImpactReport,
        prompt: str,
    ) -> tuple[dict[str, Any], bool]:
        """
        Attempt LLM call; return rule-based fallback on failure or open circuit.

        Returns:
            (result_dict, used_fallback) — used_fallback=True when LLM failed or circuit open.
        """
        allowed, reason = self.guardrails.should_allow_llm_call()
        if not allowed:
            logger.warning(
                "Remediation circuit breaker prevented LLM call — using rule-based fallback",
                cve_id=report.finding.cve.cve_id,
                reason=reason,
            )
            return self._rule_based_fallback(report), True

        self.guardrails.apply_rate_limit()

        try:
            result = self._call_llm(prompt)
            self.guardrails.record_success()
            return result, False
        except Exception as exc:
            self.guardrails.record_failure()
            logger.warning(
                "LLM call failed — using rule-based fallback",
                cve_id=report.finding.cve.cve_id,
                package=report.finding.affected_dependency.fully_qualified_name,
                error=str(exc),
            )
            return self._rule_based_fallback(report), True

    # ─── Rule-based fallback ─────────────────────────────────────────────────

    @staticmethod
    def _rule_based_fallback(report: ImpactReport) -> dict[str, Any]:
        """
        Select strategy and generate minimal guidance without the LLM.

        Rules:
          DIRECT dependency   → VERSION_UPGRADE
          TRANSITIVE dependency → TRANSITIVE_OVERRIDE
          Fixed version unknown  → MANUAL_REVIEW
        """
        dep = report.finding.affected_dependency
        fixed = dep.fixed_version

        if not fixed:
            strategy = RemediationStrategy.MANUAL_REVIEW
            approach = (
                f"LLM unavailable — no fixed version identified for "
                f"{dep.fully_qualified_name}. Manual review required."
            )
        elif dep.dependency_type == DependencyType.TRANSITIVE:
            strategy = _TRANSITIVE_STRATEGY
            approach = (
                f"LLM unavailable — rule-based fallback. "
                f"{dep.fully_qualified_name} is a transitive dependency. "
                f"Add an explicit version override to force version {fixed} "
                f"in {dep.manifest_file}."
            )
        else:
            strategy = _DIRECT_STRATEGY
            approach = (
                f"LLM unavailable — rule-based fallback. "
                f"Update {dep.fully_qualified_name} from {dep.current_version} "
                f"to {fixed} in {dep.manifest_file}."
            )

        return {
            "strategy": strategy.value,
            "upgrade_approach": approach,
            "breaking_changes": [],
            "config_changes": [],
            "testing_plan": _FALLBACK_TESTING_PLAN,
            "rollback_plan": _FALLBACK_ROLLBACK_PLAN,
        }

    # ─── Code generation ─────────────────────────────────────────────────────

    def _generate_code_changes(
        self,
        strategy: RemediationStrategy,
        dep: AffectedDependency,
        finding: Any,
        manifest_content: str = "",
    ) -> list[CodeChange]:
        """
        Generate CodeChange objects for the resolved strategy.

        Two tracks:
          VERSION_UPGRADE  → deterministic regex-based snippet (no LLM needed).
          Complex strategies → LLM codegen prompt (falls back to empty on failure).
          MANUAL_REVIEW    → always empty (human acts).

        Args:
            strategy:         The resolved RemediationStrategy.
            dep:              AffectedDependency with package, version, ecosystem info.
            finding:          The parent Finding (for CVE context in the LLM prompt).
            manifest_content: Optional current file content for exact-match extraction.

        Returns:
            list[CodeChange] — may be empty if the strategy cannot be automated.
        """
        if strategy == RemediationStrategy.MANUAL_REVIEW:
            logger.debug(
                "No code changes generated for MANUAL_REVIEW strategy",
                package=dep.fully_qualified_name,
            )
            return []

        if strategy == RemediationStrategy.VERSION_UPGRADE:
            change = _generate_version_bump(dep, manifest_content)
            if change:
                logger.debug(
                    "Deterministic version bump generated",
                    package=dep.fully_qualified_name,
                    manifest=dep.manifest_file,
                )
                return [change]
            return []

        # Complex strategies — delegate to LLM
        try:
            return self._generate_via_llm(strategy, dep, finding, manifest_content)
        except Exception as exc:
            logger.warning(
                "LLM codegen failed — returning empty code_changes",
                strategy=strategy.value,
                package=dep.fully_qualified_name,
                error=str(exc),
            )
            return []

    def _generate_via_llm(
        self,
        strategy: RemediationStrategy,
        dep: AffectedDependency,
        finding: Any,
        manifest_content: str,
    ) -> list[CodeChange]:
        """Call the codegen LLM and parse one CodeChange from its response."""
        prompt = self.codegen_prompt_template.format(
            cve_id=finding.cve.cve_id,
            package=dep.fully_qualified_name,
            current_version=dep.current_version,
            fixed_version=dep.fixed_version or "latest",
            ecosystem=dep.ecosystem.value,
            strategy=strategy.value,
            manifest_file=dep.manifest_file,
            manifest_content=manifest_content or "(content not available — generate based on ecosystem conventions)",
        )

        result = self._call_codegen_llm(prompt)

        change = CodeChange(
            file_path=result.get("file_path", dep.manifest_file),
            change_type=result.get("change_type", strategy.value),
            original_content=result.get("original_content") or None,
            new_content=result.get("new_content", ""),
            rationale=result.get("rationale", f"Remediate {finding.cve.cve_id}"),
        )

        logger.debug(
            "LLM codegen change generated",
            strategy=strategy.value,
            package=dep.fully_qualified_name,
            change_type=change.change_type,
        )
        return [change]

    def _call_codegen_llm(self, prompt: str) -> dict[str, Any]:
        """Call Ollama for code generation and return the parsed JSON response."""
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.05,   # very low — code must be precise
                "num_predict": 1024,
            },
        }

        logger.debug("Calling Ollama LLM for code generation", model=settings.llm_model)

        response = self._http.post(url, json=payload)
        response.raise_for_status()

        raw_text = response.json().get("response", "")
        parsed = json.loads(raw_text)

        if "new_content" not in parsed:
            raise ValueError("LLM codegen response missing 'new_content' field")

        return parsed

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_detailed_steps(
        llm_result: dict[str, Any],
        dep: Any,
        used_fallback: bool,
    ) -> list[str]:
        """
        Convert LLM output into the RemediationPlan.detailed_steps list.

        Combines the upgrade approach with any config changes into an
        ordered step list that developers can follow without reading
        the raw LLM output.
        """
        steps: list[str] = []

        if used_fallback:
            steps.append(f"LLM unavailable — rule-based strategy: {llm_result['upgrade_approach']}")
        else:
            steps.append(llm_result["upgrade_approach"])

        if dep.fixed_version:
            steps.append(
                f"Update {dep.fully_qualified_name} to version {dep.fixed_version} "
                f"in {dep.manifest_file}"
            )

        for cfg_change in llm_result.get("config_changes", []):
            steps.append(f"Config change: {cfg_change}")

        return steps

    @staticmethod
    def _requires_approval(
        report: ImpactReport,
        strategy: RemediationStrategy,
    ) -> bool:
        """
        Determine if this plan needs human approval before PR creation.

        Business rules:
          - CRITICAL priority → always requires approval
          - code_refactor or manual_review strategies → always requires approval
          - Everything else → auto-approvable (low-risk version bumps)
        """
        if report.priority == Priority.CRITICAL:
            return True
        if strategy in (RemediationStrategy.CODE_REFACTOR, RemediationStrategy.MANUAL_REVIEW):
            return True
        return False
