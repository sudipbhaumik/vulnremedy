"""
PR Creator Agent — creates GitHub pull requests for approved remediation plans.

Pipeline per RemediationPlan (APPROVED status only):
  1. Normalise repository string to "owner/repo" GitHub API format.
  2. Derive fix branch name: fix/cve-{cve_id}-{timestamp}.
  3. [Live only] Create the fix branch from the scanned source branch.
  4. [Live only] Apply each CodeChange via GitHub Contents API:
       - Fetch current file content.
       - Search-and-replace original_content → new_content (or full replace).
       - Commit the updated file to the fix branch.
  5. Build LLM PR description prompt from externalized template.
  6. Check circuit breaker + apply rate limit via guardrails.
  7. Call Ollama LLM → JSON {title, body}.
  8. Validate response through guardrails (truncation, fallback on empty).
  9. [Live only] Open pull request via GitHub API.
 10. Update plan.pr_url and plan.status = PR_CREATED.

dry_run=True (default):
  - Skips all GitHub API calls (no branch creation, no file commits, no PR).
  - Still runs the LLM description pipeline (or fallback if circuit open).
  - Sets plan.pr_url to a synthetic URL for tracing.
  - Useful for integration testing without a real GitHub token.

Filtering:
  Only RemediationStatus.APPROVED plans are processed.
  Plans in other states (DRAFT, PENDING_APPROVAL, etc.) are skipped.

Output: dict with:
  - success:      bool
  - prs_created:  list[dict]  — one entry per successfully processed plan
  - skipped:      list[str]   — plan IDs skipped (non-APPROVED status)
  - errors:       list[str]   — plan IDs that failed with error descriptions
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import httpx

from vulnremedy.agents.pr_creator.guardrails import (
    PRCreatorGuardrails,
    _FALLBACK_BODY_TPL,
    _FALLBACK_TITLE_TPL,
)
from vulnremedy.models.remediation import CodeChange, RemediationPlan, RemediationStatus
from vulnremedy.tools.github.tool import GitHubTool
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _normalize_repo(repository: str) -> str:
    """
    Extract the 'owner/repo' slug from a repository field.

    Handles variants:
      - "github.com/owner/repo"
      - "https://github.com/owner/repo"
      - "owner/repo"  (already normalised)
    """
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if repository.startswith(prefix):
            return repository[len(prefix):]
    return repository


def _make_branch_name(cve_id: str) -> str:
    """
    Generate a deterministic-but-timestamped fix branch name.

    Format: fix/cve-{cve_id_lower}-{YYYYMMDDHHmmss}

    Example: fix/cve-2021-44228-20240224143022
    """
    safe_id = re.sub(r"[^a-z0-9\-]", "-", cve_id.lower())
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"fix/{safe_id}-{timestamp}"


# ─── Agent ────────────────────────────────────────────────────────────────────


class PRCreatorAgent:
    """
    Creates GitHub pull requests for APPROVED RemediationPlans.

    Usage:
        agent = PRCreatorAgent()
        result = agent.create_prs(plans, dry_run=True)

        for pr in result["prs_created"]:
            print(pr["pr_url"])
    """

    def __init__(self) -> None:
        self.github = GitHubTool()
        self.guardrails = PRCreatorGuardrails()
        self.prompt_template = settings.load_prompt("pr_creator_description.txt")
        self._http = httpx.Client(timeout=60.0)

        logger.info(
            "PR Creator Agent initialized",
            llm_model=settings.llm_model,
            ollama_url=settings.ollama_base_url,
        )

    # ─── Public entry point ───────────────────────────────────────────────────

    def create_prs(
        self,
        plans: list[RemediationPlan],
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """
        Create GitHub PRs for all APPROVED RemediationPlans.

        Args:
            plans:    list[RemediationPlan] — typically from RemediationPlannerAgent.
            dry_run:  If True, no GitHub API calls are made (safe for testing).

        Returns:
            Dictionary with:
                - success:     bool
                - prs_created: list[dict]  (one per successfully processed plan)
                - skipped:     list[str]   (plan IDs skipped — non-APPROVED)
                - errors:      list[str]   (plan IDs that raised exceptions)
        """
        logger.info(
            "PR Creator starting",
            total_plans=len(plans),
            dry_run=dry_run,
        )

        prs_created: list[dict[str, Any]] = []
        skipped: list[str] = []
        errors: list[str] = []

        for plan in plans:
            plan_id = str(plan.id)
            dep = plan.finding.affected_dependency
            cve_id = plan.finding.cve.cve_id

            # Filter — only process APPROVED plans
            if plan.status != RemediationStatus.APPROVED:
                logger.debug(
                    "Skipping non-APPROVED plan",
                    plan_id=plan_id,
                    status=plan.status.value,
                    cve_id=cve_id,
                )
                skipped.append(plan_id)
                continue

            try:
                pr_info = self._create_pr_for_plan(plan, dry_run=dry_run)
                prs_created.append(pr_info)
                logger.info(
                    "PR created",
                    plan_id=plan_id,
                    cve_id=cve_id,
                    package=dep.fully_qualified_name,
                    pr_url=pr_info["pr_url"],
                    dry_run=dry_run,
                )
            except Exception as exc:
                error_msg = (
                    f"Plan {plan_id} ({cve_id} / {dep.fully_qualified_name}): {exc}"
                )
                logger.error(
                    "Error creating PR for plan",
                    plan_id=plan_id,
                    cve_id=cve_id,
                    error=str(exc),
                )
                errors.append(error_msg)

        logger.info(
            "PR Creator complete",
            total=len(plans),
            prs_created=len(prs_created),
            skipped=len(skipped),
            errors=len(errors),
            dry_run=dry_run,
        )

        return {
            "success": True,
            "prs_created": prs_created,
            "skipped": skipped,
            "errors": errors,
        }

    # ─── Per-plan pipeline ────────────────────────────────────────────────────

    def _create_pr_for_plan(
        self,
        plan: RemediationPlan,
        dry_run: bool,
    ) -> dict[str, Any]:
        """
        Full pipeline for one APPROVED plan:
        branch → file changes → LLM description → PR.
        """
        finding = plan.finding
        dep = finding.affected_dependency
        cve_id = finding.cve.cve_id

        repo = _normalize_repo(finding.repository)
        source_branch = finding.branch
        fix_branch = _make_branch_name(cve_id)

        # Step 1–2: LLM PR description (runs even in dry_run)
        prompt = self._build_description_prompt(plan)
        description, used_fallback = self._call_llm_with_fallback(plan, prompt)

        if not dry_run:
            # Step 3: Create the fix branch
            branch_result = self.github.create_branch(
                repo=repo,
                branch_name=fix_branch,
                from_branch=source_branch,
            )
            if not branch_result["success"]:
                raise RuntimeError(
                    f"Could not create branch '{fix_branch}': {branch_result['error']}"
                )

            # Step 4: Apply all code changes
            for change in plan.code_changes:
                self._apply_code_change(repo=repo, change=change, branch=fix_branch)

            # Step 5: Open the PR
            pr_result = self.github.create_pull_request(
                repo=repo,
                title=description["title"],
                body=description["body"],
                head=fix_branch,
                base=source_branch,
            )
            if not pr_result["success"]:
                raise RuntimeError(
                    f"Could not open PR: {pr_result['error']}"
                )
            pr_url: str = str(pr_result["url"])

        else:
            # dry_run: synthesise a predictable PR URL
            pr_url = f"https://github.com/{repo}/pull/DRY_RUN"

        # Step 6: Update plan state
        plan.mark_pr_created(pr_url)

        return {
            "plan_id": str(plan.id),
            "cve_id": cve_id,
            "package": dep.fully_qualified_name,
            "repository": repo,
            "fix_branch": fix_branch,
            "pr_url": pr_url,
            "pr_title": description["title"],
            "code_changes_applied": len(plan.code_changes),
            "dry_run": dry_run,
            "used_llm_fallback": used_fallback,
        }

    # ─── Code change application ─────────────────────────────────────────────

    def _apply_code_change(
        self,
        repo: str,
        change: CodeChange,
        branch: str,
    ) -> None:
        """
        Apply a single CodeChange to the fix branch via GitHub Contents API.

        Strategy:
          - If original_content is non-empty: search-and-replace in the fetched file.
          - If original_content is empty/None: use new_content as the full new file.
        """
        # Fetch current file from the new branch
        fetch_result = self.github.fetch_file(
            repo=repo, file_path=change.file_path, branch=branch
        )
        if not fetch_result["success"]:
            raise RuntimeError(
                f"Could not fetch '{change.file_path}' on branch '{branch}': "
                f"{fetch_result['error']}"
            )

        current_content: str = fetch_result["content"]  # type: ignore[assignment]

        if change.original_content:
            # Precise search-and-replace
            if change.original_content not in current_content:
                raise RuntimeError(
                    f"original_content not found in '{change.file_path}'. "
                    f"First 100 chars looked for: "
                    f"{change.original_content[:100]!r}"
                )
            new_content = current_content.replace(
                change.original_content, change.new_content, 1
            )
        else:
            # Full file replacement
            new_content = change.new_content

        commit_message = f"fix(security): {change.rationale}"

        update_result = self.github.update_file(
            repo=repo,
            file_path=change.file_path,
            content=new_content,
            commit_message=commit_message,
            branch=branch,
        )
        if not update_result["success"]:
            raise RuntimeError(
                f"Could not commit '{change.file_path}': {update_result['error']}"
            )

        logger.debug(
            "Code change applied",
            file_path=change.file_path,
            change_type=change.change_type,
            branch=branch,
        )

    # ─── Prompt building ─────────────────────────────────────────────────────

    def _build_description_prompt(self, plan: RemediationPlan) -> str:
        """Fill the externalized PR description template."""
        finding = plan.finding
        dep = finding.affected_dependency
        cve = finding.cve

        detailed_steps = self.guardrails.sanitize_plan_text(
            "\n".join(f"- {s}" for s in plan.detailed_steps) or "See summary.",
            field_name="detailed_steps",
        )
        breaking_changes = self.guardrails.sanitize_plan_text(
            "\n".join(f"- {b}" for b in plan.breaking_changes) or "None",
            field_name="breaking_changes",
        )
        testing_recs = self.guardrails.sanitize_plan_text(
            "\n".join(f"- {t}" for t in plan.testing_recommendations) or "Run the test suite.",
            field_name="testing_recommendations",
        )

        return self.prompt_template.format(
            cve_id=cve.cve_id,
            severity=finding.severity.value.upper(),
            package=dep.fully_qualified_name,
            current_version=dep.current_version,
            fixed_version=dep.fixed_version or "latest safe version",
            ecosystem=dep.ecosystem.value,
            repository=finding.repository,
            strategy=plan.strategy.value,
            summary=plan.summary,
            detailed_steps=detailed_steps,
            breaking_changes=breaking_changes,
            testing_recommendations=testing_recs,
        )

    # ─── Fallback description ─────────────────────────────────────────────────

    def _fallback_description(self, plan: RemediationPlan) -> dict[str, str]:
        """Build a deterministic PR title + body when the LLM is unavailable."""
        finding = plan.finding
        dep = finding.affected_dependency
        cve = finding.cve

        title = _FALLBACK_TITLE_TPL.format(
            package=dep.fully_qualified_name,
            fixed_version=dep.fixed_version or "latest",
            cve_id=cve.cve_id,
        )
        body = _FALLBACK_BODY_TPL.format(
            cve_id=cve.cve_id,
            severity=finding.severity.value.upper(),
            package=dep.fully_qualified_name,
            current_version=dep.current_version,
            fixed_version=dep.fixed_version or "latest",
            manifest_file=dep.manifest_file,
            testing_recommendations="\n".join(
                f"- {t}" for t in plan.testing_recommendations
            ) or "- Run the full test suite.",
            breaking_changes="\n".join(
                f"- {b}" for b in plan.breaking_changes
            ) or "None",
        )
        return {"title": title, "body": body}

    # ─── LLM integration ─────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, fallback_title: str, fallback_body: str) -> dict[str, Any]:
        """Call Ollama and return the validated {title, body} dict."""
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 768,
            },
        }

        logger.debug("Calling Ollama LLM for PR description", model=settings.llm_model)

        response = self._http.post(url, json=payload)
        response.raise_for_status()

        raw_text = response.json().get("response", "")
        parsed = json.loads(raw_text)

        return self.guardrails.validate_llm_response(
            parsed,
            fallback_title=fallback_title,
            fallback_body=fallback_body,
        )

    def _call_llm_with_fallback(
        self,
        plan: RemediationPlan,
        prompt: str,
    ) -> tuple[dict[str, str], bool]:
        """
        Attempt LLM call; return rule-based fallback on failure or open circuit.

        Returns:
            (description_dict, used_fallback) — used_fallback=True means
            the rule-based fallback was used instead of the LLM.
        """
        fallback = self._fallback_description(plan)

        allowed, reason = self.guardrails.should_allow_llm_call()
        if not allowed:
            logger.warning(
                "PR Creator circuit breaker prevented LLM call — using fallback",
                cve_id=plan.finding.cve.cve_id,
                reason=reason,
            )
            return fallback, True

        self.guardrails.apply_rate_limit()

        try:
            result = self._call_llm(
                prompt,
                fallback_title=fallback["title"],
                fallback_body=fallback["body"],
            )
            self.guardrails.record_success()
            return result, False
        except Exception as exc:
            self.guardrails.record_failure()
            logger.warning(
                "PR Creator LLM call failed — using fallback description",
                cve_id=plan.finding.cve.cve_id,
                package=plan.finding.affected_dependency.fully_qualified_name,
                error=str(exc),
            )
            return fallback, True
