"""
Guardrails for the PR Creator Agent.

Provides the same four protection layers used by the other agents,
tuned for the PR Creator's output schema (title + body):
  1. Response validation  — validate/truncate PR title and body.
  2. Input sanitisation   — strip prompt-injection from plan context injected into the prompt.
  3. Circuit breaker      — stop hammering a failing LLM endpoint.
  4. Rate limiting        — enforce a minimum inter-call delay.
"""

from __future__ import annotations

import time
from typing import Any

from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger

# ─── Constants ────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "ignore above",
    "disregard",
    "new instructions",
    "system prompt",
    "reveal",
]

_FALLBACK_TITLE_TPL = "fix(security): upgrade {package} to {fixed_version} ({cve_id})"
_FALLBACK_BODY_TPL = (
    "## Motivation\n\n"
    "Remediate {cve_id} ({severity}) in `{package}` by upgrading "
    "from `{current_version}` to `{fixed_version}`.\n\n"
    "## Changes\n\n"
    "- Updated `{manifest_file}` to pin `{package}` at `{fixed_version}`.\n\n"
    "## Testing\n\n"
    "{testing_recommendations}\n\n"
    "## Breaking Changes\n\n"
    "{breaking_changes}"
)


class PRCreatorGuardrails:
    """
    Centralises all input/output safety checks for the PR Creator Agent.

    Usage:
        guardrails = PRCreatorGuardrails()

        # Before calling the LLM
        allowed, reason = guardrails.should_allow_llm_call()
        guardrails.apply_rate_limit()

        # Sanitise plan fields before injecting into prompt
        safe_text = guardrails.sanitize_plan_text(raw_text)

        # After receiving the LLM response
        clean = guardrails.validate_llm_response(raw_response, fallback_title, fallback_body)
        guardrails.record_success()
    """

    def __init__(self) -> None:
        self._failure_count: int = 0
        self._last_call_time: float = 0.0

    # ─── 1. Response validation ───────────────────────────────────────────────

    def validate_llm_response(
        self,
        response: dict[str, Any],
        fallback_title: str,
        fallback_body: str,
    ) -> dict[str, Any]:
        """
        Validate and sanitise a parsed LLM JSON response.

        Corrections applied (with warnings logged):
          - ``title``: must be a non-empty string; truncated to
            ``settings.pr_creator_max_title_chars``; falls back to
            ``fallback_title`` if missing/empty.
          - ``body``: must be a non-empty string; truncated to
            ``settings.pr_creator_max_body_chars``; falls back to
            ``fallback_body`` if missing/empty.

        Args:
            response:       Raw dict parsed from LLM JSON output.
            fallback_title: Title to use if LLM returns nothing usable.
            fallback_body:  Body to use if LLM returns nothing usable.

        Returns:
            Sanitised dict with ``title`` and ``body`` keys.
        """
        result = dict(response)

        # --- title ---
        title = result.get("title", "")
        if not isinstance(title, str):
            logger.warning(
                "PR Creator: LLM returned non-string title — coercing",
                type=type(title).__name__,
            )
            title = str(title)
        title = title.strip()
        if not title:
            logger.warning("PR Creator: LLM returned empty title — using fallback")
            title = fallback_title
        if len(title) > settings.pr_creator_max_title_chars:
            logger.warning(
                "PR Creator: title truncated",
                original_length=len(title),
                limit=settings.pr_creator_max_title_chars,
            )
            title = title[: settings.pr_creator_max_title_chars]
        result["title"] = title

        # --- body ---
        body = result.get("body", "")
        if not isinstance(body, str):
            logger.warning(
                "PR Creator: LLM returned non-string body — coercing",
                type=type(body).__name__,
            )
            body = str(body)
        body = body.strip()
        if not body:
            logger.warning("PR Creator: LLM returned empty body — using fallback")
            body = fallback_body
        if len(body) > settings.pr_creator_max_body_chars:
            logger.warning(
                "PR Creator: body truncated",
                original_length=len(body),
                limit=settings.pr_creator_max_body_chars,
            )
            body = body[: settings.pr_creator_max_body_chars]
        result["body"] = body

        return result

    # ─── 2. Input sanitisation ────────────────────────────────────────────────

    def sanitize_plan_text(self, text: str, field_name: str = "plan_field") -> str:
        """
        Sanitise a plain-text field from the remediation plan before injecting
        it into the LLM prompt.

        - Checks for known prompt-injection patterns (case-insensitive).
        - If detected, returns ``"[Content sanitized]"``.
        - Truncates text exceeding ``settings.pr_creator_max_body_chars`` characters.

        Args:
            text:       Raw text from a RemediationPlan field.
            field_name: Name used in log messages for traceability.

        Returns:
            Safe text string.
        """
        lower = text.lower()
        if any(pattern in lower for pattern in _INJECTION_PATTERNS):
            logger.warning(
                "PR Creator: prompt injection pattern detected — sanitizing",
                field=field_name,
            )
            return "[Content sanitized]"

        limit = settings.pr_creator_max_body_chars
        if len(text) > limit:
            logger.warning(
                "PR Creator: plan field truncated",
                field=field_name,
                original_length=len(text),
                limit=limit,
            )
            return text[:limit]

        return text

    # ─── 3. Circuit breaker ───────────────────────────────────────────────────

    def should_allow_llm_call(self) -> tuple[bool, str]:
        """
        Check whether the circuit breaker permits another LLM call.

        Returns:
            ``(True, "")`` when the circuit is closed (calls allowed).
            ``(False, reason)`` when the circuit is open (calls blocked).
        """
        threshold = settings.pr_creator_circuit_breaker_max_failures
        if self._failure_count >= threshold:
            reason = (
                f"Circuit breaker open — {self._failure_count} consecutive LLM "
                f"failures (threshold: {threshold}). Using fallback PR description."
            )
            logger.warning(
                "PR Creator circuit breaker blocked LLM call",
                failure_count=self._failure_count,
            )
            return False, reason
        return True, ""

    def record_failure(self) -> None:
        """Increment the consecutive-failure counter."""
        self._failure_count += 1
        logger.debug(
            "PR Creator circuit breaker failure recorded",
            failure_count=self._failure_count,
        )

    def record_success(self) -> None:
        """Reset the consecutive-failure counter after a successful LLM call."""
        if self._failure_count > 0:
            logger.debug(
                "PR Creator circuit breaker reset after successful LLM call",
                previous_count=self._failure_count,
            )
        self._failure_count = 0

    # ─── 4. Rate limiter ─────────────────────────────────────────────────────

    def apply_rate_limit(self) -> None:
        """
        Block until the minimum inter-call interval has elapsed.

        Enforces a minimum of ``settings.pr_creator_rate_limit_interval_seconds``
        between consecutive LLM calls.
        """
        now = time.monotonic()
        elapsed = now - self._last_call_time
        interval = settings.pr_creator_rate_limit_interval_seconds
        if elapsed < interval:
            sleep_for = interval - elapsed
            logger.debug(
                "PR Creator rate limit — sleeping before LLM call",
                sleep_seconds=round(sleep_for, 3),
            )
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()
