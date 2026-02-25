"""
Guardrails for the Remediation Planner Agent.

Provides the same four protection layers as the Impact Assessor guardrails,
tuned for the Remediation Planner's output schema:
  1. Response validation  — validate strategy enum, truncate text fields, cap list lengths.
  2. Input sanitisation   — strip prompt-injection attempts from RAG migration guides.
  3. Circuit breaker      — stop hammering a failing LLM endpoint.
  4. Rate limiting        — enforce a minimum inter-call delay.
"""

from __future__ import annotations

import time
from typing import Any

from vulnremedy.models.remediation import RemediationStrategy
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger

# ─── Constants ────────────────────────────────────────────────────────────────

_VALID_STRATEGIES: set[str] = {s.value for s in RemediationStrategy}

_INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "ignore above",
    "disregard",
    "new instructions",
    "system prompt",
]

_LIST_FIELDS = ("breaking_changes", "config_changes", "testing_plan", "rollback_plan")
_TEXT_FIELDS = ("upgrade_approach",)


class RemediationGuardrails:
    """
    Centralises all input/output safety checks for the Remediation Planner Agent.

    Usage:
        guardrails = RemediationGuardrails()

        # Before calling the LLM
        allowed, reason = guardrails.should_allow_llm_call()
        if not allowed:
            raise RuntimeError(reason)
        guardrails.apply_rate_limit()

        # Sanitise RAG migration guides before injecting into prompt
        safe_context = guardrails.sanitize_migration_context(raw_chunks)

        # After receiving the LLM response
        clean = guardrails.validate_llm_response(raw_response)
        guardrails.record_success()   # reset circuit breaker on success
    """

    def __init__(self) -> None:
        self._failure_count: int = 0
        self._last_call_time: float = 0.0

    # ─── 1. Response validation ───────────────────────────────────────────────

    def validate_llm_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and sanitise a parsed LLM JSON response.

        Corrections applied (with warnings logged):
          - ``strategy``: validated against RemediationStrategy enum values;
            defaults to ``manual_review`` if unknown.
          - ``upgrade_approach``: truncated to ``settings.remediation_max_field_chars``.
          - List fields (``breaking_changes``, ``config_changes``, ``testing_plan``,
            ``rollback_plan``): non-list values replaced with empty list; items
            capped at ``settings.remediation_max_list_items``.

        Args:
            response: Raw dict parsed from the LLM JSON output.

        Returns:
            Sanitised dict with the same keys.
        """
        result = dict(response)

        # --- strategy ---
        raw_strategy = str(result.get("strategy", "")).lower().strip()
        if raw_strategy not in _VALID_STRATEGIES:
            logger.warning(
                "LLM returned unknown remediation strategy — defaulting to manual_review",
                original=result.get("strategy"),
            )
            raw_strategy = RemediationStrategy.MANUAL_REVIEW.value
        result["strategy"] = raw_strategy

        # --- text fields ---
        for field in _TEXT_FIELDS:
            value = result.get(field, "")
            if not isinstance(value, str):
                logger.warning(
                    "LLM returned non-string for text field — coercing to str",
                    field=field,
                    type=type(value).__name__,
                )
                value = str(value)
            if len(value) > settings.remediation_max_field_chars:
                logger.warning(
                    "LLM text field truncated",
                    field=field,
                    original_length=len(value),
                    limit=settings.remediation_max_field_chars,
                )
                value = value[:settings.remediation_max_field_chars]
            result[field] = value

        # --- list fields ---
        for field in _LIST_FIELDS:
            value = result.get(field, [])
            if not isinstance(value, list):
                logger.warning(
                    "LLM returned non-list for list field — replacing with empty list",
                    field=field,
                    type=type(value).__name__,
                )
                value = []
            # Ensure all items are strings
            cleaned: list[str] = [str(item) for item in value]
            # Cap length
            if len(cleaned) > settings.remediation_max_list_items:
                logger.warning(
                    "LLM list field capped",
                    field=field,
                    original_count=len(cleaned),
                    limit=settings.remediation_max_list_items,
                )
                cleaned = cleaned[:settings.remediation_max_list_items]
            result[field] = cleaned

        return result

    # ─── 2. Input sanitisation ────────────────────────────────────────────────

    def sanitize_migration_context(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format RAG migration guide chunks into a safe context block.

        For each chunk:
          - Checks for known prompt-injection patterns (case-insensitive).
          - If detected, replaces the chunk text with ``"[Content sanitized]"``.
          - Truncates any chunk text exceeding ``settings.remediation_max_field_chars`` characters.

        Args:
            chunks: Raw RAG result dicts (each with ``text`` and ``metadata`` keys).

        Returns:
            Formatted, sanitised context string for inclusion in the prompt.
        """
        if not chunks:
            return "No migration guides available in the knowledge base."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            cve_id = chunk.get("metadata", {}).get("cve_id", "unknown")
            chunk_type = chunk.get("metadata", {}).get("chunk_type", "info")
            score = chunk.get("blended_score", 0.0)
            text: str = chunk.get("text", "")

            # Prompt-injection detection
            lower_text = text.lower()
            injected = any(pattern in lower_text for pattern in _INJECTION_PATTERNS)
            if injected:
                logger.warning(
                    "Prompt injection pattern detected in migration guide chunk — sanitizing",
                    chunk_index=i,
                    cve_id=cve_id,
                )
                text = "[Content sanitized]"
            elif len(text) > settings.remediation_max_field_chars:
                logger.warning(
                    "Migration guide chunk truncated — exceeded max length",
                    chunk_index=i,
                    original_length=len(text),
                    limit=settings.remediation_max_field_chars,
                )
                text = text[:settings.remediation_max_field_chars]

            parts.append(
                f"[Guide {i} — {cve_id} / {chunk_type} (relevance: {score:.2f})]\n"
                f"{text}"
            )

        return "\n\n".join(parts)

    # ─── 3. Circuit breaker ───────────────────────────────────────────────────

    def should_allow_llm_call(self) -> tuple[bool, str]:
        """
        Check whether the circuit breaker permits another LLM call.

        The circuit opens after ``settings.remediation_circuit_breaker_max_failures``
        consecutive failures. Call ``record_success()`` to reset.

        Returns:
            ``(True, "")`` when the circuit is closed (calls allowed).
            ``(False, reason)`` when the circuit is open (calls blocked).
        """
        if self._failure_count >= settings.remediation_circuit_breaker_max_failures:
            reason = (
                f"Circuit breaker open — {self._failure_count} consecutive LLM "
                f"failures (threshold: {settings.remediation_circuit_breaker_max_failures}). "
                "Falling back to rule-based strategy selection."
            )
            logger.warning(
                "Remediation circuit breaker blocked LLM call",
                failure_count=self._failure_count,
            )
            return False, reason
        return True, ""

    def record_failure(self) -> None:
        """Increment the consecutive-failure counter."""
        self._failure_count += 1
        logger.debug(
            "Remediation circuit breaker failure recorded",
            failure_count=self._failure_count,
        )

    def record_success(self) -> None:
        """Reset the consecutive-failure counter after a successful LLM call."""
        if self._failure_count > 0:
            logger.debug(
                "Remediation circuit breaker reset after successful LLM call",
                previous_count=self._failure_count,
            )
        self._failure_count = 0

    # ─── 4. Rate limiter ─────────────────────────────────────────────────────

    def apply_rate_limit(self) -> None:
        """
        Block until the minimum inter-call interval has elapsed.

        Enforces a minimum of ``settings.remediation_rate_limit_interval_seconds``
        between consecutive LLM calls to avoid overloading the Ollama endpoint.
        """
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < settings.remediation_rate_limit_interval_seconds:
            sleep_for = settings.remediation_rate_limit_interval_seconds - elapsed
            logger.debug(
                "Remediation rate limit — sleeping before LLM call",
                sleep_seconds=round(sleep_for, 3),
            )
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()
