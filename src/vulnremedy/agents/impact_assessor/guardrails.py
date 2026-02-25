"""
Guardrails for the Impact Assessor Agent.

Provides three layers of protection:
  1. Response validation  — ensure LLM output conforms to expected schema/types.
  2. Input sanitization  — strip prompt-injection attempts from RAG-retrieved context.
  3. Circuit breaker     — stop hammering a failing LLM endpoint.
  4. Rate limiting       — enforce a minimum inter-call delay.
"""

from __future__ import annotations

import time
from typing import Any

from vulnremedy.models.impact import Priority
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger

# ─── Constants ────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS: list[str] = [
    "ignore previous",
    "ignore above",
    "disregard",
    "new instructions",
    "system prompt",
]


class ImpactAssessorGuardrails:
    """
    Centralises all input/output safety checks for the Impact Assessor Agent.

    Designed as a stateful helper — a single instance per agent keeps the
    circuit-breaker and rate-limiter state in one place.

    Usage:
        guardrails = ImpactAssessorGuardrails()

        # Before calling the LLM
        allowed, reason = guardrails.should_allow_llm_call()
        if not allowed:
            raise RuntimeError(reason)
        guardrails.apply_rate_limit()

        # Sanitise RAG context before injecting into prompt
        safe_context = guardrails.sanitize_exploit_context(raw_chunks)

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
          - ``is_exploitable``: coerced to bool if string ("true"/"false").
          - ``reasoning``: truncated to ``settings.guardrails_max_reasoning_chars`` characters.
          - ``business_impact``: truncated to ``settings.guardrails_max_business_impact_chars`` characters.
          - ``recommended_priority``: uppercased and defaulted to MEDIUM if unknown.

        Args:
            response: Raw dict parsed from the LLM JSON output.

        Returns:
            Sanitised dict with the same keys.
        """
        result = dict(response)

        # --- is_exploitable ---
        raw_exploitable = result.get("is_exploitable")
        if isinstance(raw_exploitable, str):
            coerced = raw_exploitable.strip().lower() == "true"
            logger.warning(
                "LLM returned is_exploitable as string — coercing to bool",
                original=raw_exploitable,
                coerced=coerced,
            )
            result["is_exploitable"] = coerced
        elif not isinstance(raw_exploitable, bool):
            logger.warning(
                "LLM returned unexpected type for is_exploitable — defaulting to False",
                type=type(raw_exploitable).__name__,
            )
            result["is_exploitable"] = False

        # --- reasoning ---
        reasoning = result.get("reasoning", "")
        if len(reasoning) > settings.guardrails_max_reasoning_chars:
            logger.warning(
                "LLM reasoning truncated",
                original_length=len(reasoning),
                limit=settings.guardrails_max_reasoning_chars,
            )
            result["reasoning"] = reasoning[:settings.guardrails_max_reasoning_chars]

        # --- business_impact ---
        business_impact = result.get("business_impact", "")
        if len(business_impact) > settings.guardrails_max_business_impact_chars:
            logger.warning(
                "LLM business_impact truncated",
                original_length=len(business_impact),
                limit=settings.guardrails_max_business_impact_chars,
            )
            result["business_impact"] = business_impact[:settings.guardrails_max_business_impact_chars]

        # --- recommended_priority ---
        raw_priority = str(result.get("recommended_priority", "")).upper()
        if raw_priority not in Priority.__members__:
            logger.warning(
                "LLM returned unknown priority — defaulting to MEDIUM",
                original=result.get("recommended_priority"),
            )
            raw_priority = "MEDIUM"
        result["recommended_priority"] = raw_priority

        return result

    # ─── 2. Input sanitisation ────────────────────────────────────────────────

    def sanitize_exploit_context(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format RAG chunks into a safe context block for the LLM prompt.

        For each chunk:
          - Checks for known prompt-injection patterns (case-insensitive).
          - If detected, replaces the chunk text with ``"[Content sanitized]"``.
          - Truncates any chunk text exceeding ``settings.guardrails_max_chunk_chars`` characters.

        Args:
            chunks: Raw RAG result dicts (each with ``text`` and ``metadata`` keys).

        Returns:
            Formatted, sanitised context string for inclusion in the prompt.
        """
        if not chunks:
            return "No exploit details available in the CVE knowledge base."

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
                    "Prompt injection pattern detected in RAG chunk — sanitizing",
                    chunk_index=i,
                    cve_id=cve_id,
                )
                text = "[Content sanitized]"

            # Length cap
            elif len(text) > settings.guardrails_max_chunk_chars:
                logger.warning(
                    "RAG chunk truncated — exceeded max length",
                    chunk_index=i,
                    original_length=len(text),
                    limit=settings.guardrails_max_chunk_chars,
                )
                text = text[:settings.guardrails_max_chunk_chars]

            parts.append(
                f"[Source {i} — {cve_id} / {chunk_type} (relevance: {score:.2f})]\n"
                f"{text}"
            )

        return "\n\n".join(parts)

    # ─── 3. Circuit breaker ───────────────────────────────────────────────────

    def should_allow_llm_call(self) -> tuple[bool, str]:
        """
        Check whether the circuit breaker permits another LLM call.

        The circuit opens (blocks calls) after ``settings.guardrails_circuit_breaker_max_failures``
        consecutive failures.  Call ``record_success()`` after a successful
        LLM invocation to reset the counter.

        Returns:
            ``(True, "")`` when the circuit is closed (calls allowed).
            ``(False, reason)`` when the circuit is open (calls blocked).
        """
        if self._failure_count >= settings.guardrails_circuit_breaker_max_failures:
            reason = (
                f"Circuit breaker open — {self._failure_count} consecutive LLM "
                f"failures (threshold: {settings.guardrails_circuit_breaker_max_failures}). "
                "Using deterministic fallback."
            )
            logger.warning("Circuit breaker blocked LLM call", failure_count=self._failure_count)
            return False, reason
        return True, ""

    def record_failure(self) -> None:
        """Increment the consecutive-failure counter."""
        self._failure_count += 1
        logger.debug("Circuit breaker failure recorded", failure_count=self._failure_count)

    def record_success(self) -> None:
        """Reset the consecutive-failure counter after a successful LLM call."""
        if self._failure_count > 0:
            logger.debug(
                "Circuit breaker reset after successful LLM call",
                previous_count=self._failure_count,
            )
        self._failure_count = 0

    # ─── 4. Rate limiter ─────────────────────────────────────────────────────

    def apply_rate_limit(self) -> None:
        """
        Block until the minimum inter-call interval has elapsed.

        Enforces a minimum of ``settings.guardrails_rate_limit_interval_seconds`` between consecutive
        LLM calls to avoid overloading the Ollama endpoint.
        """
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < settings.guardrails_rate_limit_interval_seconds:
            sleep_for = settings.guardrails_rate_limit_interval_seconds - elapsed
            logger.debug("Rate limit — sleeping before LLM call", sleep_seconds=round(sleep_for, 3))
            time.sleep(sleep_for)
        self._last_call_time = time.monotonic()
