"""
Reranker — Rerank top results using cross-encoder for precision.

Takes top N results from hybrid retrieval and reranks them
using a more sophisticated relevance scoring mechanism.

Architectural notes:
    - Applied only to top candidates (expensive, can't run on full corpus)
    - Uses LLM-based scoring for this portfolio project
    - Production systems use dedicated cross-encoder models
    - Configurable threshold for filtering low-relevance results
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


class Reranker:
    """
    Reranks retrieval results using LLM-based relevance scoring.
    
    Usage:
        reranker = Reranker()
        top_results = reranker.rerank(
            query="How to fix Log4Shell?",
            candidates=hybrid_results,
            top_k=5
        )
    """
    
    def __init__(
        self,
        relevance_threshold: Optional[float] = None
    ):
        """
        Initialize reranker.
        
        Args:
            relevance_threshold: Minimum relevance score to keep (default from config)
        """
        self.ollama_base_url = settings.ollama_base_url
        self.llm_model = settings.llm_model
        self.relevance_threshold = relevance_threshold or settings.reranker_relevance_threshold
        
        self.client = httpx.Client(timeout=60.0)
        
        logger.info(
            "Reranker initialized",
            model=self.llm_model,
            relevance_threshold=self.relevance_threshold
        )
    
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Rerank candidate results by relevance to query.
        
        Args:
            query: Original search query
            candidates: Results from hybrid retrieval
            top_k: Number of top results to return (default from config)
        
        Returns:
            Reranked results with relevance scores
        """
        if not candidates:
            return []
        
        top_k = top_k or settings.retrieval_top_k
        
        logger.info(
            f"Reranking {len(candidates)} candidates for query: '{query[:50]}...'"
        )
        
        # Score each candidate
        scored_candidates = []
        
        for idx, candidate in enumerate(candidates):
            relevance_score = self._score_relevance(
                query=query,
                text=candidate["text"],
                metadata=candidate.get("metadata", {})
            )
            
            scored_candidates.append({
                **candidate,
                "relevance_score": relevance_score,
                "original_rank": idx
            })
            
            # Log progress for large sets
            if (idx + 1) % 10 == 0:
                logger.debug(f"Reranking progress: {idx + 1}/{len(candidates)}")
        
        # Filter by threshold
        filtered = [
            c for c in scored_candidates 
            if c["relevance_score"] >= self.relevance_threshold
        ]
        
        logger.info(
            f"Filtered {len(scored_candidates)} candidates to {len(filtered)} "
            f"above threshold {self.relevance_threshold}"
        )
        
        # Sort by relevance score
        filtered.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Take top_k
        top_results = filtered[:top_k]
        
        logger.info(
            f"Reranking complete: returned top {len(top_results)} results"
        )
        
        return top_results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    def _score_relevance(
        self,
        query: str,
        text: str,
        metadata: dict[str, Any]
    ) -> float:
        """
        Score relevance of text to query using LLM.
        
        Uses a zero-shot prompt asking the LLM to rate relevance 0-10.
        
        Args:
            query: Search query
            text: Candidate text to score
            metadata: Chunk metadata for context
        
        Returns:
            Relevance score normalized to 0.0-1.0
        """
        # Build prompt
        prompt = self._build_relevance_prompt(query, text, metadata)
        
        # Call Ollama
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Deterministic scoring
                "num_predict": 10    # Short response expected
            }
        }
        
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        response_text = data.get("response", "").strip()
        
        # Parse score from response
        score = self._parse_score(response_text)
        
        return score
    
    def _build_relevance_prompt(
        self,
        query: str,
        text: str,
        metadata: dict[str, Any]
    ) -> str:
        """
        Build prompt for LLM-based relevance scoring.
        
        Asks LLM to rate how well the text answers the query.
        """
        # Extract key metadata
        cve_id = metadata.get("cve_id", "unknown")
        chunk_type = metadata.get("chunk_type", "unknown")
        severity = metadata.get("severity", "unknown")
        
        prompt = f"""Rate how relevant this CVE information is to the query on a scale of 0-10.

Query: {query}

CVE Information:
- CVE ID: {cve_id}
- Type: {chunk_type}
- Severity: {severity}

Text:
{text[:500]}...

Respond with ONLY a number from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = perfectly answers the query

Score:"""
        
        return prompt
    
    def _parse_score(self, response_text: str) -> float:
        """
        Parse relevance score from LLM response.
        
        Handles various response formats:
        - "8" → 0.8
        - "8/10" → 0.8
        - "Score: 8" → 0.8
        - "8.5" → 0.85
        
        Returns:
            Score normalized to 0.0-1.0 range
        """
        import re
        
        # Try to find a number in the response
        match = re.search(r'(\d+\.?\d*)', response_text)
        
        if match:
            score = float(match.group(1))
            
            # Normalize to 0-1 if it's in 0-10 range
            if score > 1.0:
                score = score / 10.0
            
            # Clamp to valid range
            score = max(0.0, min(1.0, score))
            
            return score
        
        # Fallback: couldn't parse, return neutral score
        logger.warning(
            f"Failed to parse score from response: '{response_text}', using 0.5"
        )
        return 0.5
    
    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, *args):
        """Context manager support."""
        self.close()