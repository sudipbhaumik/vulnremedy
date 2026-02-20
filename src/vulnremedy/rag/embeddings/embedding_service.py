"""
Embedding Service — Convert text chunks to vector embeddings.

Uses Ollama's nomic-embed-text model to generate 768-dimensional
embeddings for text chunks.

Architectural notes:
    - Batching: Embeds multiple chunks per API call for efficiency
    - Caching: Embeddings cached to disk to avoid re-computation
    - Retry logic: Handles transient Ollama failures gracefully
    - Progress tracking: Logs progress for long-running operations
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from vulnremedy.rag.chunking.cve_chunker import CVEChunk
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


class EmbeddedChunk:
    """
    A chunk with its embedding vector.
    
    Contains the original text, metadata, and the computed embedding.
    """
    
    def __init__(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any]
    ):
        self.text = text
        self.embedding = embedding
        self.metadata = metadata
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata
        }


class EmbeddingService:
    """
    Generates embeddings for text chunks using Ollama.
    
    Usage:
        service = EmbeddingService()
        embedded_chunks = service.embed_chunks(chunks)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize embedding service.
        
        Args:
            cache_dir: Where to cache embeddings (default from config)
            batch_size: Number of chunks to embed per batch (default from config)
        """
        self.ollama_base_url = settings.ollama_base_url
        self.embedding_model = settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        
        self.cache_dir = cache_dir or Path(settings.embedding_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP client with longer timeout for embedding operations
        self.client = httpx.Client(timeout=120.0)
        
        # Cache for this session
        self._cache: dict[str, list[float]] = {}
        self._load_disk_cache()
        
        logger.info(
            "Embedding Service initialized",
            model=self.embedding_model,
            batch_size=self.batch_size,
            cache_dir=str(self.cache_dir),
            ollama_url=self.ollama_base_url
        )
    
    def embed_chunks(
        self,
        chunks: list[CVEChunk],
        show_progress: bool = True
    ) -> list[EmbeddedChunk]:
        """
        Embed a list of chunks.
        
        Processes chunks in batches for efficiency.
        Uses cache to avoid re-embedding same text.
        
        Args:
            chunks: List of CVEChunk objects to embed
            show_progress: Whether to log progress
        
        Returns:
            List of EmbeddedChunk objects with embeddings
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return []
        
        logger.info(f"Starting embedding of {len(chunks)} chunks")
        
        embedded_chunks = []
        total = len(chunks)
        processed = 0
        cache_hits = 0
        cache_misses = 0
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            if show_progress:
                logger.info(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} chunks)"
                )
            
            # Check cache for each chunk in batch
            batch_to_embed = []
            batch_indices = []
            
            for idx, chunk in enumerate(batch):
                cache_key = self._get_cache_key(chunk.text)
                
                if cache_key in self._cache:
                    # Cache hit
                    embedding = self._cache[cache_key]
                    embedded_chunks.append(
                        EmbeddedChunk(
                            text=chunk.text,
                            embedding=embedding,
                            metadata=chunk.metadata
                        )
                    )
                    cache_hits += 1
                else:
                    # Cache miss - needs embedding
                    batch_to_embed.append(chunk)
                    batch_indices.append(idx)
                    cache_misses += 1
            
            # Embed chunks that weren't in cache
            if batch_to_embed:
                embeddings = self._embed_batch([c.text for c in batch_to_embed])
                
                for chunk, embedding in zip(batch_to_embed, embeddings):
                    # Add to cache
                    cache_key = self._get_cache_key(chunk.text)
                    self._cache[cache_key] = embedding
                    
                    embedded_chunks.append(
                        EmbeddedChunk(
                            text=chunk.text,
                            embedding=embedding,
                            metadata=chunk.metadata
                        )
                    )
            
            processed += len(batch)
            
            if show_progress and processed % 100 == 0:
                logger.info(
                    f"Progress: {processed}/{total} chunks embedded "
                    f"(cache hits: {cache_hits}, misses: {cache_misses})"
                )
        
        # Save updated cache to disk
        self._save_disk_cache()
        
        logger.info(
            f"Embedding complete. Total: {len(embedded_chunks)}, "
            f"Cache hits: {cache_hits}, Cache misses: {cache_misses}"
        )
        
        return embedded_chunks
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts using Ollama API.
        
        This method is decorated with @retry for resilience.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (one per text)
        """
        url = f"{self.ollama_base_url}/api/embeddings"
        
        embeddings = []
        
        # Ollama's embedding API processes one text at a time
        # We call it multiple times within one batch for efficiency
        for text in texts:
            # Truncate very long texts to avoid Ollama crashes
            # nomic-embed-text supports ~8000 tokens (~32000 chars)
            # We limit to 8000 chars to be safe
            max_length = settings.embedding_max_length
            if len(text) > max_length:
                logger.warning(
                    f"Text too long for embedding {max_length}, truncating to 8000 chars: "
                    f"{text[:100]}..."
                )
                text = text[:max_length] + "... [truncated]"

            # Debug logging
            logger.debug(f"Embedding text of length {len(text)} chars")
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            try:
                response = self.client.post(url, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama embedding failed for text (len={len(text)}): {text[:200]}...")
                raise
            
            data = response.json()
            embedding = data.get("embedding")
            
            if not embedding:
                raise ValueError(f"No embedding returned from Ollama for text: {text[:100]}")
            
            embeddings.append(embedding)
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for a text.
        
        Uses MD5 hash of text for compact, deterministic keys.
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_disk_cache(self) -> None:
        """
        Load embedding cache from disk.
        
        Cache format: {text_hash: embedding_vector}
        """
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        if not cache_file.exists():
            logger.info("No embedding cache found, starting fresh")
            return
        
        try:
            with open(cache_file, "r") as f:
                self._cache = json.load(f)
            
            logger.info(f"Loaded {len(self._cache)} embeddings from cache")
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            self._cache = {}
    
    def _save_disk_cache(self) -> None:
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cache, f)
            
            logger.debug(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear both in-memory and disk cache."""
        self._cache = {}
        cache_file = self.cache_dir / "embeddings_cache.json"
        
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Embedding cache cleared")
    
    def close(self) -> None:
        """Close HTTP client and save cache."""
        self._save_disk_cache()
        self.client.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, *args):
        """Context manager support."""
        self.close()