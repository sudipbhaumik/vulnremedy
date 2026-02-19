"""
Vector Store — ChromaDB wrapper for storing and retrieving embeddings.

Handles all interactions with ChromaDB including storage,
retrieval, and metadata filtering.

Architectural notes:
    - Each collection represents a knowledge domain (cve_knowledge)
    - Metadata filters enable precise retrieval (severity, ecosystem)
    - Supports both vector similarity and metadata-based queries
    - Production-ready with error handling and connection management
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from vulnremedy.rag.embeddings.embedding_service import EmbeddedChunk
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


class VectorStore:
    """
    ChromaDB wrapper for vector storage and retrieval.
    
    Usage:
        store = VectorStore()
        store.add_chunks(embedded_chunks)
        results = store.search("Log4Shell vulnerability", top_k=5)
    """
    
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Where to store ChromaDB data (default from config)
            collection_name: Collection name (default: "cve_knowledge")
        """
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "CVE knowledge base for VulnRemedy"}
        )
        
        logger.info(
            "Vector Store initialized",
            collection=self.collection_name,
            persist_dir=self.persist_directory,
            documents=self.collection.count()
        )
    
    def add_chunks(
        self,
        chunks: list[EmbeddedChunk],
        batch_size: Optional[int] = None
    ) -> None:
        """
        Add embedded chunks to vector store.
        
        Processes in batches for efficiency.
        
        Args:
            chunks: List of EmbeddedChunk objects
            batch_size: Number of chunks to add per batch (default from config)
        """
        batch_size = batch_size or settings.retrieval_batch_size
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        total = len(chunks)
        added = 0
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            ids = [f"{chunk.metadata.get('cve_id', 'unknown')}_{i + idx}" 
                   for idx, chunk in enumerate(batch)]
            documents = [chunk.text for chunk in batch]
            embeddings = [chunk.embedding for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            added += len(batch)
            
            if added % settings.retrieval_batch_size == 0 and added > 0:
                logger.info(f"Progress: {added}/{total} chunks added")
        
        logger.info(f"Successfully added {total} chunks to vector store")
    
    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Search vector store using embedding similarity.
        
        Args:
            query: Original query text (for logging)
            query_embedding: Query embedding vector
            top_k: Number of results to return (default from config)
            filters: Metadata filters (e.g., {"severity": "critical"})
        
        Returns:
            List of results with text, metadata, and similarity scores
        """
        top_k = top_k or settings.retrieval_top_k
        logger.debug(
            f"Vector search: query='{query[:50]}...', top_k={top_k}, filters={filters}"
        )
        
        # Build where clause from filters
        where_clause = None
        if filters:
            where_clause = self._build_where_clause(filters)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx],
                    "score": 1 / (1 + results["distances"][0][idx])  # Convert distance to similarity score
                })
        
        logger.debug(f"Found {len(formatted_results)} results")
        
        return formatted_results
    
    def search_by_metadata(
        self,
        filters: dict[str, Any],
        limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Search using metadata filters only (no vector similarity).
        
        Useful for queries like "all critical CVEs" or "all Maven vulnerabilities".
        
        Args:
            filters: Metadata filters (e.g., {"severity": "critical"})
            limit: Maximum results to return (default from config)
        
        Returns:
            List of matching chunks
        """
        limit = limit or settings.retrieval_metadata_limit
        where_clause = self._build_where_clause(filters)
        
        results = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        formatted_results = []
        
        if results["ids"]:
            for idx in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][idx],
                    "text": results["documents"][idx],
                    "metadata": results["metadatas"][idx]
                })
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution."""
        self.client.delete_collection(name=self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")
    
    def reset_collection(self) -> None:
        """Reset collection (delete and recreate)."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "CVE knowledge base for VulnRemedy"}
        )
        logger.info(f"Reset collection: {self.collection_name}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        count = self.collection.count()
        
        # Sample metadata to understand distribution
        sample = self.collection.peek(limit=settings.retrieval_stats_sample_size)
        
        stats = {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
        
        # Analyze metadata if documents exist
        if sample["metadatas"]:
            severities = {}
            chunk_types = {}
            sources = {}
            
            for metadata in sample["metadatas"]:
                severity = metadata.get("severity", "unknown")
                chunk_type = metadata.get("chunk_type", "unknown")
                source = metadata.get("source", "unknown")
                
                severities[severity] = severities.get(severity, 0) + 1
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                sources[source] = sources.get(source, 0) + 1
            
            stats["sample_severities"] = severities
            stats["sample_chunk_types"] = chunk_types
            stats["sample_sources"] = sources
        
        return stats
    
    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """
        Build ChromaDB where clause from filters.
        
        Supports:
        - Equality: {"severity": "critical"}
        - In: {"ecosystems": ["maven", "npm"]} (if field contains any of these)
        """
        if not filters:
            return {}
        
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # "In" condition - metadata field contains any of these values
                conditions.append({key: {"$in": value}})
            else:
                # Equality condition
                conditions.append({key: value})
        
        # Combine with AND if multiple conditions
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        
        return {}