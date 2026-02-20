"""
Retriever — Hybrid retrieval combining vector search and keyword matching.

Implements a two-stage retrieval strategy:
1. Vector similarity search (semantic)
2. Keyword search (BM25 for exact matches)
3. Merge and blend scores

Architectural notes:
    - Hybrid search catches both semantic and exact matches
    - Configurable blend weights (vector vs keyword importance)
    - Metadata filtering applied to both search strategies
    - Results include provenance (which search found them)
"""

from __future__ import annotations
import math
import re
from collections import defaultdict
from typing import Any, Optional

from vulnremedy.rag.embeddings.embedding_service import EmbeddingService
from vulnremedy.rag.retrieval.vector_store import VectorStore
from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


class HybridRetriever:
    """
    Combines vector similarity and keyword search for robust retrieval.
    
    Usage:
        retriever = HybridRetriever(vector_store, embedding_service)
        results = retriever.retrieve("How to fix Log4Shell?", top_k=5)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_service: EmbeddingService instance
            vector_weight: Weight for vector search scores (default from config)
            keyword_weight: Weight for keyword search scores (default from config)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        self.vector_weight = vector_weight or settings.retrieval_vector_weight
        self.keyword_weight = keyword_weight or settings.retrieval_keyword_weight
        
        # Validate weights sum to 1.0
        total_weight = self.vector_weight + self.keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Retrieval weights don't sum to 1.0 (got {total_weight}), normalizing"
            )
            self.vector_weight = self.vector_weight / total_weight
            self.keyword_weight = self.keyword_weight / total_weight
        
        logger.info(
            "Hybrid Retriever initialized",
            vector_weight=self.vector_weight,
            keyword_weight=self.keyword_weight
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        vector_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query
            top_k: Final number of results to return (default from config)
            filters: Metadata filters (e.g., {"severity": "critical"})
            vector_top_k: Results from vector search (default: top_k * 2)
            keyword_top_k: Results from keyword search (default: top_k * 2)
        
        Returns:
            List of results with text, metadata, and blended scores
        """
        top_k = top_k or settings.retrieval_top_k
        vector_top_k = vector_top_k or (top_k * 2)
        keyword_top_k = keyword_top_k or (top_k * 2)
        
        logger.info(
            f"Hybrid retrieval: query='{query[:50]}...', "
            f"top_k={top_k}, filters={filters}"
        )
        
        # Stage 1: Vector search
        vector_results = self._vector_search(
            query=query,
            top_k=vector_top_k,
            filters=filters
        )
        
        # Stage 2: Keyword search
        keyword_results = self._keyword_search(
            query=query,
            top_k=keyword_top_k,
            filters=filters
        )
        
        # Stage 3: Merge and blend
        merged_results = self._merge_results(
            vector_results=vector_results,
            keyword_results=keyword_results
        )
        
        # Sort by blended score and take top_k
        merged_results.sort(key=lambda x: x["blended_score"], reverse=True)
        final_results = merged_results[:top_k]
        
        logger.info(
            f"Hybrid retrieval complete: "
            f"vector={len(vector_results)}, keyword={len(keyword_results)}, "
            f"merged={len(merged_results)}, final={len(final_results)}"
        )
        
        return final_results
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Returns results with normalized scores (0-1 range).
        """
        # Generate query embedding
        query_embedding = self._embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Add search source to metadata
        for result in results:
            result["search_source"] = "vector"
            result["vector_score"] = result["score"]
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Perform keyword-based search using BM25 algorithm.
        
        Returns results with normalized scores (0-1 range).
        """
        # Get all documents from vector store (with filters if provided)
        if filters:
            documents = self.vector_store.search_by_metadata(
                filters=filters,
                limit=settings.retrieval_metadata_limit
            )
        else:
            # Get all documents - this could be expensive for large collections
            # In production, you'd use a dedicated search index (Elasticsearch, etc.)
            documents = self.vector_store.search_by_metadata(
                filters={},
                limit=settings.retrieval_metadata_limit
            )
        
        if not documents:
            logger.warning("No documents available for keyword search")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Calculate BM25 scores
        bm25_scores = self._calculate_bm25(
            query_tokens=query_tokens,
            documents=documents
        )
        
        # Sort by score and take top_k
        scored_results = [
            {
                **doc,
                "score": score,
                "keyword_score": score,
                "search_source": "keyword"
            }
            for doc, score in bm25_scores
        ]
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results[:top_k]
    
    def _merge_results(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Merge vector and keyword results, blending scores.
        
        For chunks found by both searches, blend their scores.
        For chunks found by only one, use that score with appropriate weight.
        """
        # Index results by chunk ID
        results_by_id = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result["id"]
            results_by_id[chunk_id] = {
                **result,
                "vector_score": result.get("vector_score", 0.0),
                "keyword_score": 0.0,  # Default if not found by keyword search
                "found_by": ["vector"]
            }
        
        # Merge keyword results
        for result in keyword_results:
            chunk_id = result["id"]
            
            if chunk_id in results_by_id:
                # Found by both - update keyword score
                results_by_id[chunk_id]["keyword_score"] = result.get("keyword_score", 0.0)
                results_by_id[chunk_id]["found_by"].append("keyword")
            else:
                # Found only by keyword search
                results_by_id[chunk_id] = {
                    **result,
                    "vector_score": 0.0,
                    "keyword_score": result.get("keyword_score", 0.0),
                    "found_by": ["keyword"]
                }
        
        # Calculate blended scores
        merged_results = []
        
        for chunk_id, result in results_by_id.items():
            vector_score = result["vector_score"]
            keyword_score = result["keyword_score"]
            
            blended_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            
            result["blended_score"] = blended_score
            merged_results.append(result)
        
        return merged_results
    
    def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for query text."""
        # Use embedding service to embed single query
        # This is less efficient than batch embedding but fine for single queries
        embeddings = self.embedding_service._embed_batch([query])
        return embeddings[0]
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for keyword search.
        
        Simple tokenization: lowercase, split on non-alphanumeric, remove stopwords.
        """
        # Lowercase
        text = text.lower()
        
        # Extract tokens (alphanumeric sequences)
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove common stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with'
        }
        
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        return tokens
    
    def _calculate_bm25(
        self,
        query_tokens: list[str],
        documents: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Calculate BM25 scores for documents.
        
        BM25 is a ranking function used by search engines.
        
        Args:
            query_tokens: Tokenized query
            documents: List of document dicts with 'text' field
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling length normalization
        
        Returns:
            List of (document, score) tuples
        """
        # Tokenize all documents
        doc_tokens_list = [self._tokenize(doc["text"]) for doc in documents]
        
        # Calculate document frequencies
        doc_count = len(documents)
        doc_freq = defaultdict(int)
        
        for doc_tokens in doc_tokens_list:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Calculate average document length
        avg_doc_length = sum(len(tokens) for tokens in doc_tokens_list) / doc_count
        
        # Calculate BM25 scores
        scores = []
        
        for doc, doc_tokens in zip(documents, doc_tokens_list):
            score = 0.0
            doc_length = len(doc_tokens)
            
            # Count term frequencies in document
            term_freq = defaultdict(int)
            for token in doc_tokens:
                term_freq[token] += 1
            
            # Calculate score for each query token
            for token in query_tokens:
                if token not in term_freq:
                    continue
                
                # IDF component
                df = doc_freq.get(token, 0)
                idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
                
                # Term frequency component
                tf = term_freq[token]
                
                # Length normalization
                norm = k1 * ((1 - b) + b * (doc_length / avg_doc_length))
                
                # BM25 formula
                score += idf * (tf * (k1 + 1)) / (tf + norm)
            
            scores.append((doc, score))
        
        # Normalize scores to 0-1 range
        if scores:
            max_score = max(s for _, s in scores)
            if max_score > 0:
                scores = [(doc, score / max_score) for doc, score in scores]
        
        return scores