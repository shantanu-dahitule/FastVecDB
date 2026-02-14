"""
Query Result Cache - Caches query results for repeated queries.
"""

import hashlib
import json
from typing import List, Dict, Optional, Tuple, Any


class QueryResultCache:
    """
    Cache for query results.
    
    Uses query vector hash as key to detect identical or similar queries.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize query result cache.
        
        Args:
            max_size: Maximum number of cached queries
        """
        self.max_size = max_size
        self.cache: Dict[str, Tuple[List[Dict], float]] = {}
        self.access_order: List[str] = []
    
    def _hash_query(self, query_vector: List[float], top_k: int, metric: str) -> str:
        """
        Generate hash for a query.
        
        Args:
            query_vector: Query vector
            top_k: Number of results requested
            metric: Similarity metric used
            
        Returns:
            Hash string
        """
        # Create a stable hash from query parameters
        query_str = json.dumps({
            'vector': [round(x, 6) for x in query_vector],  # Round for stability
            'top_k': top_k,
            'metric': metric
        }, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()
    
    def get(self, query_vector: List[float], top_k: int, metric: str) -> Optional[List[Dict]]:
        """
        Get cached results for a query.
        
        Args:
            query_vector: Query vector
            top_k: Number of results requested
            metric: Similarity metric used
            
        Returns:
            Cached results or None if not found
        """
        cache_key = self._hash_query(query_vector, top_k, metric)
        
        if cache_key in self.cache:
            # Update access order (LRU)
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key][0]
        
        return None
    
    def put(
        self,
        query_vector: List[float],
        top_k: int,
        metric: str,
        results: List[Dict],
        query_time: float
    ) -> None:
        """
        Cache query results.
        
        Args:
            query_vector: Query vector
            top_k: Number of results requested
            metric: Similarity metric used
            results: Query results to cache
            query_time: Time taken for the query (for statistics)
        """
        cache_key = self._hash_query(query_vector, top_k, metric)
        
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Add new entry
        self.cache[cache_key] = (results, query_time)
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
    
    def clear(self) -> None:
        """Clear all cached queries."""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': 0.0  # Would need to track hits/misses
        }

