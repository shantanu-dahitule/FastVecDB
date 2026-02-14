"""
Cache Manager - Coordinates all caching layers.
"""

from typing import List, Dict, Optional
from fastvecdb.cache.query_cache import QueryResultCache
from fastvecdb.cache.neighborhood_cache import NeighborhoodCache
from fastvecdb.cache.hot_vector_cache import HotVectorCache


class CacheManager:
    """
    Manages all caching layers in FastVecDB.
    
    Coordinates:
    - Query Result Cache
    - Semantic Neighborhood Cache
    - Hot Vector Cache
    """
    
    def __init__(
        self,
        query_cache_size: int = 1000,
        neighborhood_size: int = 50,
        hot_vector_cache_size: int = 1000
    ):
        """
        Initialize cache manager.
        
        Args:
            query_cache_size: Maximum size of query result cache
            neighborhood_size: Maximum neighbors per vector in neighborhood cache
            hot_vector_cache_size: Maximum size of hot vector cache
        """
        self.query_cache = QueryResultCache(max_size=query_cache_size)
        self.neighborhood_cache = NeighborhoodCache(max_neighbors=neighborhood_size)
        self.hot_vector_cache = HotVectorCache(max_size=hot_vector_cache_size)
    
    def get_cached_query(
        self,
        query_vector: List[float],
        top_k: int,
        metric: str
    ) -> Optional[List[Dict]]:
        """
        Check if query results are cached.
        
        Args:
            query_vector: Query vector
            top_k: Number of results requested
            metric: Similarity metric used
            
        Returns:
            Cached results or None
        """
        return self.query_cache.get(query_vector, top_k, metric)
    
    def cache_query(
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
            results: Query results
            query_time: Time taken for query
        """
        self.query_cache.put(query_vector, top_k, metric, results, query_time)
    
    def get_cached_neighbors(self, vector_id: str) -> Optional[List[Dict]]:
        """
        Get cached neighbors for a vector.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Cached neighbors or None
        """
        return self.neighborhood_cache.get_neighbors(vector_id)
    
    def cache_neighbors(self, vector_id: str, neighbors: List[Dict]) -> None:
        """
        Cache neighbors for a vector.
        
        Args:
            vector_id: ID of the vector
            neighbors: List of neighbor results
        """
        self.neighborhood_cache.cache_neighbors(vector_id, neighbors)
    
    def get_cached_vector(self, vector_id: str) -> Optional[List[float]]:
        """
        Get a cached vector.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Cached vector or None
        """
        return self.hot_vector_cache.get(vector_id)
    
    def cache_vector(self, vector_id: str, vector: List[float]) -> None:
        """
        Cache a vector.
        
        Args:
            vector_id: ID of the vector
            vector: Vector data
        """
        self.hot_vector_cache.put(vector_id, vector)
    
    def invalidate_vector(self, vector_id: str) -> None:
        """
        Invalidate all cache entries for a vector.
        
        Args:
            vector_id: ID of the vector to invalidate
        """
        self.hot_vector_cache.remove(vector_id)
        self.neighborhood_cache.invalidate(vector_id)
        # Note: Query cache is not invalidated as it's query-based, not vector-based
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.neighborhood_cache.clear()
        self.hot_vector_cache.clear()
    
    def get_stats(self) -> Dict:
        """
        Get statistics for all caches.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'query_cache': self.query_cache.get_stats(),
            'neighborhood_cache': self.neighborhood_cache.get_stats(),
            'hot_vector_cache': self.hot_vector_cache.get_stats()
        }

