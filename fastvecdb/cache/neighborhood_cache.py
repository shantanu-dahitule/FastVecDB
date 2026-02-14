"""
Semantic Neighborhood Cache - Caches vectors that are semantically similar.
"""

from typing import List, Dict, Set, Optional
from collections import defaultdict


class NeighborhoodCache:
    """
    Cache for semantic neighborhoods.
    
    When a vector is queried, its neighbors are cached for faster subsequent queries.
    """
    
    def __init__(self, max_neighbors: int = 50):
        """
        Initialize neighborhood cache.
        
        Args:
            max_neighbors: Maximum number of neighbors to cache per vector
        """
        self.max_neighbors = max_neighbors
        self.neighborhoods: Dict[str, List[Dict]] = {}
        self.reverse_index: Dict[str, Set[str]] = defaultdict(set)  # vector_id -> set of query_ids
    
    def get_neighbors(self, vector_id: str) -> Optional[List[Dict]]:
        """
        Get cached neighbors for a vector.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            List of neighbor results or None if not cached
        """
        return self.neighborhoods.get(vector_id)
    
    def cache_neighbors(self, query_vector_id: str, neighbors: List[Dict]) -> None:
        """
        Cache neighbors for a query vector.
        
        Args:
            query_vector_id: ID of the query vector
            neighbors: List of neighbor results
        """
        # Limit number of neighbors
        limited_neighbors = neighbors[:self.max_neighbors]
        self.neighborhoods[query_vector_id] = limited_neighbors
        
        # Update reverse index
        for neighbor in limited_neighbors:
            neighbor_id = neighbor.get('id')
            if neighbor_id:
                self.reverse_index[neighbor_id].add(query_vector_id)
    
    def invalidate(self, vector_id: str) -> None:
        """
        Invalidate cache entries for a vector.
        
        Args:
            vector_id: ID of the vector to invalidate
        """
        # Remove from neighborhoods
        if vector_id in self.neighborhoods:
            del self.neighborhoods[vector_id]
        
        # Remove from reverse index
        if vector_id in self.reverse_index:
            del self.reverse_index[vector_id]
        
        # Remove from other neighborhoods' reverse indices
        for query_id, neighbor_set in list(self.reverse_index.items()):
            neighbor_set.discard(vector_id)
            if not neighbor_set:
                del self.reverse_index[query_id]
    
    def clear(self) -> None:
        """Clear all cached neighborhoods."""
        self.neighborhoods.clear()
        self.reverse_index.clear()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_vectors': len(self.neighborhoods),
            'max_neighbors': self.max_neighbors
        }

