"""
Hot Vector Cache - Caches frequently accessed vectors in memory.
"""

from typing import List, Dict, Optional
from collections import OrderedDict


class HotVectorCache:
    """
    LRU cache for frequently accessed vectors.
    
    Keeps hot vectors in memory for fast access without disk I/O.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize hot vector cache.
        
        Args:
            max_size: Maximum number of vectors to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
    
    def get(self, vector_id: str) -> Optional[List[float]]:
        """
        Get a cached vector.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Vector data or None if not cached
        """
        if vector_id in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(vector_id)
            return self.cache[vector_id]
        return None
    
    def put(self, vector_id: str, vector: List[float]) -> None:
        """
        Cache a vector.
        
        Args:
            vector_id: ID of the vector
            vector: Vector data
        """
        if vector_id in self.cache:
            # Update existing entry
            self.cache.move_to_end(vector_id)
            self.cache[vector_id] = vector
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[vector_id] = vector
    
    def remove(self, vector_id: str) -> None:
        """
        Remove a vector from cache.
        
        Args:
            vector_id: ID of the vector to remove
        """
        if vector_id in self.cache:
            del self.cache[vector_id]
    
    def clear(self) -> None:
        """Clear all cached vectors."""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size
        }

