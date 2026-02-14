"""
Vector utilities and operations.
"""

import math
from typing import List, Optional


class Vector:
    """Represents a vector with metadata."""
    
    def __init__(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[dict] = None,
        normalized: bool = False
    ):
        """
        Initialize a vector.
        
        Args:
            id: Unique identifier for the vector
            vector: The vector data
            metadata: Optional metadata dictionary
            normalized: Whether the vector is already normalized
        """
        self.id = id
        self.vector = vector
        self.metadata = metadata or {}
        self.normalized = normalized
        self._norm_cache: Optional[float] = None
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the vector."""
        return len(self.vector)
    
    @property
    def norm(self) -> float:
        """Compute and cache the L2 norm of the vector."""
        if self._norm_cache is None:
            self._norm_cache = math.sqrt(sum(x * x for x in self.vector))
        return self._norm_cache
    
    def normalize(self) -> "Vector":
        """
        Create a normalized copy of this vector.
        
        Returns:
            New normalized Vector instance
        """
        if self.normalized:
            return Vector(self.id, self.vector.copy(), self.metadata.copy(), True)
        
        norm = self.norm
        if norm == 0.0:
            raise ValueError("Cannot normalize zero vector")
        
        normalized_vec = [x / norm for x in self.vector]
        return Vector(self.id, normalized_vec, self.metadata.copy(), True)
    
    def __repr__(self) -> str:
        return f"Vector(id={self.id!r}, dim={self.dimension}, normalized={self.normalized})"


def normalize_vector(vec: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Vector to normalize
        
    Returns:
        Normalized vector
        
    Raises:
        ValueError: If vector is zero vector
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return [x / norm for x in vec]

