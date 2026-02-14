"""
Vector Buckets - Organize vectors for efficient search.
"""

import time
from typing import List, Dict, Optional, Tuple
from fastvecdb.core.similarity import SimilarityMetric, compute_similarity
from fastvecdb.core.vector import Vector


class VectorBucket:
    """
    A bucket containing a collection of vectors.
    
    Buckets help organize vectors and reduce search space.
    """
    
    def __init__(self, bucket_id: str, dimension: int):
        """
        Initialize a vector bucket.
        
        Args:
            bucket_id: Unique identifier for the bucket
            dimension: Dimension of vectors in this bucket
        """
        self.bucket_id = bucket_id
        self.dimension = dimension
        self.vectors: Dict[str, Vector] = {}
        self.created_at = time.time()
    
    def add_vector(self, vector: Vector) -> None:
        """
        Add a vector to the bucket.
        
        Args:
            vector: Vector to add
            
        Raises:
            ValueError: If vector dimension doesn't match bucket dimension
        """
        if vector.dimension != self.dimension:
            raise ValueError(
                f"Vector dimension {vector.dimension} doesn't match bucket dimension {self.dimension}"
            )
        self.vectors[vector.id] = vector
    
    def remove_vector(self, vector_id: str) -> bool:
        """
        Remove a vector from the bucket.
        
        Args:
            vector_id: ID of the vector to remove
            
        Returns:
            True if vector was removed, False if not found
        """
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Get a vector from the bucket.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Vector or None if not found
        """
        return self.vectors.get(vector_id)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        metric: SimilarityMetric,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar vectors in this bucket.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            metric: Similarity metric to use
            threshold: Optional similarity threshold (for similarity metrics) or distance threshold (for distance metrics)
            
        Returns:
            List of search results, each containing 'id', 'score', and 'metadata'
        """
        results = []
        
        for vector in self.vectors.values():
            try:
                if metric == SimilarityMetric.EUCLIDEAN:
                    # For distance metrics, lower is better
                    score = compute_similarity(query_vector, vector.vector, metric)
                    if threshold is not None and score > threshold:
                        continue
                    # Negate for consistent sorting (higher is better)
                    results.append({
                        'id': vector.id,
                        'score': -score,  # Negate distance for sorting
                        'distance': score,
                        'metadata': vector.metadata
                    })
                else:
                    # For similarity metrics, higher is better
                    score = compute_similarity(query_vector, vector.vector, metric)
                    if threshold is not None and score < threshold:
                        continue
                    results.append({
                        'id': vector.id,
                        'score': score,
                        'metadata': vector.metadata
                    })
            except ValueError:
                # Skip vectors that cause errors (e.g., dimension mismatch)
                continue
        
        # Sort by score (descending) and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def size(self) -> int:
        """Get the number of vectors in this bucket."""
        return len(self.vectors)
    
    def list_vector_ids(self) -> List[str]:
        """List all vector IDs in this bucket."""
        return list(self.vectors.keys())

