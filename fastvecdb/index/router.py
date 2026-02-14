"""
Query Router - Routes queries to appropriate buckets and coordinates search.
"""

import time
from typing import List, Dict, Optional
from fastvecdb.core.similarity import SimilarityMetric
from fastvecdb.index.bucket import VectorBucket


class QueryRouter:
    """
    Routes queries to appropriate buckets and coordinates search across buckets.
    """
    
    def __init__(self):
        """Initialize query router."""
        self.buckets: Dict[str, VectorBucket] = {}
        self.default_bucket_id = "default"
    
    def create_bucket(self, bucket_id: str, dimension: int) -> VectorBucket:
        """
        Create a new vector bucket.
        
        Args:
            bucket_id: Unique identifier for the bucket
            dimension: Dimension of vectors in this bucket
            
        Returns:
            Created VectorBucket instance
        """
        if bucket_id in self.buckets:
            raise ValueError(f"Bucket {bucket_id} already exists")
        
        bucket = VectorBucket(bucket_id, dimension)
        self.buckets[bucket_id] = bucket
        return bucket
    
    def get_bucket(self, bucket_id: str) -> Optional[VectorBucket]:
        """
        Get a bucket by ID.
        
        Args:
            bucket_id: ID of the bucket
            
        Returns:
            VectorBucket or None if not found
        """
        return self.buckets.get(bucket_id)
    
    def get_or_create_bucket(self, bucket_id: str, dimension: int) -> VectorBucket:
        """
        Get an existing bucket or create a new one.
        
        Args:
            bucket_id: ID of the bucket
            dimension: Dimension of vectors (used if creating new bucket)
            
        Returns:
            VectorBucket instance
        """
        bucket = self.get_bucket(bucket_id)
        if bucket is None:
            bucket = self.create_bucket(bucket_id, dimension)
        return bucket
    
    def add_vector_to_bucket(self, bucket_id: str, vector) -> None:
        """
        Add a vector to a bucket.
        
        Args:
            bucket_id: ID of the bucket
            vector: Vector to add
        """
        bucket = self.get_or_create_bucket(bucket_id, vector.dimension)
        bucket.add_vector(vector)
    
    def remove_vector_from_bucket(self, bucket_id: str, vector_id: str) -> bool:
        """
        Remove a vector from a bucket.
        
        Args:
            bucket_id: ID of the bucket
            vector_id: ID of the vector to remove
            
        Returns:
            True if vector was removed, False if not found
        """
        bucket = self.get_bucket(bucket_id)
        if bucket:
            return bucket.remove_vector(vector_id)
        return False
    
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        metric: SimilarityMetric,
        bucket_ids: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search across buckets.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            metric: Similarity metric to use
            bucket_ids: Optional list of bucket IDs to search (None = search all)
            threshold: Optional similarity/distance threshold
            
        Returns:
            List of search results sorted by score
        """
        all_results = []
        
        # Determine which buckets to search
        buckets_to_search = []
        if bucket_ids:
            for bid in bucket_ids:
                bucket = self.get_bucket(bid)
                if bucket:
                    buckets_to_search.append(bucket)
        else:
            # Search all buckets
            buckets_to_search = list(self.buckets.values())
        
        # Search each bucket
        for bucket in buckets_to_search:
            try:
                results = bucket.search(query_vector, top_k, metric, threshold)
                all_results.extend(results)
            except Exception:
                # Skip buckets that cause errors
                continue
        
        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def list_buckets(self) -> List[str]:
        """List all bucket IDs."""
        return list(self.buckets.keys())
    
    def get_bucket_stats(self) -> Dict[str, int]:
        """
        Get statistics for all buckets.
        
        Returns:
            Dictionary mapping bucket_id to vector count
        """
        return {bid: bucket.size() for bid, bucket in self.buckets.items()}

