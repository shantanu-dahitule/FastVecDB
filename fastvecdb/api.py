"""
FastVecDB API - Main interface for vector search and retrieval.
"""

import os
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

from fastvecdb.core.similarity import SimilarityMetric, compute_similarity
from fastvecdb.core.vector import Vector
from fastvecdb.core.storage import SQLiteStorage, MmapVectorStorage, PickleSnapshotStorage
from fastvecdb.cache.cache_manager import CacheManager
from fastvecdb.index.router import QueryRouter
from fastvecdb.index.bucket import VectorBucket


class FastVecDB:
    """
    FastVecDB - Pure-Python vector search and retrieval framework.
    
    Main API for vector operations with intelligent caching and persistent storage.
    """
    
    def __init__(
        self,
        storage_path: str = "./fastvecdb_data",
        dimension: Optional[int] = None,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        enable_cache: bool = True,
        query_cache_size: int = 1000,
        neighborhood_cache_size: int = 50,
        hot_vector_cache_size: int = 1000
    ):
        """
        Initialize FastVecDB.
        
        Args:
            storage_path: Path to storage directory
            dimension: Expected vector dimension (required for first insert)
            similarity_metric: Default similarity metric to use
            enable_cache: Whether to enable caching
            query_cache_size: Maximum size of query result cache
            neighborhood_cache_size: Maximum neighbors per vector in neighborhood cache
            hot_vector_cache_size: Maximum size of hot vector cache
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        
        # Initialize storage backends
        db_path = str(self.storage_path / "metadata.db")
        self.metadata_storage = SQLiteStorage(db_path)
        
        # Mmap storage will be initialized when dimension is known
        self.vector_storage: Optional[MmapVectorStorage] = None
        mmap_path = str(self.storage_path / "vectors.mmap")
        if dimension:
            self.vector_storage = MmapVectorStorage(mmap_path, dimension)
        
        # Snapshot storage
        snapshot_path = str(self.storage_path / "snapshot.pkl")
        self.snapshot_storage = PickleSnapshotStorage(snapshot_path)
        
        # Initialize caching
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache_manager = CacheManager(
                query_cache_size=query_cache_size,
                neighborhood_size=neighborhood_cache_size,
                hot_vector_cache_size=hot_vector_cache_size
            )
        else:
            self.cache_manager = None
        
        # Initialize query router
        self.router = QueryRouter()
        
        # Vector ID to bucket mapping
        self.vector_to_bucket: Dict[str, str] = {}
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """Load existing vectors from storage."""
        try:
            vector_ids = self.metadata_storage.list_vectors()
            for vector_id in vector_ids:
                metadata = self.metadata_storage.load_metadata(vector_id)
                if metadata:
                    bucket_id = metadata.get('bucket_id', 'default')
                    self.vector_to_bucket[vector_id] = bucket_id
        except Exception:
            # If loading fails, start fresh
            pass
    
    def _ensure_dimension(self, dimension: int) -> None:
        """
        Ensure dimension is set and matches.
        
        Args:
            dimension: Vector dimension
            
        Raises:
            ValueError: If dimension doesn't match
        """
        if self.dimension is None:
            self.dimension = dimension
            # Initialize mmap storage
            mmap_path = str(self.storage_path / "vectors.mmap")
            self.vector_storage = MmapVectorStorage(mmap_path, dimension)
        elif self.dimension != dimension:
            raise ValueError(
                f"Vector dimension {dimension} doesn't match database dimension {self.dimension}"
            )
    
    def insert(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict] = None,
        bucket_id: Optional[str] = None
    ) -> None:
        """
        Insert a vector into the database.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: Vector data
            metadata: Optional metadata dictionary
            bucket_id: Optional bucket ID (defaults to 'default')
        """
        self._ensure_dimension(len(vector))
        
        if bucket_id is None:
            bucket_id = "default"
        
        # Create Vector object
        vec_obj = Vector(vector_id, vector, metadata or {})
        
        # Add to bucket
        self.router.add_vector_to_bucket(bucket_id, vec_obj)
        self.vector_to_bucket[vector_id] = bucket_id
        
        # Store in persistent storage
        self.metadata_storage.save_vector(vector_id, vector, metadata or {}, bucket_id)
        
        # Cache the vector
        if self.cache_manager:
            self.cache_manager.cache_vector(vector_id, vector)
    
    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Update an existing vector.
        
        Args:
            vector_id: ID of the vector to update
            vector: New vector data (optional)
            metadata: New metadata (optional, will merge with existing)
            
        Raises:
            ValueError: If vector not found
        """
        bucket_id = self.vector_to_bucket.get(vector_id)
        if bucket_id is None:
            raise ValueError(f"Vector {vector_id} not found")
        
        bucket = self.router.get_bucket(bucket_id)
        if bucket is None:
            raise ValueError(f"Bucket {bucket_id} not found")
        
        existing_vec = bucket.get_vector(vector_id)
        if existing_vec is None:
            raise ValueError(f"Vector {vector_id} not found in bucket")
        
        # Update vector data if provided
        new_vector = vector if vector is not None else existing_vec.vector
        
        # Update metadata
        new_metadata = existing_vec.metadata.copy()
        if metadata:
            new_metadata.update(metadata)
        
        # Create updated Vector object
        updated_vec = Vector(vector_id, new_vector, new_metadata)
        
        # Update in bucket
        bucket.add_vector(updated_vec)
        
        # Update in storage
        self.metadata_storage.save_vector(vector_id, new_vector, new_metadata, bucket_id)
        
        # Invalidate caches
        if self.cache_manager:
            self.cache_manager.invalidate_vector(vector_id)
            self.cache_manager.cache_vector(vector_id, new_vector)
    
    def delete(self, vector_id: str) -> None:
        """
        Delete a vector from the database.
        
        Args:
            vector_id: ID of the vector to delete
        """
        bucket_id = self.vector_to_bucket.get(vector_id)
        if bucket_id:
            self.router.remove_vector_from_bucket(bucket_id, vector_id)
            del self.vector_to_bucket[vector_id]
        
        # Remove from storage
        self.metadata_storage.delete_vector(vector_id)
        
        # Invalidate caches
        if self.cache_manager:
            self.cache_manager.invalidate_vector(vector_id)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        metric: Optional[SimilarityMetric] = None,
        bucket_ids: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            metric: Similarity metric to use (defaults to instance default)
            bucket_ids: Optional list of bucket IDs to search
            threshold: Optional similarity/distance threshold
            
        Returns:
            List of search results, each containing 'id', 'score', and 'metadata'
        """
        if metric is None:
            metric = self.similarity_metric
        
        start_time = time.time()
        
        # Check query cache
        if self.cache_manager:
            cached_results = self.cache_manager.get_cached_query(
                query_vector, top_k, metric.value
            )
            if cached_results is not None:
                return cached_results
        
        # Perform search
        results = self.router.search(
            query_vector, top_k, metric, bucket_ids, threshold
        )
        
        query_time = time.time() - start_time
        
        # Cache results
        if self.cache_manager:
            self.cache_manager.cache_query(
                query_vector, top_k, metric.value, results, query_time
            )
        
        return results
    
    def get(self, vector_id: str) -> Optional[Dict]:
        """
        Get a vector by ID.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Dictionary with 'id', 'vector', and 'metadata', or None if not found
        """
        # Check hot vector cache
        if self.cache_manager:
            cached_vector = self.cache_manager.get_cached_vector(vector_id)
            if cached_vector is not None:
                metadata = self.metadata_storage.load_metadata(vector_id) or {}
                return {
                    'id': vector_id,
                    'vector': cached_vector,
                    'metadata': metadata
                }
        
        # Get from bucket
        bucket_id = self.vector_to_bucket.get(vector_id)
        if bucket_id:
            bucket = self.router.get_bucket(bucket_id)
            if bucket:
                vec = bucket.get_vector(vector_id)
                if vec:
                    # Cache it
                    if self.cache_manager:
                        self.cache_manager.cache_vector(vector_id, vec.vector)
                    return {
                        'id': vec.id,
                        'vector': vec.vector,
                        'metadata': vec.metadata
                    }
        
        return None
    
    def list_vectors(self) -> List[str]:
        """
        List all vector IDs in the database.
        
        Returns:
            List of vector IDs
        """
        return self.metadata_storage.list_vectors()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_vectors': len(self.vector_to_bucket),
            'dimension': self.dimension,
            'similarity_metric': self.similarity_metric.value,
            'buckets': self.router.get_bucket_stats(),
            'cache_enabled': self.enable_cache
        }
        
        if self.cache_manager:
            stats['cache_stats'] = self.cache_manager.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all()
    
    def close(self) -> None:
        """Close the database and clean up resources."""
        if self.metadata_storage:
            self.metadata_storage.close()
        if self.vector_storage:
            self.vector_storage.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

