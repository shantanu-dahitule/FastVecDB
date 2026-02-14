"""
Example of batch inserting vectors.
"""

from fastvecdb import FastVecDB, SimilarityMetric
import random
import time


def generate_vector(dim: int, seed: int = None) -> list:
    """Generate a deterministic vector."""
    if seed is not None:
        random.seed(seed)
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def main():
    dimension = 128
    num_vectors = 1000
    
    print(f"Batch inserting {num_vectors} vectors...")
    
    # Initialize database
    db = FastVecDB(
        storage_path="./example_data_batch",
        dimension=dimension,
        similarity_metric=SimilarityMetric.COSINE
    )
    
    # Batch insert
    start_time = time.time()
    
    for i in range(num_vectors):
        vector = generate_vector(dimension, seed=i)
        db.insert(
            vector_id=f"batch_vec_{i}",
            vector=vector,
            metadata={
                "index": i,
                "batch": "test_batch",
                "created_at": time.time()
            },
            bucket_id="batch_vectors"
        )
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Inserted {i + 1}/{num_vectors} vectors "
                  f"({rate:.1f} vectors/sec)")
    
    elapsed = time.time() - start_time
    print(f"\nInserted {num_vectors} vectors in {elapsed:.2f} seconds")
    print(f"Average rate: {num_vectors / elapsed:.1f} vectors/sec")
    
    # Test search performance
    print("\nTesting search performance...")
    query_vector = generate_vector(dimension, seed=9999)
    
    # First search (cold cache)
    start_time = time.time()
    results1 = db.search(query_vector, top_k=10)
    time1 = time.time() - start_time
    print(f"First search (cold): {time1*1000:.2f}ms, found {len(results1)} results")
    
    # Second search (warm cache)
    start_time = time.time()
    results2 = db.search(query_vector, top_k=10)
    time2 = time.time() - start_time
    print(f"Second search (warm): {time2*1000:.2f}ms, found {len(results2)} results")
    print(f"Cache speedup: {time1/time2:.1f}x")
    
    # Get statistics
    stats = db.get_stats()
    print(f"\nDatabase statistics:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Buckets: {stats['buckets']}")
    if stats.get('cache_stats'):
        print(f"  Cache stats: {stats['cache_stats']}")
    
    # Close
    db.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

