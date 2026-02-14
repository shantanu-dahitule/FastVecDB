"""
FastVecDB Cache Quick Demo - Simple Example

A simple, easy-to-understand demonstration of FastVecDB's caching benefits.
Perfect for quick demos and understanding the core value proposition.
"""

import time
from fastvecdb import FastVecDB, SimilarityMetric


def main():
    print("=" * 60)
    print("FastVecDB Cache Quick Demo")
    print("=" * 60)
    
    # Create database with caching
    print("\n1. Creating database with intelligent caching...")
    db = FastVecDB(
        storage_path="./quick_cache_demo",
        dimension=64,
        similarity_metric=SimilarityMetric.COSINE,
        enable_cache=True
    )
    
    # Insert some vectors
    print("\n2. Inserting sample vectors...")
    vectors = [
        ([1.0] * 64, "vector1", {"name": "Document 1"}),
        ([0.9] * 64, "vector2", {"name": "Document 2"}),
        ([0.8] * 64, "vector3", {"name": "Document 3"}),
    ]
    
    for vector, vec_id, metadata in vectors:
        db.insert(vec_id, vector, metadata)
        print(f"   ✓ Inserted {vec_id}")
    
    # First search (COLD - no cache)
    print("\n3. First search (COLD - no cache):")
    query = [0.95] * 64
    start = time.time()
    results1 = db.search(query, top_k=2)
    cold_time = time.time() - start
    print(f"   Time: {cold_time*1000:.2f}ms")
    print(f"   Found: {len(results1)} results")
    
    # Second search (WARM - cache hit!)
    print("\n4. Second search (WARM - cache hit!):")
    start = time.time()
    results2 = db.search(query, top_k=2)  # Same query!
    warm_time = time.time() - start
    print(f"   Time: {warm_time*1000:.2f}ms")
    print(f"   Found: {len(results2)} results")
    
    # Show speedup
    if warm_time > 0:
        speedup = cold_time / warm_time
        print(f"\n   🚀 Cache made it {speedup:.1f}x faster!")
    
    # Show cache stats
    print("\n5. Cache Statistics:")
    stats = db.get_stats()
    if stats.get('cache_stats'):
        cache_stats = stats['cache_stats']
        print(f"   Query cache: {cache_stats.get('query_cache', {}).get('size', 0)} queries cached")
        print(f"   Hot vectors: {cache_stats.get('hot_vector_cache', {}).get('size', 0)} vectors cached")
    
    print("\n" + "=" * 60)
    print("Key Benefit: Repeated queries are instant!")
    print("=" * 60)
    
    db.close()


if __name__ == "__main__":
    main()

