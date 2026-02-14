"""
Comprehensive test suite for FastVecDB cache implementation.

Tests all cache layers:
- Query Result Cache
- Semantic Neighborhood Cache
- Hot Vector Cache
- Cache Manager integration
"""

import time
from fastvecdb import FastVecDB, SimilarityMetric
from fastvecdb.cache.query_cache import QueryResultCache
from fastvecdb.cache.neighborhood_cache import NeighborhoodCache
from fastvecdb.cache.hot_vector_cache import HotVectorCache
from fastvecdb.cache.cache_manager import CacheManager


def test_query_cache():
    """Test Query Result Cache functionality."""
    print("\n" + "=" * 60)
    print("Testing Query Result Cache")
    print("=" * 60)
    
    cache = QueryResultCache(max_size=5)
    
    # Test cache miss
    query_vec = [1.0, 2.0, 3.0]
    result = cache.get(query_vec, top_k=10, metric="cosine")
    assert result is None, "Cache should be empty initially"
    print("✓ Cache miss works correctly")
    
    # Test cache put and get
    results = [
        {"id": "vec1", "score": 0.95, "metadata": {}},
        {"id": "vec2", "score": 0.85, "metadata": {}}
    ]
    cache.put(query_vec, top_k=10, metric="cosine", results=results, query_time=0.001)
    
    cached = cache.get(query_vec, top_k=10, metric="cosine")
    assert cached is not None, "Cached result should be found"
    assert len(cached) == 2, "Cached result should have 2 items"
    assert cached[0]["id"] == "vec1", "First result should match"
    print("✓ Cache put and get work correctly")
    
    # Test cache with different parameters (should miss)
    cached2 = cache.get(query_vec, top_k=5, metric="cosine")
    assert cached2 is None, "Different top_k should cause cache miss"
    print("✓ Cache respects query parameters")
    
    # Test cache with different metric (should miss)
    cached3 = cache.get(query_vec, top_k=10, metric="euclidean")
    assert cached3 is None, "Different metric should cause cache miss"
    print("✓ Cache respects similarity metric")
    
    # Test LRU eviction
    for i in range(6):
        vec = [float(i)] * 3
        cache.put(vec, top_k=10, metric="cosine", 
                 results=[{"id": f"vec{i}", "score": 0.9}], query_time=0.001)
    
    # First inserted should be evicted
    first_result = cache.get([0.0] * 3, top_k=10, metric="cosine")
    assert first_result is None, "LRU eviction should remove oldest entry"
    print("✓ LRU eviction works correctly")
    
    # Test cache stats
    stats = cache.get_stats()
    assert stats['size'] == 5, "Cache should have max_size entries"
    print("✓ Cache statistics work correctly")
    
    # Test clear
    cache.clear()
    assert cache.get(query_vec, top_k=10, metric="cosine") is None, "Cache should be empty after clear"
    print("✓ Cache clear works correctly")
    
    print("\n✓ All Query Cache tests passed!")


def test_neighborhood_cache():
    """Test Semantic Neighborhood Cache functionality."""
    print("\n" + "=" * 60)
    print("Testing Neighborhood Cache")
    print("=" * 60)
    
    cache = NeighborhoodCache(max_neighbors=3)
    
    # Test cache miss
    neighbors = cache.get_neighbors("vec1")
    assert neighbors is None, "Cache should be empty initially"
    print("✓ Cache miss works correctly")
    
    # Test cache neighbors
    neighbors_data = [
        {"id": "vec2", "score": 0.95, "metadata": {}},
        {"id": "vec3", "score": 0.90, "metadata": {}},
        {"id": "vec4", "score": 0.85, "metadata": {}},
        {"id": "vec5", "score": 0.80, "metadata": {}}  # Should be limited
    ]
    cache.cache_neighbors("vec1", neighbors_data)
    
    cached = cache.get_neighbors("vec1")
    assert cached is not None, "Cached neighbors should be found"
    assert len(cached) == 3, "Should limit to max_neighbors"
    assert cached[0]["id"] == "vec2", "First neighbor should match"
    print("✓ Cache neighbors works correctly")
    print("✓ Max neighbors limit works correctly")
    
    # Test invalidation
    cache.invalidate("vec1")
    neighbors = cache.get_neighbors("vec1")
    assert neighbors is None, "Invalidated vector should not be cached"
    print("✓ Cache invalidation works correctly")
    
    # Test reverse index
    cache.cache_neighbors("query1", [{"id": "vec1", "score": 0.9}])
    cache.cache_neighbors("query2", [{"id": "vec1", "score": 0.8}])
    
    # Invalidate vec1, should affect both queries
    cache.invalidate("vec1")
    assert cache.get_neighbors("query1") is None or len(cache.get_neighbors("query1")) == 0, \
           "Invalidation should affect reverse index"
    print("✓ Reverse index invalidation works correctly")
    
    # Test stats
    stats = cache.get_stats()
    assert 'cached_vectors' in stats, "Stats should include cached_vectors"
    print("✓ Cache statistics work correctly")
    
    # Test clear
    cache.clear()
    assert cache.get_neighbors("vec1") is None, "Cache should be empty after clear"
    print("✓ Cache clear works correctly")
    
    print("\n✓ All Neighborhood Cache tests passed!")


def test_hot_vector_cache():
    """Test Hot Vector Cache (LRU) functionality."""
    print("\n" + "=" * 60)
    print("Testing Hot Vector Cache")
    print("=" * 60)
    
    cache = HotVectorCache(max_size=3)
    
    # Test cache miss
    vec = cache.get("vec1")
    assert vec is None, "Cache should be empty initially"
    print("✓ Cache miss works correctly")
    
    # Test cache put and get
    vector_data = [1.0, 2.0, 3.0, 4.0]
    cache.put("vec1", vector_data)
    
    cached = cache.get("vec1")
    assert cached is not None, "Cached vector should be found"
    assert cached == vector_data, "Cached vector should match"
    print("✓ Cache put and get work correctly")
    
    # Test LRU behavior
    cache.put("vec1", vector_data)  # Update existing
    cache.put("vec2", [2.0, 3.0, 4.0, 5.0])
    cache.put("vec3", [3.0, 4.0, 5.0, 6.0])
    
    # Access vec1 to make it most recently used
    cache.get("vec1")
    
    # Add vec4, should evict vec2 (least recently used)
    cache.put("vec4", [4.0, 5.0, 6.0, 7.0])
    
    assert cache.get("vec2") is None, "vec2 should be evicted (LRU)"
    assert cache.get("vec1") is not None, "vec1 should still be cached (MRU)"
    assert cache.get("vec3") is not None, "vec3 should still be cached"
    assert cache.get("vec4") is not None, "vec4 should be cached"
    print("✓ LRU eviction works correctly")
    
    # Test remove
    cache.remove("vec1")
    assert cache.get("vec1") is None, "Removed vector should not be cached"
    print("✓ Cache remove works correctly")
    
    # Test stats
    stats = cache.get_stats()
    assert stats['size'] == 2, "Cache should have 2 entries after removal"
    assert stats['max_size'] == 3, "Max size should be 3"
    print("✓ Cache statistics work correctly")
    
    # Test clear
    cache.clear()
    assert cache.get("vec3") is None, "Cache should be empty after clear"
    print("✓ Cache clear works correctly")
    
    print("\n✓ All Hot Vector Cache tests passed!")


def test_cache_manager():
    """Test Cache Manager integration."""
    print("\n" + "=" * 60)
    print("Testing Cache Manager")
    print("=" * 60)
    
    manager = CacheManager(
        query_cache_size=10,
        neighborhood_size=5,
        hot_vector_cache_size=10
    )
    
    # Test query cache
    query_vec = [1.0, 2.0, 3.0]
    results = [{"id": "vec1", "score": 0.9}]
    
    cached = manager.get_cached_query(query_vec, top_k=10, metric="cosine")
    assert cached is None, "Query cache should be empty initially"
    
    manager.cache_query(query_vec, top_k=10, metric="cosine", results=results, query_time=0.001)
    cached = manager.get_cached_query(query_vec, top_k=10, metric="cosine")
    assert cached is not None, "Query should be cached"
    print("✓ Query cache integration works")
    
    # Test neighborhood cache
    neighbors = [{"id": "vec2", "score": 0.85}]
    manager.cache_neighbors("vec1", neighbors)
    cached_neighbors = manager.get_cached_neighbors("vec1")
    assert cached_neighbors is not None, "Neighbors should be cached"
    print("✓ Neighborhood cache integration works")
    
    # Test hot vector cache
    vector_data = [1.0, 2.0, 3.0]
    manager.cache_vector("vec1", vector_data)
    cached_vec = manager.get_cached_vector("vec1")
    assert cached_vec is not None, "Vector should be cached"
    assert cached_vec == vector_data, "Cached vector should match"
    print("✓ Hot vector cache integration works")
    
    # Test invalidation
    manager.invalidate_vector("vec1")
    assert manager.get_cached_vector("vec1") is None, "Invalidated vector should not be cached"
    assert manager.get_cached_neighbors("vec1") is None, "Invalidated neighbors should not be cached"
    # Query cache is not invalidated (it's query-based, not vector-based)
    print("✓ Vector invalidation works correctly")
    
    # Test stats
    stats = manager.get_stats()
    assert 'query_cache' in stats, "Stats should include query_cache"
    assert 'neighborhood_cache' in stats, "Stats should include neighborhood_cache"
    assert 'hot_vector_cache' in stats, "Stats should include hot_vector_cache"
    print("✓ Cache manager statistics work correctly")
    
    # Test clear all
    manager.clear_all()
    assert manager.get_cached_query(query_vec, top_k=10, metric="cosine") is None, \
           "All caches should be cleared"
    print("✓ Clear all caches works correctly")
    
    print("\n✓ All Cache Manager tests passed!")


def test_fastvecdb_cache_integration():
    """Test cache integration with FastVecDB API."""
    print("\n" + "=" * 60)
    print("Testing FastVecDB Cache Integration")
    print("=" * 60)
    
    # Create database with caching enabled
    db = FastVecDB(
        storage_path="./test_cache_data",
        dimension=4,
        similarity_metric=SimilarityMetric.COSINE,
        enable_cache=True,
        query_cache_size=10,
        neighborhood_cache_size=5,
        hot_vector_cache_size=10
    )
    
    # Insert vectors
    vectors = [
        ([1.0, 0.0, 0.0, 0.0], "vec1"),
        ([0.0, 1.0, 0.0, 0.0], "vec2"),
        ([0.0, 0.0, 1.0, 0.0], "vec3"),
    ]
    
    for vector, vec_id in vectors:
        db.insert(vec_id, vector, metadata={"name": vec_id})
    
    print("✓ Vectors inserted")
    
    # Test query cache - first query (cold)
    query_vec = [1.0, 0.0, 0.0, 0.0]
    start_time = time.time()
    results1 = db.search(query_vec, top_k=2)
    time1 = time.time() - start_time
    
    # Second query (warm - should hit cache)
    start_time = time.time()
    results2 = db.search(query_vec, top_k=2)
    time2 = time.time() - start_time
    
    assert len(results1) == len(results2), "Results should match"
    assert results1[0]['id'] == results2[0]['id'], "Results should be identical"
    assert time2 < time1, "Cached query should be faster"
    print(f"✓ Query cache works (cold: {time1*1000:.2f}ms, warm: {time2*1000:.2f}ms)")
    
    # Test hot vector cache - first get (may load from storage)
    vec1 = db.get("vec1")
    assert vec1 is not None, "Should retrieve vector"
    
    # Second get (should hit cache)
    vec1_cached = db.get("vec1")
    assert vec1_cached is not None, "Should retrieve from cache"
    assert vec1['vector'] == vec1_cached['vector'], "Cached vector should match"
    print("✓ Hot vector cache works")
    
    # Test cache invalidation on update
    db.update("vec1", metadata={"updated": True})
    # Vector should be re-cached after update
    updated_vec = db.get("vec1")
    assert updated_vec['metadata']['updated'] == True, "Updated vector should reflect changes"
    print("✓ Cache invalidation on update works")
    
    # Test cache invalidation on delete
    db.delete("vec1")
    deleted_vec = db.get("vec1")
    assert deleted_vec is None, "Deleted vector should not be retrievable"
    print("✓ Cache invalidation on delete works")
    
    # Test cache statistics
    stats = db.get_stats()
    assert stats['cache_enabled'] == True, "Cache should be enabled"
    assert 'cache_stats' in stats, "Stats should include cache information"
    print("✓ Cache statistics available in DB stats")
    
    # Test with cache disabled
    db_no_cache = FastVecDB(
        storage_path="./test_cache_data_no_cache",
        dimension=4,
        enable_cache=False
    )
    db_no_cache.insert("vec1", [1.0, 0.0, 0.0, 0.0])
    stats_no_cache = db_no_cache.get_stats()
    assert stats_no_cache['cache_enabled'] == False, "Cache should be disabled"
    print("✓ Cache can be disabled")
    
    # Cleanup
    db.close()
    db_no_cache.close()
    
    print("\n✓ All FastVecDB cache integration tests passed!")


def test_cache_performance():
    """Test cache performance benefits."""
    print("\n" + "=" * 60)
    print("Testing Cache Performance")
    print("=" * 60)
    
    db = FastVecDB(
        storage_path="./test_cache_perf",
        dimension=64,
        enable_cache=True
    )
    
    # Insert multiple vectors
    num_vectors = 100
    for i in range(num_vectors):
        vector = [float(i % 10)] * 64  # Create some similarity
        db.insert(f"vec_{i}", vector)
    
    print(f"✓ Inserted {num_vectors} vectors")
    
    # Test repeated queries
    query_vec = [5.0] * 64
    num_queries = 10
    
    # First query (cold)
    start = time.time()
    results1 = db.search(query_vec, top_k=10)
    time_cold = time.time() - start
    
    # Subsequent queries (warm)
    times_warm = []
    for _ in range(num_queries - 1):
        start = time.time()
        results = db.search(query_vec, top_k=10)
        times_warm.append(time.time() - start)
    
    avg_warm = sum(times_warm) / len(times_warm)
    speedup = time_cold / avg_warm if avg_warm > 0 else 1.0
    
    print(f"  Cold query: {time_cold*1000:.2f}ms")
    print(f"  Warm query (avg): {avg_warm*1000:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    
    assert speedup >= 1.0, "Cached queries should be at least as fast"
    print("✓ Cache provides performance benefits")
    
    # Test cache hit rate with identical queries
    cache_stats = db.get_stats().get('cache_stats', {})
    query_cache_stats = cache_stats.get('query_cache', {})
    print(f"  Query cache size: {query_cache_stats.get('size', 0)}")
    
    db.close()
    print("\n✓ Cache performance tests passed!")


def main():
    """Run all cache tests."""
    print("=" * 60)
    print("FastVecDB Cache Implementation Test Suite")
    print("=" * 60)
    
    test_results = []
    
    try:
        test_query_cache()
        test_results.append(("Query Cache", True))
    except Exception as e:
        print(f"\n✗ Query Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Query Cache", False))
    
    try:
        test_neighborhood_cache()
        test_results.append(("Neighborhood Cache", True))
    except Exception as e:
        print(f"\n✗ Neighborhood Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Neighborhood Cache", False))
    
    try:
        test_hot_vector_cache()
        test_results.append(("Hot Vector Cache", True))
    except Exception as e:
        print(f"\n✗ Hot Vector Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Hot Vector Cache", False))
    
    try:
        test_cache_manager()
        test_results.append(("Cache Manager", True))
    except Exception as e:
        print(f"\n✗ Cache Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Cache Manager", False))
    
    try:
        test_fastvecdb_cache_integration()
        test_results.append(("FastVecDB Integration", True))
    except Exception as e:
        print(f"\n✗ FastVecDB Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("FastVecDB Integration", False))
    
    try:
        test_cache_performance()
        test_results.append(("Cache Performance", True))
    except Exception as e:
        print(f"\n✗ Cache Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Cache Performance", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in test_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s}: {status}")
    
    all_passed = all(result[1] for result in test_results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All cache tests passed!")
    else:
        print("✗ Some cache tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

