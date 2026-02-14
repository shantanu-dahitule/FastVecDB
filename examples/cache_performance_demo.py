"""
FastVecDB Cache Performance Demo

This example showcases the intelligent caching system - a key differentiator
that makes FastVecDB perform exceptionally well for real-world applications.

The cache provides:
- Instant repeated queries (query result cache)
- Faster similar searches (neighborhood cache)
- Reduced disk I/O (hot vector cache)
"""

import time
import random
from fastvecdb import FastVecDB, SimilarityMetric


def generate_embedding(text: str, dimension: int = 128) -> list:
    """
    Simple text-to-embedding function.
    
    In production, use proper embedding models like:
    - OpenAI text-embedding-ada-002
    - Sentence Transformers
    - HuggingFace models
    """
    # Simple deterministic embedding based on text hash
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    vector = []
    for i in range(dimension):
        byte_val = hash_bytes[i % len(hash_bytes)]
        vector.append((byte_val / 255.0) * 2 - 1)
    
    # Normalize
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print("-" * 70)


def demo_cache_benefits():
    """
    Demonstrate cache performance benefits with a realistic scenario.
    
    Scenario: Document search system where users frequently search for
    similar documents (common in RAG pipelines, recommendation systems, etc.)
    """
    
    print("\n" + "=" * 70)
    print("  FastVecDB Cache Performance Demonstration")
    print("  Key Selling Point: Intelligent Multi-Layer Caching")
    print("=" * 70)
    
    # Initialize database with caching enabled
    print("\n📦 Initializing FastVecDB with intelligent caching...")
    db = FastVecDB(
        storage_path="./cache_demo_data",
        dimension=128,
        similarity_metric=SimilarityMetric.COSINE,
        enable_cache=True,
        query_cache_size=100,      # Cache up to 100 query results
        neighborhood_cache_size=20, # Cache 20 neighbors per vector
        hot_vector_cache_size=200  # Keep 200 hot vectors in memory
    )
    
    # Simulate a document database
    print("\n📚 Indexing documents...")
    documents = [
        "Python is a high-level programming language",
        "Machine learning enables computers to learn from data",
        "Vector databases store high-dimensional embeddings",
        "FastVecDB provides intelligent caching for fast searches",
        "RAG combines retrieval with language model generation",
        "Embeddings represent text as numerical vectors",
        "Similarity search finds related content quickly",
        "Caching reduces computation and improves latency",
        "Python supports object-oriented programming",
        "Databases store and retrieve structured information",
    ]
    
    # Add more documents for realistic scale
    for i in range(90):
        documents.append(f"Document {i+11}: Technical content about data science and AI")
    
    print(f"  Adding {len(documents)} documents to the database...")
    start = time.time()
    for i, doc in enumerate(documents):
        embedding = generate_embedding(doc, dimension=128)
        db.insert(
            vector_id=f"doc_{i+1}",
            vector=embedding,
            metadata={"text": doc, "index": i+1}
        )
    insert_time = time.time() - start
    print(f"  ✓ Indexed {len(documents)} documents in {insert_time:.2f}s")
    
    # Get initial stats
    stats = db.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    print(f"  Cache enabled: {stats['cache_enabled']}")
    
    # ========================================================================
    # DEMO 1: Query Result Cache - Repeated Queries
    # ========================================================================
    print_separator("DEMO 1: Query Result Cache - Repeated Queries")
    
    print("\n🔍 Scenario: User searches for 'Python programming' multiple times")
    print("   (Common in web applications, chatbots, RAG systems)")
    
    query_text = "Python programming language"
    query_vector = generate_embedding(query_text, dimension=128)
    
    # First query (COLD - no cache)
    print("\n  First Query (COLD - no cache):")
    start = time.time()
    results1 = db.search(query_vector, top_k=5)
    cold_time = time.time() - start
    
    print(f"    ⏱️  Time: {cold_time*1000:.2f}ms")
    print(f"    📄 Results: {len(results1)} documents found")
    for i, result in enumerate(results1[:3], 1):
        print(f"       {i}. {result['metadata']['text'][:50]}... (score: {result['score']:.3f})")
    
    # Repeated queries (WARM - cache hit)
    print("\n  Repeated Queries (WARM - cache hit):")
    warm_times = []
    for i in range(5):
        start = time.time()
        results = db.search(query_vector, top_k=5)
        warm_times.append(time.time() - start)
    
    avg_warm = sum(warm_times) / len(warm_times)
    speedup = cold_time / avg_warm
    
    print(f"    ⏱️  Average time: {avg_warm*1000:.2f}ms")
    print(f"    🚀 Speedup: {speedup:.1f}x faster!")
    print(f"    💡 Benefit: Instant results for repeated queries")
    
    # ========================================================================
    # DEMO 2: Similar Query Cache - Semantic Similarity
    # ========================================================================
    print_separator("DEMO 2: Hot Vector Cache - Frequent Access")
    
    print("\n🔍 Scenario: Frequently accessed documents (popular content)")
    print("   (Common in recommendation systems, trending content)")
    
    # Access same vector multiple times
    doc_id = "doc_1"
    
    print(f"\n  First access to '{doc_id}' (may load from disk):")
    start = time.time()
    doc1 = db.get(doc_id)
    first_access = time.time() - start
    print(f"    ⏱️  Time: {first_access*1000:.2f}ms")
    
    print(f"\n  Subsequent accesses (from hot vector cache):")
    cached_times = []
    for i in range(10):
        start = time.time()
        doc = db.get(doc_id)
        cached_times.append(time.time() - start)
    
    avg_cached = sum(cached_times) / len(cached_times)
    cache_speedup = first_access / avg_cached if avg_cached > 0 else 1.0
    
    print(f"    ⏱️  Average time: {avg_cached*1000:.2f}ms")
    print(f"    🚀 Speedup: {cache_speedup:.1f}x faster!")
    print(f"    💡 Benefit: Popular content served instantly from memory")
    
    # ========================================================================
    # DEMO 3: Real-World Workload - Mixed Query Pattern
    # ========================================================================
    print_separator("DEMO 3: Real-World Workload Simulation")
    
    print("\n🔍 Scenario: Simulating realistic query patterns")
    print("   - Some queries repeat (cache hits)")
    print("   - Some queries are similar (partial cache benefits)")
    print("   - Some queries are new (cache misses)")
    
    # Create query pattern: 60% repeated, 30% similar, 10% new
    query_patterns = [
        ("Python programming", 5),      # Repeated 5 times
        ("machine learning AI", 4),     # Repeated 4 times
        ("vector database search", 3),   # Repeated 3 times
        ("caching performance", 2),     # Repeated 2 times
        ("new unique query 1", 1),      # New query
        ("new unique query 2", 1),      # New query
    ]
    
    print("\n  Running mixed query workload...")
    total_queries = sum(count for _, count in query_patterns)
    query_times = []
    
    for query_text, count in query_patterns:
        query_vec = generate_embedding(query_text, dimension=128)
        for _ in range(count):
            start = time.time()
            results = db.search(query_vec, top_k=5)
            query_times.append(time.time() - start)
    
    avg_time = sum(query_times) / len(query_times)
    total_time = sum(query_times)
    
    print(f"\n  Results:")
    print(f"    📊 Total queries: {total_queries}")
    print(f"    ⏱️  Average query time: {avg_time*1000:.2f}ms")
    print(f"    ⏱️  Total time: {total_time*1000:.2f}ms")
    print(f"    💡 Benefit: Cached queries are near-instant, improving overall throughput")
    
    # Show cache statistics
    stats = db.get_stats()
    cache_stats = stats.get('cache_stats', {})
    
    print_separator("Cache Statistics")
    print("\n📈 Current Cache State:")
    
    if cache_stats:
        query_cache = cache_stats.get('query_cache', {})
        hot_cache = cache_stats.get('hot_vector_cache', {})
        neighborhood_cache = cache_stats.get('neighborhood_cache', {})
        
        print(f"\n  Query Result Cache:")
        print(f"    Cached queries: {query_cache.get('size', 0)}/{query_cache.get('max_size', 0)}")
        
        print(f"\n  Hot Vector Cache:")
        print(f"    Cached vectors: {hot_cache.get('size', 0)}/{hot_cache.get('max_size', 0)}")
        
        print(f"\n  Neighborhood Cache:")
        print(f"    Cached neighborhoods: {neighborhood_cache.get('cached_vectors', 0)}")
    
    # ========================================================================
    # DEMO 4: Comparison - With vs Without Cache
    # ========================================================================
    print_separator("DEMO 4: Performance Comparison - Cache ON vs OFF")
    
    print("\n🔍 Comparing performance with cache enabled vs disabled")
    
    # Test with cache (already enabled)
    test_query = generate_embedding("Python machine learning", dimension=128)
    
    # Warm up cache
    db.search(test_query, top_k=5)
    
    # Measure with cache
    cache_on_times = []
    for _ in range(10):
        start = time.time()
        db.search(test_query, top_k=5)
        cache_on_times.append(time.time() - start)
    avg_with_cache = sum(cache_on_times) / len(cache_on_times)
    
    # Test without cache
    db_no_cache = FastVecDB(
        storage_path="./cache_demo_data_no_cache",
        dimension=128,
        enable_cache=False  # Cache disabled!
    )
    
    # Insert same documents
    for i, doc in enumerate(documents[:20]):  # Smaller set for comparison
        embedding = generate_embedding(doc, dimension=128)
        db_no_cache.insert(f"doc_{i+1}", embedding, metadata={"text": doc})
    
    # Measure without cache
    cache_off_times = []
    test_query_no_cache = generate_embedding("Python machine learning", dimension=128)
    for _ in range(10):
        start = time.time()
        db_no_cache.search(test_query_no_cache, top_k=5)
        cache_off_times.append(time.time() - start)
    avg_without_cache = sum(cache_off_times) / len(cache_off_times)
    
    improvement = (avg_without_cache / avg_with_cache) if avg_with_cache > 0 else 1.0
    
    print(f"\n  Results (10 repeated queries):")
    print(f"    With Cache:    {avg_with_cache*1000:.2f}ms average")
    print(f"    Without Cache: {avg_without_cache*1000:.2f}ms average")
    print(f"    🚀 Improvement: {improvement:.1f}x faster with cache!")
    
    db_no_cache.close()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Key Takeaways")
    
    print("\n✨ FastVecDB's Intelligent Caching Provides:")
    print("\n  1. ⚡ Query Result Cache")
    print("     → Instant results for repeated queries")
    print("     → Perfect for RAG pipelines, chatbots, web apps")
    print("     → Reduces computation by caching full query results")
    
    print("\n  2. 🔥 Hot Vector Cache")
    print("     → Frequently accessed vectors stay in memory")
    print("     → Eliminates disk I/O for popular content")
    print("     → LRU eviction keeps cache efficient")
    
    print("\n  3. 🌐 Neighborhood Cache")
    print("     → Caches semantically similar vectors")
    print("     → Speeds up similar searches")
    print("     → Reduces redundant similarity calculations")
    
    print("\n  💡 Real-World Impact:")
    print("     → Predictable, low-latency responses")
    print("     → Reduced CPU usage for repeated operations")
    print("     → Better user experience in production")
    print("     → Cost-effective (no external cache services needed)")
    
    print("\n" + "=" * 70)
    print("  Cache Performance Demo Complete!")
    print("=" * 70)
    
    # Cleanup
    db.close()


if __name__ == "__main__":
    try:
        demo_cache_benefits()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()

