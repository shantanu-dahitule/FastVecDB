"""
Example RAG (Retrieval-Augmented Generation) pipeline using FastVecDB.
"""

from fastvecdb import FastVecDB, SimilarityMetric


def simple_embedding(text: str, dimension: int = 128) -> list:
    """
    Simple text embedding function.
    
    In production, you would use a proper embedding model like:
    - OpenAI embeddings
    - Sentence transformers
    - HuggingFace models
    
    This is just a placeholder for demonstration.
    """
    # Simple hash-based embedding (not recommended for production!)
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to vector
    vector = []
    for i in range(dimension):
        byte_val = hash_bytes[i % len(hash_bytes)]
        vector.append((byte_val / 255.0) * 2 - 1)  # Normalize to [-1, 1]
    
    # Normalize
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def main():
    # Knowledge base documents
    documents = [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language known for its simplicity and readability.",
            "category": "programming"
        },
        {
            "id": "doc2",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "category": "ai"
        },
        {
            "id": "doc3",
            "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors.",
            "category": "databases"
        },
        {
            "id": "doc4",
            "text": "FastVecDB is a pure-Python vector search framework with intelligent caching.",
            "category": "databases"
        },
        {
            "id": "doc5",
            "text": "RAG combines retrieval of relevant documents with language model generation.",
            "category": "ai"
        },
    ]
    
    print("Initializing FastVecDB for RAG pipeline...")
    db = FastVecDB(
        storage_path="./rag_example_data",
        dimension=128,
        similarity_metric=SimilarityMetric.COSINE
    )
    
    # Index documents
    print("\nIndexing documents...")
    for doc in documents:
        embedding = simple_embedding(doc["text"], dimension=128)
        db.insert(
            vector_id=doc["id"],
            vector=embedding,
            metadata={
                "text": doc["text"],
                "category": doc["category"]
            },
            bucket_id="knowledge_base"
        )
        print(f"  Indexed: {doc['id']}")
    
    # Query
    print("\n" + "="*60)
    print("RAG Query Pipeline")
    print("="*60)
    
    queries = [
        "What is Python?",
        "Tell me about vector databases",
        "How does machine learning work?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Generate query embedding
        query_embedding = simple_embedding(query, dimension=128)
        
        # Retrieve relevant documents
        results = db.search(
            query_vector=query_embedding,
            top_k=3,
            bucket_ids=["knowledge_base"]
        )
        
        print(f"Retrieved {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['id']} (similarity: {result['score']:.4f})")
            print(f"   Category: {result['metadata']['category']}")
            print(f"   Text: {result['metadata']['text']}")
        
        # In a real RAG pipeline, you would:
        # 1. Retrieve top-k documents (done above)
        # 2. Format context from retrieved documents
        # 3. Send context + query to LLM
        # 4. Return LLM response
    
    # Demonstrate cache benefits
    print("\n" + "="*60)
    print("Cache Performance Test")
    print("="*60)
    
    import time
    
    query_embedding = simple_embedding("Python programming", dimension=128)
    
    # First query (cold)
    start = time.time()
    results1 = db.search(query_embedding, top_k=5)
    time1 = time.time() - start
    
    # Second query (warm - should hit cache)
    start = time.time()
    results2 = db.search(query_embedding, top_k=5)
    time2 = time.time() - start
    
    print(f"First query (cold): {time1*1000:.2f}ms")
    print(f"Second query (warm): {time2*1000:.2f}ms")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Get stats
    stats = db.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Total documents: {stats['total_vectors']}")
    if stats.get('cache_stats'):
        print(f"  Cache: {stats['cache_stats']}")
    
    db.close()
    print("\nRAG pipeline example complete!")


if __name__ == "__main__":
    main()

