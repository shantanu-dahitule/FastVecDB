"""
Basic FastVecDB usage example.
"""

from fastvecdb import FastVecDB, SimilarityMetric


def main():
    # Initialize database
    print("Initializing FastVecDB...")
    db = FastVecDB(
        storage_path="./example_data",
        dimension=128,
        similarity_metric=SimilarityMetric.COSINE
    )
    
    # Insert some vectors
    print("\nInserting vectors...")
    vectors = [
        ([0.1] * 128, "doc1", {"title": "Python Tutorial", "category": "programming"}),
        ([0.2] * 128, "doc2", {"title": "JavaScript Guide", "category": "programming"}),
        ([0.3] * 128, "doc3", {"title": "Machine Learning Basics", "category": "ai"}),
        ([0.4] * 128, "doc4", {"title": "Deep Learning Advanced", "category": "ai"}),
        ([0.5] * 128, "doc5", {"title": "Data Science Handbook", "category": "data"}),
    ]
    
    for vector, doc_id, metadata in vectors:
        db.insert(
            vector_id=doc_id,
            vector=vector,
            metadata=metadata
        )
        print(f"  Inserted {doc_id}")
    
    # Search
    print("\nSearching for similar vectors...")
    query_vector = [0.15] * 128  # Similar to doc1
    results = db.search(query_vector, top_k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. ID: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Metadata: {result['metadata']}")
    
    # Get a specific vector
    print("\n\nRetrieving specific vector...")
    doc = db.get("doc1")
    if doc:
        print(f"Found: {doc['id']}")
        print(f"Metadata: {doc['metadata']}")
    
    # Update a vector
    print("\n\nUpdating vector metadata...")
    db.update("doc1", metadata={"title": "Updated Python Tutorial", "version": 2})
    updated_doc = db.get("doc1")
    print(f"Updated metadata: {updated_doc['metadata']}")
    
    # List all vectors
    print("\n\nAll vectors in database:")
    vector_ids = db.list_vectors()
    for vid in vector_ids:
        print(f"  - {vid}")
    
    # Get statistics
    print("\n\nDatabase statistics:")
    stats = db.get_stats()
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Similarity metric: {stats['similarity_metric']}")
    print(f"Buckets: {stats['buckets']}")
    if stats.get('cache_stats'):
        print(f"Cache stats: {stats['cache_stats']}")
    
    # Close database
    print("\n\nClosing database...")
    db.close()
    print("Done!")


if __name__ == "__main__":
    main()

