"""
Example demonstrating different similarity metrics.
"""

from fastvecdb import FastVecDB, SimilarityMetric
import random


def generate_random_vector(dim: int) -> list:
    """Generate a random normalized vector."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def main():
    dimension = 64
    
    # Create vectors
    print("Creating test vectors...")
    vectors = [
        (generate_random_vector(dimension), f"vec_{i}")
        for i in range(10)
    ]
    
    # Test each similarity metric
    metrics = [
        SimilarityMetric.COSINE,
        SimilarityMetric.DOT_PRODUCT,
        SimilarityMetric.EUCLIDEAN
    ]
    
    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"Testing {metric.value.upper()} similarity")
        print('='*60)
        
        # Create database with this metric
        db = FastVecDB(
            storage_path=f"./example_data_{metric.value}",
            dimension=dimension,
            similarity_metric=metric
        )
        
        # Insert vectors
        for vector, vec_id in vectors:
            db.insert(vec_id, vector, metadata={"metric": metric.value})
        
        # Search with query vector
        query_vector = generate_random_vector(dimension)
        results = db.search(query_vector, top_k=5)
        
        print(f"\nTop 5 results for query:")
        for i, result in enumerate(results, 1):
            score = result['score']
            if metric == SimilarityMetric.EUCLIDEAN:
                # For euclidean, we negated the distance for sorting
                distance = -score
                print(f"{i}. {result['id']}: distance={distance:.4f}")
            else:
                print(f"{i}. {result['id']}: similarity={score:.4f}")
        
        # Clean up
        db.close()
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
    print("\nNote:")
    print("- Cosine similarity: Range [-1, 1], higher = more similar")
    print("- Dot product: Range depends on vector norms, higher = more similar")
    print("- Euclidean distance: Range [0, ∞), lower = more similar")


if __name__ == "__main__":
    main()

