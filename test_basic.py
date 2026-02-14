"""
Basic test script to verify FastVecDB installation and functionality.
"""

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from fastvecdb import FastVecDB, SimilarityMetric
        from fastvecdb.core import cosine_similarity, dot_product, euclidean_distance
        from fastvecdb.core import Vector, normalize_vector
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_operations():
    """Test basic database operations."""
    print("\nTesting basic operations...")
    try:
        from fastvecdb import FastVecDB, SimilarityMetric
        
        # Create database
        db = FastVecDB(
            storage_path="./test_data",
            dimension=4,
            similarity_metric=SimilarityMetric.COSINE
        )
        
        # Insert vectors
        db.insert("vec1", [1.0, 0.0, 0.0, 0.0], {"name": "vector1"})
        db.insert("vec2", [0.0, 1.0, 0.0, 0.0], {"name": "vector2"})
        db.insert("vec3", [0.0, 0.0, 1.0, 0.0], {"name": "vector3"})
        print("✓ Insert operations successful")
        
        # Search
        results = db.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) > 0, "Search should return results"
        assert results[0]['id'] == 'vec1', "First result should be vec1"
        print("✓ Search operations successful")
        
        # Get vector
        vec = db.get("vec1")
        assert vec is not None, "Should retrieve vector"
        assert vec['id'] == 'vec1', "Retrieved vector should match"
        print("✓ Get operations successful")
        
        # Update
        db.update("vec1", metadata={"name": "updated_vector1"})
        updated = db.get("vec1")
        assert updated['metadata']['name'] == 'updated_vector1', "Update should work"
        print("✓ Update operations successful")
        
        # List vectors
        vector_ids = db.list_vectors()
        assert len(vector_ids) == 3, "Should have 3 vectors"
        print("✓ List operations successful")
        
        # Get stats
        stats = db.get_stats()
        assert stats['total_vectors'] == 3, "Stats should show 3 vectors"
        print("✓ Stats operations successful")
        
        # Clean up
        db.close()
        print("✓ All basic operations successful")
        return True
        
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity_metrics():
    """Test similarity metrics."""
    print("\nTesting similarity metrics...")
    try:
        from fastvecdb.core import cosine_similarity, dot_product, euclidean_distance
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        # Cosine similarity
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.001, "Identical vectors should have cosine similarity 1.0"
        sim = cosine_similarity(vec1, vec3)
        assert abs(sim - 0.0) < 0.001, "Orthogonal vectors should have cosine similarity 0.0"
        print("✓ Cosine similarity works")
        
        # Dot product
        dot = dot_product(vec1, vec2)
        assert abs(dot - 1.0) < 0.001, "Dot product of unit vectors should be 1.0"
        print("✓ Dot product works")
        
        # Euclidean distance
        dist = euclidean_distance(vec1, vec2)
        assert abs(dist - 0.0) < 0.001, "Identical vectors should have distance 0.0"
        dist = euclidean_distance(vec1, vec3)
        assert abs(dist - 1.414) < 0.01, "Orthogonal unit vectors should have distance sqrt(2)"
        print("✓ Euclidean distance works")
        
        print("✓ All similarity metrics work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Similarity metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FastVecDB Basic Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Similarity Metrics", test_similarity_metrics()))
    results.append(("Basic Operations", test_basic_operations()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

