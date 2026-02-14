"""
Similarity metrics implemented using pure Python standard library.

Supports:
- Cosine Similarity
- Dot Product
- Euclidean (L2) Distance
"""

import math
from enum import Enum
from typing import List, Tuple


class SimilarityMetric(Enum):
    """Enumeration of supported similarity metrics."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute dot product of two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Dot product value
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} != {len(vec2)}")
    
    return sum(a * b for a, b in zip(vec1, vec2))


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value in range [-1, 1]
        
    Raises:
        ValueError: If vectors have different dimensions or are zero vectors
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} != {len(vec2)}")
    
    dot = dot_product(vec1, vec2)
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cannot compute cosine similarity for zero vectors")
    
    return dot / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (always non-negative)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} != {len(vec2)}")
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def compute_similarity(
    vec1: List[float],
    vec2: List[float],
    metric: SimilarityMetric
) -> float:
    """
    Compute similarity/distance using the specified metric.
    
    For similarity metrics (cosine, dot_product), higher values mean more similar.
    For distance metrics (euclidean), lower values mean more similar.
    
    Args:
        vec1: First vector
        vec2: Second vector
        metric: Similarity metric to use
        
    Returns:
        Similarity or distance value
    """
    if metric == SimilarityMetric.COSINE:
        return cosine_similarity(vec1, vec2)
    elif metric == SimilarityMetric.DOT_PRODUCT:
        return dot_product(vec1, vec2)
    elif metric == SimilarityMetric.EUCLIDEAN:
        return euclidean_distance(vec1, vec2)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")



