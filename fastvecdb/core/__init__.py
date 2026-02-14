"""Core vector operations and similarity metrics."""

from fastvecdb.core.similarity import SimilarityMetric, cosine_similarity, dot_product, euclidean_distance
from fastvecdb.core.vector import Vector, normalize_vector

__all__ = [
    "SimilarityMetric",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "Vector",
    "normalize_vector",
]

