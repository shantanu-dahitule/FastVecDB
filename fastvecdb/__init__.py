"""
FastVecDB - Pure-Python vector search and retrieval framework.

A framework designed for simplicity, intelligent caching, and real-world performance,
with zero third-party dependencies.
"""

from fastvecdb.api import FastVecDB
from fastvecdb.core.similarity import SimilarityMetric

__version__ = "1.0.0"
__all__ = ["FastVecDB", "SimilarityMetric"]

