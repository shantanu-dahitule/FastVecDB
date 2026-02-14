# FastVecDB

FastVecDB is a pure-Python vector search and retrieval framework designed for simplicity, intelligent caching, and real-world performance, with zero third-party dependencies.

It is built for teams who want vector search without operational complexity, external services, or heavy native dependencies.

## 🚀 Why FastVecDB?

Most vector databases focus on raw mathematical speed and require:

- Native extensions (C++ / CUDA)
- External services
- Complex deployments
- Heavy infrastructure

FastVecDB takes a different approach.

FastVecDB optimizes for real application workloads, where:

- Queries repeat or are semantically similar
- Predictable latency matters more than peak throughput
- Deployment simplicity is critical
- External dependencies are restricted

## ✨ Key Features

✅ **Pure Python** (Standard Library Only)  
✅ **Zero Dependencies**  
✅ **pip installable**  
✅ **Intelligent multi-layer caching**  
✅ **Incremental inserts & updates**  
✅ **Persistent storage** (no external DB)  
✅ **Framework-level extensibility**  
✅ **Commercial-friendly licensing**

## 📦 Installation

```bash
pip install fastvecdb
```

**No Docker.**  
**No Redis.**  
**No NumPy.**  
**No GPUs.**

## 🧠 Design Philosophy

FastVecDB is a framework, not just a vector index.

It prioritizes:

- Ease of adoption
- Predictable performance
- Cache-aware retrieval
- Operational simplicity
- Enterprise compatibility

It does not attempt to compete on brute-force vector math speed with FAISS or GPU-based systems.

## 🏗 Architecture Overview

```
Application
    |
FastVecDB API
    |
Query Router
    |
Multi-Layer Cache
    |        \
Vector Buckets   Result Cache
    |
Pure-Python Similarity Engine
    |
Persistent Storage (SQLite / mmap / pickle)
```

## 🔍 Supported Similarity Metrics

- **Cosine Similarity**
- **Dot Product**
- **Euclidean (L2) Distance**

All implemented using Python standard library only.

## ⚡ Intelligent Caching (Core Differentiator)

FastVecDB includes built-in caching strategies that most vector databases lack:

### Cache Types

- **Query Result Cache** - Caches complete query results
- **Semantic Neighborhood Cache** - Caches vectors that are semantically similar
- **Hot Vector Cache** - LRU cache for frequently accessed vectors

### Benefits

- Faster repeated queries
- Lower CPU usage
- Predictable latency
- Excellent performance for RAG pipelines

In real applications, caching often matters more than raw ANN speed.

## 📂 Persistence & Storage

FastVecDB does not require any external database.

Supported persistence backends:

- **sqlite3** for metadata
- **pickle** for snapshots
- **mmap** for memory-mapped vector blocks

## 📊 Performance Philosophy

FastVecDB focuses on end-to-end latency, not microbenchmarks.

Performance gains come from:

- Smart caching
- Reduced search space
- Efficient memory usage
- Eliminating network hops

## 🚀 Quick Start

### Basic Usage

```python
from fastvecdb import FastVecDB, SimilarityMetric

# Initialize database
db = FastVecDB(
    storage_path="./my_vectors",
    dimension=128,
    similarity_metric=SimilarityMetric.COSINE
)

# Insert vectors
db.insert(
    vector_id="doc1",
    vector=[0.1, 0.2, 0.3, ...],  # Your 128-dim vector
    metadata={"title": "Document 1", "category": "tech"}
)

# Search
results = db.search(
    query_vector=[0.15, 0.25, 0.35, ...],
    top_k=10
)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}")
    print(f"Metadata: {result['metadata']}")

# Close database
db.close()
```

### Using Different Similarity Metrics

```python
from fastvecdb import FastVecDB, SimilarityMetric

# Cosine similarity (default)
db = FastVecDB(dimension=128, similarity_metric=SimilarityMetric.COSINE)

# Dot product
db = FastVecDB(dimension=128, similarity_metric=SimilarityMetric.DOT_PRODUCT)

# Euclidean distance
db = FastVecDB(dimension=128, similarity_metric=SimilarityMetric.EUCLIDEAN)

# Or specify per query
results = db.search(
    query_vector=[...],
    top_k=10,
    metric=SimilarityMetric.EUCLIDEAN
)
```

### Working with Buckets

```python
# Insert vectors into specific buckets
db.insert("doc1", vector=[...], bucket_id="tech_docs")
db.insert("doc2", vector=[...], bucket_id="tech_docs")
db.insert("doc3", vector=[...], bucket_id="science_docs")

# Search specific buckets
results = db.search(
    query_vector=[...],
    top_k=10,
    bucket_ids=["tech_docs"]  # Only search tech documents
)
```

### Updating Vectors

```python
# Update metadata
db.update("doc1", metadata={"title": "Updated Title", "version": 2})

# Update vector and metadata
db.update(
    "doc1",
    vector=[0.2, 0.3, 0.4, ...],  # New vector
    metadata={"updated": True}
)
```

### Managing Cache

```python
# Get cache statistics
stats = db.get_stats()
print(stats['cache_stats'])

# Clear all caches
db.clear_cache()
```

### Context Manager

```python
# Use as context manager for automatic cleanup
with FastVecDB(dimension=128) as db:
    db.insert("doc1", vector=[...])
    results = db.search(query_vector=[...], top_k=10)
    # Automatically closed on exit
```

## 📖 API Reference

### FastVecDB

#### `__init__(storage_path, dimension, similarity_metric, enable_cache, ...)`

Initialize FastVecDB instance.

**Parameters:**
- `storage_path` (str): Path to storage directory (default: "./fastvecdb_data")
- `dimension` (int, optional): Expected vector dimension
- `similarity_metric` (SimilarityMetric): Default similarity metric (default: COSINE)
- `enable_cache` (bool): Enable caching (default: True)
- `query_cache_size` (int): Max query cache size (default: 1000)
- `neighborhood_cache_size` (int): Max neighbors per vector (default: 50)
- `hot_vector_cache_size` (int): Max hot vector cache size (default: 1000)

#### `insert(vector_id, vector, metadata=None, bucket_id=None)`

Insert a vector into the database.

#### `update(vector_id, vector=None, metadata=None)`

Update an existing vector.

#### `delete(vector_id)`

Delete a vector from the database.

#### `search(query_vector, top_k=10, metric=None, bucket_ids=None, threshold=None)`

Search for similar vectors.

**Returns:** List of dictionaries with 'id', 'score', and 'metadata'

#### `get(vector_id)`

Get a vector by ID.

**Returns:** Dictionary with 'id', 'vector', and 'metadata', or None

#### `list_vectors()`

List all vector IDs in the database.

#### `get_stats()`

Get database statistics.

#### `clear_cache()`

Clear all caches.

#### `close()`

Close the database and clean up resources.

## 🎯 Use Cases

- **RAG Pipelines** - Semantic search for retrieval-augmented generation
- **Document Search** - Find similar documents by embedding
- **Recommendation Systems** - Find similar items
- **Deduplication** - Find duplicate or near-duplicate content
- **Embedding Storage** - Persistent storage for ML embeddings

## 🔧 Configuration

### Cache Configuration

```python
db = FastVecDB(
    dimension=128,
    enable_cache=True,
    query_cache_size=2000,        # More cached queries
    neighborhood_cache_size=100,   # More neighbors cached
    hot_vector_cache_size=5000     # More vectors in memory
)
```

### Disable Caching

```python
db = FastVecDB(
    dimension=128,
    enable_cache=False  # Disable all caching
)
```

## 📝 Examples

See the `examples/` directory for more detailed examples:

- `cache_quick_demo.py` - **Quick cache performance demo** (start here!)
- `cache_performance_demo.py` - **Comprehensive cache showcase** (key selling point!)
- `basic_usage.py` - Basic operations
- `rag_pipeline.py` - RAG pipeline example
- `batch_insert.py` - Batch insertion
- `similarity_metrics.py` - Comparing similarity metrics

### 🚀 Cache Performance Examples

FastVecDB's intelligent caching is a key differentiator. Try these examples to see the performance benefits:

```bash
# Quick demo (simple, fast)
python examples/cache_quick_demo.py

# Comprehensive demo (detailed, real-world scenarios)
python examples/cache_performance_demo.py
```

These demos showcase:
- **Query Result Cache**: Instant repeated queries
- **Hot Vector Cache**: Fast access to popular content
- **Neighborhood Cache**: Efficient similar searches
- **Real-world performance**: Measurable speedups in production scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Commercial-friendly license (check LICENSE file for details).

## Acknowledgments

FastVecDB delivers production-ready vector search without the operational overhead of traditional vector databases, offering enterprise-grade features and persistent storage in a fully open-source library.

---

