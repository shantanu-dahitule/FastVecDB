"""
Storage backends for persistent vector storage.

Supports:
- SQLite for metadata
- Pickle for snapshots
- mmap for memory-mapped vector blocks
"""

import pickle
import sqlite3
import mmap
import os
import struct
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path


class StorageBackend:
    """Base class for storage backends."""
    
    def save_vector(self, vector_id: str, vector: List[float], metadata: Dict) -> None:
        """Save a vector."""
        raise NotImplementedError
    
    def load_vector(self, vector_id: str) -> Tuple[List[float], Dict]:
        """Load a vector."""
        raise NotImplementedError
    
    def delete_vector(self, vector_id: str) -> None:
        """Delete a vector."""
        raise NotImplementedError
    
    def list_vectors(self) -> List[str]:
        """List all vector IDs."""
        raise NotImplementedError
    
    def close(self) -> None:
        """Close the storage backend."""
        pass


class SQLiteStorage(StorageBackend):
    """SQLite-based storage for metadata and vector references."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                metadata TEXT,
                bucket_id TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bucket_id ON vectors(bucket_id)
        """)
        self.conn.commit()
    
    def save_vector(self, vector_id: str, vector: List[float], metadata: Dict, bucket_id: Optional[str] = None) -> None:
        """Save vector metadata to SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO vectors (id, dimension, metadata, bucket_id)
            VALUES (?, ?, ?, ?)
        """, (vector_id, len(vector), json.dumps(metadata), bucket_id))
        self.conn.commit()
    
    def load_metadata(self, vector_id: str) -> Optional[Dict]:
        """Load vector metadata from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT metadata FROM vectors WHERE id = ?", (vector_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def delete_vector(self, vector_id: str) -> None:
        """Delete vector from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
        self.conn.commit()
    
    def list_vectors(self) -> List[str]:
        """List all vector IDs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM vectors")
        return [row[0] for row in cursor.fetchall()]
    
    def get_bucket_vectors(self, bucket_id: str) -> List[str]:
        """Get all vector IDs in a bucket."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM vectors WHERE bucket_id = ?", (bucket_id,))
        return [row[0] for row in cursor.fetchall()]
    
    def close(self) -> None:
        """Close SQLite connection."""
        if self.conn:
            self.conn.close()


class MmapVectorStorage:
    """Memory-mapped storage for vector data."""
    
    def __init__(self, file_path: str, dimension: int):
        """
        Initialize memory-mapped vector storage.
        
        Args:
            file_path: Path to memory-mapped file
            dimension: Dimension of vectors
        """
        self.file_path = file_path
        self.dimension = dimension
        self.vector_size = dimension * 8  # 8 bytes per float64
        self.file = None
        self.mmap = None
        self._ensure_file()
    
    def _ensure_file(self) -> None:
        """Ensure the memory-mapped file exists."""
        if not os.path.exists(self.file_path):
            # Create file with initial size
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'wb') as f:
                # Write header: dimension (4 bytes) + count (4 bytes)
                f.write(struct.pack('II', self.dimension, 0))
                f.write(b'\x00' * (1024 * 1024))  # 1MB initial size
        
        self.file = open(self.file_path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
    
    def write_vector(self, index: int, vector: List[float]) -> None:
        """
        Write a vector at the given index.
        
        Args:
            index: Index position (0-based)
            vector: Vector data
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {len(vector)} != {self.dimension}")
        
        offset = 8 + (index * self.vector_size)  # Skip header
        if offset + self.vector_size > len(self.mmap):
            # Extend file
            self.mmap.resize(offset + self.vector_size)
        
        # Write vector as binary
        for i, val in enumerate(vector):
            self.mmap[offset + i * 8:offset + (i + 1) * 8] = struct.pack('d', val)
        self.mmap.flush()
    
    def read_vector(self, index: int) -> List[float]:
        """
        Read a vector at the given index.
        
        Args:
            index: Index position (0-based)
            
        Returns:
            Vector data
        """
        offset = 8 + (index * self.vector_size)
        if offset + self.vector_size > len(self.mmap):
            raise IndexError(f"Vector index out of range: {index}")
        
        vector = []
        for i in range(self.dimension):
            val_bytes = self.mmap[offset + i * 8:offset + (i + 1) * 8]
            vector.append(struct.unpack('d', val_bytes)[0])
        return vector
    
    def close(self) -> None:
        """Close memory-mapped file."""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()


class PickleSnapshotStorage:
    """Pickle-based snapshot storage for full database state."""
    
    def __init__(self, snapshot_path: str):
        """
        Initialize pickle snapshot storage.
        
        Args:
            snapshot_path: Path to snapshot file
        """
        self.snapshot_path = snapshot_path
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(self, data: Any) -> None:
        """
        Save a snapshot of the data.
        
        Args:
            data: Data to save
        """
        with open(self.snapshot_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_snapshot(self) -> Any:
        """
        Load a snapshot.
        
        Returns:
            Loaded data
        """
        if not os.path.exists(self.snapshot_path):
            return None
        
        with open(self.snapshot_path, 'rb') as f:
            return pickle.load(f)

