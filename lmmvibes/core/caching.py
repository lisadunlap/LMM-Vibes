"""
LMDB-based caching for LLM responses and embeddings.
"""

import json
import lmdb
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union

class LMDBCache:
    def __init__(self, cache_dir: str = ".cache/lmmvibes", max_size: int = 1024 * 1024 * 1024):
        """Initialize LMDB cache.
        
        Args:
            cache_dir: Directory to store LMDB files
            max_size: Max size of LMDB in bytes (default 1GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate environments for completions and embeddings
        self.completions_env = lmdb.open(
            str(self.cache_dir / "completions"),
            map_size=max_size,
            subdir=True,
            readonly=False,
            lock=True,
        )
        
        self.embeddings_env = lmdb.open(
            str(self.cache_dir / "embeddings"),
            map_size=max_size,
            subdir=True,
            readonly=False,
            lock=True,
        )

    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from input data."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get_completion(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached completion response."""
        key = self._get_cache_key(request_data).encode()
        with self.completions_env.begin() as txn:
            value = txn.get(key)
            if value is not None:
                return json.loads(value)
        return None

    def set_completion(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """Cache completion response."""
        key = self._get_cache_key(request_data).encode()
        value = json.dumps(response_data).encode()
        with self.completions_env.begin(write=True) as txn:
            txn.put(key, value)

    def get_embedding(self, text: Union[str, list]) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._get_cache_key({"text": text}).encode()
        with self.embeddings_env.begin() as txn:
            value = txn.get(key)
            if value is not None:
                return np.frombuffer(value, dtype=np.float32)
        return None

    def set_embedding(self, text: Union[str, list], embedding: np.ndarray) -> None:
        """Cache embedding."""
        key = self._get_cache_key({"text": text}).encode()
        value = np.array(embedding, dtype=np.float32).tobytes()
        with self.embeddings_env.begin(write=True) as txn:
            txn.put(key, value)

    def close(self):
        """Close LMDB environments."""
        self.completions_env.close()
        self.embeddings_env.close() 