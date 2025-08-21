"""
LMDB-based caching for LLM responses and embeddings.
"""

import json
import lmdb
import hashlib
import numpy as np
import re
from pathlib import Path
from typing import Optional, Dict, Any, Union

def parse_size_string(size_str: str) -> int:
    """Parse a size string like '10GB' or '1TB' into bytes."""
    if isinstance(size_str, int):
        return size_str
    
    # Remove any whitespace and convert to uppercase
    size_str = size_str.strip().upper()
    
    # Match pattern like "10GB", "1.5TB", etc.
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)$', size_str)
    if not match:
        raise ValueError(f"Invalid size string: {size_str}. Expected format like '10GB' or '1TB'")
    
    number = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }
    
    return int(number * multipliers[unit])

class LMDBCache:
    def __init__(self, cache_dir: str = ".cache/stringsight", max_size: Union[str, int] = "10GB", *, lock: Optional[bool] = None, readonly: bool = False):
        """Initialize LMDB cache.
        
        Args:
            cache_dir: Directory to store LMDB files
            max_size: Max size of LMDB in bytes or size string like "10GB" (default 10GB)
            lock: Whether to use LMDB file locks. If None, reads from env `STRINGSIGHT_LMDB_LOCK` (default: True)
            readonly: Open the environments read-only
        """
        import os
        self.cache_dir = Path(os.environ.get("STRINGSIGHT_CACHE_DIR", cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse max_size if it's a string
        if isinstance(max_size, str):
            # Allow override via env var
            env_size = os.environ.get("STRINGSIGHT_LMDB_MAX_SIZE", max_size)
            max_size_bytes = parse_size_string(env_size)
        else:
            max_size_bytes = max_size
        
        # Resolve lock setting
        if lock is None:
            env_lock = os.environ.get("STRINGSIGHT_LMDB_LOCK", "1").strip()
            lock = env_lock not in ("0", "false", "False")
        
        # Separate environments for completions and embeddings
        self.completions_env = lmdb.open(
            str(self.cache_dir / "completions"),
            map_size=max_size_bytes,
            subdir=True,
            readonly=readonly,
            lock=lock,
        )
        
        self.embeddings_env = lmdb.open(
            str(self.cache_dir / "embeddings"),
            map_size=max_size_bytes,
            subdir=True,
            readonly=readonly,
            lock=lock,
        )

        # Validate environments early to surface clear errors
        try:
            with self.completions_env.begin() as _:
                pass
            with self.embeddings_env.begin() as _:
                pass
        except Exception as exc:
            # Raise a clear error with guidance for common filesystem issues
            raise RuntimeError(
                "Failed to initialize LMDB cache. This is often caused by filesystems that do not support file locking (e.g., some network mounts). "
                "You can disable LMDB locking by setting STRINGSIGHT_LMDB_LOCK=0, or disable the cache entirely by setting STRINGSIGHT_DISABLE_LMDB_CACHE=1. "
                f"Cache dir: {self.cache_dir}. Original error: {exc}"
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