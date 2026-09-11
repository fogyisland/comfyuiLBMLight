"""Thread-safe TTL-bounded model cache for LBM-Pro.

ComfyUI invokes nodes many times in a session. Loading a 1–2 GB LBM
checkpoint takes 5–10 seconds. This cache memoizes loaded models by an
opaque key (typically f"{name}|{task}|{precision}") with a TTL so
unused models get released.

The cache is *not* a true LRU: it does not maintain insertion order
or evict on capacity.  It only checks the per-entry ``last_used``
timestamp and lets a fresh loader take over once the entry has
aged past the TTL.  Call :meth:`unload` to drop a specific entry,
:meth:`clear` to drop everything.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable


class LBMModelCache:
    """Singleton cache mapping opaque keys to model objects."""

    _cache: dict[str, dict[str, Any]] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_load(
        cls,
        key: str,
        loader_fn: Callable[[], Any],
        ttl_seconds: int = 600,
    ) -> Any:
        """Return cached object or invoke loader_fn once and cache result."""
        with cls._lock:
            entry = cls._cache.get(key)
            now = time.monotonic()
            if entry is not None and (now - entry["last_used"]) < ttl_seconds:
                entry["last_used"] = now
                return entry["model"]
            obj = loader_fn()
            cls._cache[key] = {"model": obj, "last_used": now}
            return obj

    @classmethod
    def unload(cls, key: str) -> bool:
        with cls._lock:
            return cls._cache.pop(key, None) is not None

    @classmethod
    def clear(cls) -> int:
        with cls._lock:
            n = len(cls._cache)
            cls._cache.clear()
            return n

    @classmethod
    def size(cls) -> int:
        with cls._lock:
            return len(cls._cache)

    @classmethod
    def contains(cls, key: str) -> bool:
        with cls._lock:
            return key in cls._cache
