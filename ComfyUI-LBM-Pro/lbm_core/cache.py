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

Stampede protection: ``get_or_load`` runs the loader *outside* the
cache lock and dedupes concurrent loads on the same key via a
per-key ``Future``.  Two threads racing for the same key observe
the same in-flight load; two threads loading different keys never
serialize on each other.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict


class LBMModelCache:
    """Singleton cache mapping opaque keys to model objects."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _inflight: Dict[str, Future] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_load(
        cls,
        key: str,
        loader_fn: Callable[[], Any],
        ttl_seconds: int = 600,
    ) -> Any:
        """Return cached object or invoke loader_fn once and cache result.

        The loader runs *outside* ``_lock`` so a slow load for one
        key does not block a fast load for a different key.  Two
        threads hitting the same key while the loader is in flight
        both observe the same ``Future`` — the loader is invoked
        exactly once.

        Side effect: every visit scans the cache for entries whose
        TTL has elapsed and pops them so a long-lived process does
        not accumulate stale model handles (which would otherwise
        only get released when a fresh loader took the same key).
        """
        # Cache hit (cheap, common case) plus opportunistic sweep of
        # expired entries — keeping the dict bounded under a busy
        # multi-model workflow.
        with cls._lock:
            now = time.monotonic()
            expired_keys = [
                k for k, e in cls._cache.items()
                if (now - e["last_used"]) >= ttl_seconds
            ]
            for ek in expired_keys:
                cls._cache.pop(ek, None)
            entry = cls._cache.get(key)
            if entry is not None and (now - entry["last_used"]) < ttl_seconds:
                entry["last_used"] = now
                return entry["model"]

        # Try to claim ownership of the load for this key.  Under the
        # lock we either find an in-flight future (someone else is
        # already loading) or we register our own.
        future: Future = Future()
        with cls._lock:
            winner = cls._inflight.get(key)
            if winner is None:
                # We are the loader thread.  Register the future
                # under the lock so only one thread per key starts.
                cls._inflight[key] = future
                winner = future

        if winner is not future:
            # Another thread is already loading this key.  Wait for
            # its result rather than starting a duplicate loader.
            model = winner.result()
            with cls._lock:
                entry = cls._cache.get(key)
                if entry is not None:
                    entry["last_used"] = time.monotonic()
                else:
                    # The winner hasn't stored its result yet (very
                    # tight race).  Insert a placeholder so subsequent
                    # callers also see the cache hit.
                    cls._cache[key] = {
                        "model": model,
                        "last_used": time.monotonic(),
                    }
            return model

        # We are the loader thread.  Run the loader *outside* the lock
        # so concurrent loads on other keys are not serialised.
        try:
            model = loader_fn()
        except BaseException as exc:
            with cls._lock:
                cls._inflight.pop(key, None)
            future.set_exception(exc)
            raise

        future.set_result(model)

        with cls._lock:
            cls._inflight.pop(key, None)
            cls._cache[key] = {"model": model, "last_used": time.monotonic()}
        return model

    @classmethod
    def unload(cls, key: str) -> bool:
        """Free a single entry's VRAM and drop it from the cache.

        Moves the cached model to CPU (releasing device memory) and
        asks the ComfyUI model manager to flush any cached CUDA
        blocks.  Both calls are wrapped in ``try/except`` so unit
        tests that don't import ``comfy.model_management`` still work.
        """
        with cls._lock:
            entry = cls._cache.pop(key, None)
        if entry is None:
            return False
        _release_vram(entry["model"])
        return True

    @classmethod
    def clear(cls) -> int:
        """Drop every entry, freeing VRAM along the way."""
        with cls._lock:
            entries = list(cls._cache.values())
            cls._cache.clear()
        for entry in entries:
            _release_vram(entry["model"])
        return len(entries)

    @classmethod
    def size(cls) -> int:
        with cls._lock:
            return len(cls._cache)

    @classmethod
    def contains(cls, key: str) -> bool:
        with cls._lock:
            return key in cls._cache


def _release_vram(model: Any) -> None:
    """Best-effort: move a model off the GPU and flush CUDA cache.

    All calls are wrapped in ``try/except`` so unit tests that don't
    import ``comfy.model_management`` still work.
    """
    try:
        model.cpu()
    except Exception:
        pass
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass
