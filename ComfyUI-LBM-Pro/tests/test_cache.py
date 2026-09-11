import threading
import time
from lbm_core.cache import LBMModelCache


def _fake_loader(name):
    return {"name": name, "loaded_at": time.time()}


def test_cache_miss_calls_loader():
    LBMModelCache.clear()
    obj = LBMModelCache.get_or_load("k1", lambda: _fake_loader("alpha"))
    assert obj["name"] == "alpha"


def test_cache_hit_skips_loader():
    LBMModelCache.clear()
    counter = {"n": 0}

    def loader():
        counter["n"] += 1
        return {"v": counter["n"]}

    LBMModelCache.get_or_load("k2", loader)
    LBMModelCache.get_or_load("k2", loader)
    LBMModelCache.get_or_load("k2", loader)
    assert counter["n"] == 1


def test_different_keys_different_loaders():
    LBMModelCache.clear()
    LBMModelCache.get_or_load("a", lambda: "A")
    LBMModelCache.get_or_load("b", lambda: "B")
    assert LBMModelCache.get_or_load("a", lambda: "A") == "A"
    assert LBMModelCache.get_or_load("b", lambda: "B") == "B"


def test_unload_removes_entry():
    LBMModelCache.clear()
    LBMModelCache.get_or_load("x", lambda: 42)
    assert LBMModelCache.contains("x")
    assert LBMModelCache.unload("x") is True
    assert not LBMModelCache.contains("x")


def test_unload_missing_returns_false():
    LBMModelCache.clear()
    assert LBMModelCache.unload("nope") is False


def test_clear_returns_count():
    LBMModelCache.clear()
    LBMModelCache.get_or_load("p", lambda: 1)
    LBMModelCache.get_or_load("q", lambda: 2)
    n = LBMModelCache.clear()
    assert n == 2
    assert LBMModelCache.size() == 0


def test_ttl_expiry_calls_loader_again():
    LBMModelCache.clear()
    counter = {"n": 0}

    def loader():
        counter["n"] += 1
        return counter["n"]

    LBMModelCache.get_or_load("ttl", loader, ttl_seconds=1)
    LBMModelCache.get_or_load("ttl", loader, ttl_seconds=1)
    time.sleep(1.2)
    LBMModelCache.get_or_load("ttl", loader, ttl_seconds=1)
    assert counter["n"] == 2


def test_cache_does_not_block_other_keys():
    """Two threads on different keys must not serialize on each other.

    The slow loader blocks on a 2-second event; the fast loader has
    no blocker. The fast loader's ``result()`` must return in well
    under 2 seconds, proving the cache released its lock before
    invoking the loader.
    """
    import concurrent.futures

    LBMModelCache.clear()

    slow_event = threading.Event()

    def _slow_loader():
        slow_event.wait(timeout=2.0)
        return "slow"

    def _fast_loader():
        return "fast"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        slow_future = pool.submit(LBMModelCache.get_or_load, "slow_key", _slow_loader)
        # Give the slow call a moment to enter the loader (so it is
        # actually running its blocking wait at the time we start the
        # fast call). Without this, the fast call could short-circuit
        # through the cache-miss path entirely before the slow call
        # has acquired the lock.
        time.sleep(0.1)
        fast_start = time.monotonic()
        fast_future = pool.submit(LBMModelCache.get_or_load, "fast_key", _fast_loader)
        fast_result = fast_future.result(timeout=0.5)
        fast_elapsed = time.monotonic() - fast_start

    # Release the slow call so the executor can shut down cleanly.
    slow_event.set()
    assert slow_future.result(timeout=2.0) == "slow"
    assert fast_result == "fast"
    # If the cache were holding the lock while the loader ran, the fast
    # call would block for ~2s; allow a generous bound for CI jitter.
    assert fast_elapsed < 0.5, (
        f"fast loader blocked for {fast_elapsed:.2f}s — the cache is "
        "holding its lock while invoking the loader."
    )


def test_cache_unload_moves_to_cpu():
    """``unload`` must call ``.cpu()`` on the cached model.

    Wrapped with try/except in production so unit tests that don't
    import ``comfy.model_management`` still work. We mock ``.cpu()``
    on the model object to verify the call is made.
    """
    from unittest.mock import MagicMock

    LBMModelCache.clear()

    # A fake model object whose ``.cpu()`` returns a sentinel we can
    # recognise.  We assert on the call, not on the return value.
    sentinel = object()
    fake_model = MagicMock()
    fake_model.cpu.return_value = sentinel

    LBMModelCache.get_or_load("cpu_key", lambda: fake_model)
    # The cache stored the original (MagicMock) instance — the
    # production unload should still invoke ``.cpu()`` on it.
    assert LBMModelCache.unload("cpu_key") is True
    assert fake_model.cpu.called, "unload did not invoke .cpu() on the cached model"
