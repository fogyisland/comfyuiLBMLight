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
