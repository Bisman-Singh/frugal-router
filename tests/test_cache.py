from frugal_router.cache import ResponseCache


def test_key_is_stable_and_order_independent():
    a = ResponseCache.key(model="m", prompt="p", max_tokens=32)
    b = ResponseCache.key(max_tokens=32, prompt="p", model="m")
    assert a == b
    assert a != ResponseCache.key(model="m", prompt="p2", max_tokens=32)


def test_roundtrip_in_memory():
    cache = ResponseCache()
    key = ResponseCache.key(model="m", prompt="p")
    assert cache.get(key) is None
    cache.put(key, {"text": "42", "prompt_tokens": 10, "completion_tokens": 2})
    assert cache.get(key) == {"text": "42", "prompt_tokens": 10, "completion_tokens": 2}


def test_persists_to_disk(tmp_path):
    path = str(tmp_path / "cache.sqlite")
    key = ResponseCache.key(model="m", prompt="p")
    ResponseCache(path).put(key, {"text": "hi"})
    assert ResponseCache(path).get(key) == {"text": "hi"}
