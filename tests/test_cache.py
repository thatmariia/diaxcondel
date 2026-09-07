"""Tests for the on-disk cache used by atlas downloads."""

import numpy as np
import pytest

from diaxcondel._cache import (
    cache_dir,
    cache_key,
    caching_enabled,
    clear_cache,
    load_or_compute,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DIAXCONDEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DIAXCONDEL_NO_CACHE", raising=False)
    return tmp_path / "cache"


def _counting_compute(counter):
    def compute():
        counter.append(1)
        return {"matrix": np.arange(4.0).reshape(2, 2)}, {"note": "computed"}

    return compute


def test_second_call_reads_from_disk(isolated_cache):
    calls = []
    first = load_or_compute("test", {"a": 1}, _counting_compute(calls))
    second = load_or_compute("test", {"a": 1}, _counting_compute(calls))
    assert len(calls) == 1
    assert second.from_cache and not first.from_cache
    np.testing.assert_array_equal(first.arrays["matrix"], second.arrays["matrix"])
    assert second.metadata["note"] == "computed"
    assert second.path is not None and second.path.is_file()


def test_different_payloads_do_not_collide():
    calls = []
    load_or_compute("test", {"a": 1}, _counting_compute(calls))
    load_or_compute("test", {"a": 2}, _counting_compute(calls))
    assert len(calls) == 2
    assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})


def test_corrupt_entry_is_recomputed():
    calls = []
    entry = load_or_compute("test", {"a": 1}, _counting_compute(calls))
    assert entry.path is not None
    entry.path.write_bytes(b"not an npz")
    recomputed = load_or_compute("test", {"a": 1}, _counting_compute(calls))
    assert len(calls) == 2
    assert not recomputed.from_cache


def test_caching_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DIAXCONDEL_NO_CACHE", "1")
    assert not caching_enabled()
    calls = []
    entry = load_or_compute("test", {"a": 1}, _counting_compute(calls))
    load_or_compute("test", {"a": 1}, _counting_compute(calls))
    assert len(calls) == 2
    assert entry.path is None


def test_clear_cache_counts_removed_files():
    load_or_compute("one", {"a": 1}, _counting_compute([]))
    load_or_compute("two", {"a": 1}, _counting_compute([]))
    assert clear_cache("one") == 2  # arrays plus metadata sidecar
    assert clear_cache() == 2
    assert clear_cache() == 0
    assert not cache_dir().exists()


def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("DIAXCONDEL_CACHE_DIR", raising=False)
    platform_default = cache_dir()
    assert platform_default.name == "diaxcondel"
    assert platform_default.is_absolute()
    monkeypatch.setenv("DIAXCONDEL_CACHE_DIR", str(tmp_path / "explicit"))
    assert cache_dir() == tmp_path / "explicit"
