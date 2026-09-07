"""
On-disk cache for expensive, deterministic external lookups.

Atlas queries (see :mod:`diaxcondel.connectome.atlases.siibra_atlas`) download
and aggregate hundreds of subject matrices. The result is a pure function of
the query, so it is cached as an ``.npz`` array bundle plus a JSON sidecar
holding provenance metadata.

The cache location is ``$DIAXCONDEL_CACHE_DIR`` when set, and otherwise the
platform's own cache directory (via ``platformdirs``, so the path follows the
conventions of Linux, macOS and Windows rather than this package's guess).

Set ``DIAXCONDEL_NO_CACHE=1`` to bypass reads and writes entirely (useful in
tests and on read-only filesystems).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from platformdirs import user_cache_dir

from ._typing import FloatArray

_ENV_DIR = "DIAXCONDEL_CACHE_DIR"
_ENV_DISABLE = "DIAXCONDEL_NO_CACHE"


def cache_dir() -> Path:
    """
    Return the directory used for cached downloads.

    Returns
    -------
    Path
        Cache root. The directory is not created by this call.
    """
    override = os.environ.get(_ENV_DIR)
    if override:
        return Path(override).expanduser()
    return Path(user_cache_dir("diaxcondel"))


def caching_enabled() -> bool:
    """Return ``True`` unless ``DIAXCONDEL_NO_CACHE`` is set to a truthy value."""
    return os.environ.get(_ENV_DISABLE, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def cache_key(payload: Mapping[str, object]) -> str:
    """
    Return a stable short hash of a JSON-serialisable query description.

    Parameters
    ----------
    payload : Mapping[str, object]
        Query description. Must be JSON-serialisable; key order is
        irrelevant.

    Returns
    -------
    str
        16-character hexadecimal digest.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """
    Arrays plus provenance metadata, either freshly computed or loaded.

    Attributes
    ----------
    arrays : dict of str to FloatArray
        The cached numeric payload.
    metadata : dict
        JSON-serialisable provenance (query parameters, region names, units).
    path : Path or None
        Location of the ``.npz`` file, or ``None`` when caching is disabled.
    from_cache : bool
        ``True`` if the entry was read from disk rather than computed.
    """

    arrays: dict[str, FloatArray]
    metadata: dict[str, object]
    path: Path | None
    from_cache: bool


def load_or_compute(
    namespace: str,
    payload: Mapping[str, object],
    compute: Callable[[], tuple[Mapping[str, FloatArray], Mapping[str, object]]],
) -> CacheEntry:
    """
    Return cached arrays for ``payload``, computing and storing them if absent.

    Parameters
    ----------
    namespace : str
        Sub-directory name grouping related queries (e.g. ``"siibra"``).
    payload : Mapping[str, object]
        JSON-serialisable description of the query; hashed into the filename.
    compute : callable
        Zero-argument callable returning ``(arrays, metadata)``. Only called
        on a cache miss.

    Returns
    -------
    CacheEntry
        Arrays, metadata, and whether the result came from disk.

    Notes
    -----
    A corrupt or unreadable cache file is treated as a miss and overwritten;
    failures to *write* the cache are non-fatal (the computed value is still
    returned).
    """
    if not caching_enabled():
        arrays, metadata = compute()
        return CacheEntry(arrays=dict(arrays), metadata=dict(metadata), path=None, from_cache=False)

    key = cache_key(payload)
    directory = cache_dir() / namespace
    npz_path = directory / f"{key}.npz"
    json_path = directory / f"{key}.json"

    if npz_path.is_file() and json_path.is_file():
        try:
            with np.load(npz_path) as handle:
                arrays = {name: handle[name] for name in handle.files}
            metadata = json.loads(json_path.read_text())
            return CacheEntry(arrays=arrays, metadata=metadata, path=npz_path, from_cache=True)
        except (OSError, ValueError, EOFError, json.JSONDecodeError):
            # Corrupt entry: fall through and recompute.
            pass

    computed_arrays, computed_metadata = compute()
    arrays = {k: np.asarray(v) for k, v in computed_arrays.items()}
    metadata = dict(computed_metadata)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
        path: Path | None = npz_path
    except OSError:
        path = None
    return CacheEntry(arrays=arrays, metadata=metadata, path=path, from_cache=False)


def clear_cache(namespace: str | None = None) -> int:
    """
    Delete cached files.

    Parameters
    ----------
    namespace : str, optional
        Only clear this sub-directory. ``None`` clears the whole cache root.

    Returns
    -------
    int
        Number of files removed.
    """
    root = cache_dir() if namespace is None else cache_dir() / namespace
    if not root.exists():
        return 0
    n_files = sum(1 for path in root.rglob("*") if path.is_file())
    shutil.rmtree(root)
    return n_files
