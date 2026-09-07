# Testing

```bash
poetry run pytest                 # atlas downloads are skipped
poetry run pytest --run-network   # include them
poetry run ruff check             # lint
poetry run ruff format            # format
```

`tests/conftest.py` provides the shared fixtures: a seeded generator, a
pre-built five-region connectome with a GEV calibre kernel, and default
simulation parameters at 1 kHz with 300 lags.

## Conventions

What to test
: Prefer a test that asserts a numerical property over one that asserts an
  error message. The suite checks that the sliding-window engine matches the
  defining recursion, that the winding-number stability test agrees with dense
  eigenvalues, that Welch matches the closed form, and that a local delay of
  5 ms moves every mean delay by 5 ms.

Network tests are marked
: Anything that downloads real atlas data carries `@pytest.mark.network` and
  is skipped without `--run-network`. The siibra adapter's own logic (units,
  aggregation, masking, grouping, caching) is tested offline against a fake
  siibra module in `tests/test_connectome/test_siibra_atlas.py`. Extend that
  rather than adding network tests.

The playground is tested headlessly
: With `streamlit.testing.v1.AppTest`. Seed a small specification into
  `session_state["dx_seed_spec"]` to keep it fast.

Cache tests are isolated
: Anything touching the disk cache must point `DIAXCONDEL_CACHE_DIR` at
  `tmp_path`.

## Code style

Ruff does both the linting and the formatting; `ruff.toml` is the only style
configuration. Line length 120, NumPy docstrings, rules `F, E, W, I, B, C4,
UP, N, D`. Tests are linted but exempt from the docstring rules, since those
exist for the package's public surface and a test's name is its
documentation.

`target-version` there must match `requires-python` in `pyproject.toml`. A
higher setting lets ruff rewrite code into syntax the project claims to
support but older interpreters reject.

Docstrings are part of the user interface: the dashboard reads them for its
help text and the API reference is generated from them, so write them for a
reader who has not read the paper.
