"""
Hand-entered connectomes.

Numbers typed into a table or pasted from a paper need the same validation as
numbers pulled from an atlas: square, finite, symmetric, zero on the
diagonal. These helpers parse the text form used by saved specifications and
by the playground's editable tables, and return ordinary providers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from diaxcondel._typing import FloatArray

from .distance import ManualDistances
from .regions import RegionSet
from .weights import ManualConnectivity


def parse_matrix(text: str, *, n: int | None = None) -> FloatArray:
    r"""
    Parse a matrix written as JSON, or as rows of comma/space-separated numbers.

    Parameters
    ----------
    text : str
        ``"[[0, 5], [5, 0]]"`` or ``"0 5\\n5 0"``. Empty text gives a zero
        matrix, which requires ``n``.
    n : int, optional
        Expected size. When given, the result must be ``(n, n)``.

    Returns
    -------
    FloatArray of shape (N, N)
        Parsed matrix.

    Raises
    ------
    ValueError
        If the text cannot be read as a square numeric matrix of the
        expected size.
    """
    text = (text or "").strip()
    if not text:
        if n is None:
            raise ValueError("empty matrix text and no size given")
        return np.zeros((n, n), dtype=float)

    values: object
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        rows = [row for row in text.replace(";", "\n").splitlines() if row.strip()]
        values = [[float(cell) for cell in row.replace(",", " ").split()] for row in rows]

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix; parsed shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite numbers")
    if n is not None and matrix.shape[0] != n:
        raise ValueError(f"expected a {n}x{n} matrix; got {matrix.shape[0]}x{matrix.shape[1]}")
    return matrix


def format_matrix(matrix: FloatArray, *, decimals: int = 4) -> str:
    """
    Render a matrix as compact JSON, ready to store in a specification.

    Parameters
    ----------
    matrix : FloatArray of shape (N, N)
        Matrix to render.
    decimals : int
        Decimal places to keep.

    Returns
    -------
    str
        JSON text that :func:`parse_matrix` reads back.
    """
    rounded = np.round(np.asarray(matrix, dtype=float), decimals)
    return json.dumps([[float(value) for value in row] for row in rounded])


def parse_codes(text: str, *, n: int | None = None, prefix: str = "R") -> tuple[str, ...]:
    """
    Parse region codes from a comma- or newline-separated list.

    Parameters
    ----------
    text : str
        ``"FL, PL, OL"``. Empty text generates ``R1 ... Rn``, which requires
        ``n``.
    n : int, optional
        Expected number of codes.
    prefix : str
        Prefix for generated codes.

    Returns
    -------
    tuple of str
        Region codes.

    Raises
    ------
    ValueError
        If the codes are duplicated or their count does not match ``n``.
    """
    codes = [code.strip() for code in (text or "").replace("\n", ",").split(",") if code.strip()]
    if not codes:
        if n is None:
            raise ValueError("no region codes given and no size to generate them from")
        codes = [f"{prefix}{i + 1}" for i in range(n)]
    if len(set(codes)) != len(codes):
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        raise ValueError(f"duplicate region codes: {duplicates}")
    if n is not None and len(codes) != n:
        raise ValueError(f"expected {n} region codes; got {len(codes)}")
    return tuple(codes)


def load_matrix(path: str | Path) -> FloatArray:
    """
    Read a square matrix from a text or NumPy file.

    Parameters
    ----------
    path : str or Path
        File to read. ``.npy`` and ``.npz`` are read with NumPy (an ``.npz``
        must hold exactly one array); anything else is read as delimited
        text, with the delimiter inferred from the first data line and lines
        beginning with ``#`` treated as comments.

    Returns
    -------
    FloatArray of shape (N, N)
        The matrix, as floats.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the contents are not a single square numeric matrix.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"no such matrix file: {file}")
    if file.suffix == ".npy":
        matrix = np.load(file)
    elif file.suffix == ".npz":
        with np.load(file) as bundle:
            names = list(bundle.files)
            if len(names) != 1:
                raise ValueError(f"{file} holds {len(names)} arrays; it must hold exactly one")
            matrix = bundle[names[0]]
    else:
        text = [line for line in file.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not text:
            raise ValueError(f"{file} contains no data")
        delimiter = "\t" if "\t" in text[0] else ("," if "," in text[0] else None)
        matrix = np.loadtxt(text, delimiter=delimiter)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{file} must contain a square matrix; got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{file} contains non-finite values")
    return matrix


def load_codes(path: str | Path) -> tuple[str, ...]:
    """
    Read region codes from a text file, one per line or comma-separated.

    Parameters
    ----------
    path : str or Path
        File to read. Blank lines and lines beginning with ``#`` are ignored.

    Returns
    -------
    tuple of str
        Region codes, in file order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If it holds no codes.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"no such region-code file: {file}")
    codes: list[str] = []
    for line in file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        codes.extend(part.strip() for part in stripped.split(",") if part.strip())
    if not codes:
        raise ValueError(f"{file} contains no region codes")
    return tuple(codes)


def manual_distances(
    matrix_cm: FloatArray | str,
    codes: Sequence[str] | str = (),
    *,
    symmetrize: bool = True,
    name: str = "manual distances",
) -> ManualDistances:
    """
    Build a distance provider from hand-entered values.

    Parameters
    ----------
    matrix_cm : FloatArray or str
        Distances in centimetres, as an array or as text (see
        :func:`parse_matrix`).
    codes : sequence of str or str
        Region codes; empty generates ``R1 ... Rn``.
    symmetrize : bool
        Average the matrix with its transpose. Distances are symmetric by
        definition, so this quietly fixes a half-filled table; set it to
        ``False`` to have an asymmetric entry rejected instead.
    name : str
        Provider name for figures and reports.

    Returns
    -------
    ManualDistances
        Validated provider with a zero diagonal.
    """
    matrix = parse_matrix(matrix_cm) if isinstance(matrix_cm, str) else np.asarray(matrix_cm, dtype=float)
    if symmetrize:
        matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    region_codes = parse_codes(codes, n=matrix.shape[0]) if isinstance(codes, str) else tuple(codes)
    regions = RegionSet.from_codes(list(region_codes) or [f"R{i + 1}" for i in range(matrix.shape[0])])
    return ManualDistances(
        regions=regions,
        matrix_cm=matrix,
        name=name,
        metadata={"source": "manual", "units": "cm", "symmetrized": symmetrize},
    )


def manual_connectivity(
    regions: RegionSet,
    weights: FloatArray | str,
    *,
    symmetrize: bool = False,
    name: str = "manual connectivity",
) -> ManualConnectivity:
    """
    Build a connectivity provider from hand-entered values.

    Parameters
    ----------
    regions : RegionSet
        Regions the matrix refers to; its size fixes the expected shape.
    weights : FloatArray or str
        Weight matrix, as an array or as text (see :func:`parse_matrix`).
        Entry ``[i, j]`` is the weight from source ``j`` onto target ``i``.
    symmetrize : bool
        Average with the transpose. Off by default: directed weights are
        meaningful here, unlike distances.
    name : str
        Provider name for figures and reports.

    Returns
    -------
    ManualConnectivity
        Validated provider with a zero diagonal (recurrent self-excitation is
        set at build time, together with the local delay it needs).
    """
    matrix = parse_matrix(weights, n=len(regions)) if isinstance(weights, str) else np.asarray(weights, dtype=float)
    matrix = np.array(matrix, dtype=float, copy=True)
    if symmetrize:
        matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return ManualConnectivity(
        regions=regions,
        weights=np.clip(matrix, 0.0, None),
        name=name,
        metadata={"source": "manual", "symmetrized": symmetrize},
    )
