"""Canonical parquet I/O for trajectory data.

write_canonical / read_canonical work exclusively with the 6 columns defined
in TRAJECTORY_COLUMNS.  Any extra columns (e.g. is_anomaly_gt, kind) are
silently dropped on write so the canonical file stays schema-clean.

load_open is an optional stub for public open datasets — it is not on the
critical path and does not perform any network I/O.
"""

from __future__ import annotations

import pathlib
from typing import Union

import pandas as pd

from contracts.schema import TRAJECTORY_COLUMNS

PathLike = Union[str, pathlib.Path]


def write_canonical(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to parquet using exactly TRAJECTORY_COLUMNS order.

    Extra columns are dropped silently.  Missing canonical columns raise
    ``KeyError``.

    Parameters
    ----------
    df:
        Source DataFrame; must contain all columns in TRAJECTORY_COLUMNS.
    path:
        Destination file path.  Parent directory must exist or be created
        by the caller.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = df[list(TRAJECTORY_COLUMNS)].copy()
    canonical.to_parquet(path, index=False, engine="pyarrow")


def read_canonical(path: PathLike) -> pd.DataFrame:
    """Read a canonical parquet file and return exactly TRAJECTORY_COLUMNS.

    Parameters
    ----------
    path:
        Path to a parquet file previously written by ``write_canonical``.

    Returns
    -------
    DataFrame with column order matching TRAJECTORY_COLUMNS.
    """
    df = pd.read_parquet(pathlib.Path(path), engine="pyarrow")
    return df[list(TRAJECTORY_COLUMNS)]


def load_open(path: PathLike) -> pd.DataFrame | None:
    """Load a public open GPS dataset in canonical column order.

    This is a stub — no network I/O is performed.  Callers should:
    1. Download the dataset separately and store locally.
    2. Pass the local file path here.
    3. Implement the format-specific parsing for the chosen dataset.

    The function is intentionally left as a stub so it is importable and
    callable without any external dependencies.  It raises
    ``NotImplementedError`` to make the stub status explicit.

    Parameters
    ----------
    path:
        Path to a locally stored open dataset file.

    Raises
    ------
    NotImplementedError:
        Always, until a real dataset loader is implemented.
    """
    raise NotImplementedError(
        "load_open is a documented stub.  Download a public GPS dataset, "
        "then implement parsing here.  No network calls are made automatically."
    )
