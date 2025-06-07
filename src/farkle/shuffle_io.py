# ── shuffle_io.py ─────────────────────────────────────────────────────────────
"""Utilities for persisting and retrieving pre-computed strategy-grid shuffles.

The file is written as a raw NumPy `uint16` array shaped
    (n_shuffles, n_strats)      # e.g. (10 223, 8 160)

Because `uint16` tops out at 65 535, it easily holds every index < 8 160.
The file can be memory-mapped later so any worker can grab any slice
without copying the whole thing into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

import numpy as np


# --------------------------------------------------------------------------- #
# 1.  Generate & write                                                         #
# --------------------------------------------------------------------------- #
def write_shuffle_file(
    path: str | Path,
    *,
    n_strats: int   = 8_160,
    n_shuffles: int = 10_223,
    dtype           = np.uint16,
    seed: int | None = None,
    log_every: int  = 500,
) -> Path:
    """
    Pre-computes `n_shuffles` random permutations of `range(n_strats)` and
    writes them to *path* in raw binary form.

    The function streams one shuffle at a time into a writable memmap so the
    peak RAM footprint is one permutation (≈16 kB).

    Returns
    -------
    pathlib.Path
        The resolved file path.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rng  = np.random.default_rng(seed)
    mmap = np.memmap(
        path,
        dtype=dtype,
        mode="w+",
        shape=(n_shuffles, n_strats),
    )

    for i in range(n_shuffles):
        mmap[i] = rng.permutation(n_strats)
        if i % log_every == 0:
            print(f"[shuffle_io] generated {i:>5}/{n_shuffles} shuffles", end="\r")

    mmap.flush()           # make sure everything is on disk
    del mmap               # close the memmap
    print(f"\n[shuffle_io] wrote {n_shuffles} shuffles → {path} "
          f"({path.stat().st_size/1_048_576:.1f} MiB)")
    return path


# --------------------------------------------------------------------------- #
# 2.  Open & slice                                                             #
# --------------------------------------------------------------------------- #
@overload
def read_shuffles(
    path: str | Path,
    /,
    start: int,
    *,
    count: int | None   = None,
    as_array: Literal[True],
    n_strats: int       = 8_160,
    dtype               = np.uint16,
) -> np.ndarray: ...
@overload
def read_shuffles(
    path: str | Path,
    /,
    start: int,
    *,
    count: int | None   = None,
    as_array: Literal[False] = ...,
    n_strats: int       = 8_160,
    dtype               = np.uint16,
) -> np.memmap: ...

def read_shuffles(
    path: str | Path,
    /,
    start: int,
    *,
    count: int | None   = None,
    as_array: bool      = True,
    n_strats: int       = 8_160,
    dtype               = np.uint16,
) -> np.ndarray | np.memmap:
    """
    Opens the file created by :func:`write_shuffle_file` and returns *count*
    consecutive shuffles starting at *start* (0-based).

    Parameters
    ----------
    start : int
        Index of the first shuffle you want.  Negative indices follow NumPy’s
        convention (e.g. -1 is the last shuffle).
    count : int | None, default None
        How many shuffles to read.  ``None`` ⇒ read to the end.
    as_array : bool, default True
        • ``True``  ⇒ returns a *copy* (safe to modify).  
        • ``False`` ⇒ returns a *memmap view* (zero-copy, read-only).
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # First open as 1-D memmap, then reshape lazily
    raw   = np.memmap(path, dtype=dtype, mode="r")
    n_tot = raw.size // n_strats
    mmap  = raw.reshape((n_tot, n_strats))

    # Normalise slice
    if start < 0:
        start += n_tot
    stop = n_tot if count is None else start + count
    if not (0 <= start < stop <= n_tot):
        raise IndexError("Requested slice out of bounds")

    view = mmap[start:stop]
    return view.copy() if as_array else view


# --------------------------------------------------------------------------- #
# Example usage (delete or guard with if __name__ == "__main__")              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    FILE = write_shuffle_file("data/shuffles.bin", seed=42)

    # Grab shuffles 100–109 (inclusive) *without* copying:
    window = read_shuffles(FILE, 100, count=10, as_array=False)
    print(f"Shape: {window.shape}, dtype: {window.dtype}, first row: {window[0][:10]}")