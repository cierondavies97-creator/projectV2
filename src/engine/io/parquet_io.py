from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl


def write_parquet(
    df: pl.DataFrame,
    dir_path: Path,
    *,
    file_name: str = "part.parquet",
    overwrite: bool = True,
) -> Path:
    """
    Write a DataFrame to a partition directory.

    Policy:
      - Ensure dir_path exists.
      - Write a single file (configurable name).
      - Atomic replace into final path (safe against partial writes).
      - Overwrite if it already exists (default True; deterministic for a given run).
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    final_path = dir_path / file_name
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing parquet: {final_path}")

    # Write to a temp file in the same directory so os.replace() is atomic on the same filesystem.
    fd, tmp_name = tempfile.mkstemp(prefix=f".tmp_{file_name}_", suffix=".parquet", dir=str(dir_path))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, final_path)  # atomic on same filesystem
    finally:
        # If something went wrong before replace, cleanup temp file.
        if tmp_path.exists() and tmp_path != final_path:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return final_path


def read_parquet_dir(dir_path: Path) -> pl.DataFrame:
    """
    Read all *.parquet files from a directory into a single DataFrame.

    Behaviour:
      - If dir_path does not exist -> raise FileNotFoundError.
      - If dir_path exists but has no *.parquet files -> return empty DataFrame.
      - Otherwise, read and vertical-concat all files.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Parquet directory not found: {dir_path}")

    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = [pl.read_parquet(fp) for fp in files]
    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")
