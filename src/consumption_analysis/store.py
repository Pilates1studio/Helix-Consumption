"""Persist the ingested account table so the slow source read happens once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .ingest import AccountData

_KINDS = ("days", "usage", "budget")


def save(accounts: AccountData, cfg: StudyConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = accounts.meta.copy()
    for kind, grids in (("days", accounts.days), ("usage", accounts.usage),
                        ("budget", accounts.budgets or {})):
        for year, grid in grids.items():
            for i, period in enumerate(cfg.periods):
                frame[f"{kind}|{year}|{period}"] = grid[:, i]
    frame.to_parquet(path, index=False, compression="zstd")
    return path


def load(cfg: StudyConfig, path: str | Path) -> AccountData:
    frame = pd.read_parquet(path)
    meta_cols = [c for c in frame.columns if "|" not in c]
    grids: dict[str, dict[str, np.ndarray]] = {k: {} for k in _KINDS}

    years: dict[str, set[str]] = {k: set() for k in _KINDS}
    for col in frame.columns:
        if "|" in col:
            kind, year, _ = col.split("|")
            years[kind].add(year)

    for kind in _KINDS:
        for year in sorted(years[kind]):
            cols = [f"{kind}|{year}|{p}" for p in cfg.periods]
            grids[kind][year] = frame[cols].to_numpy(dtype=float)

    return AccountData(
        meta=frame[meta_cols].copy(),
        days=grids["days"],
        usage=grids["usage"],
        budgets=grids["budget"] or None,
    )
