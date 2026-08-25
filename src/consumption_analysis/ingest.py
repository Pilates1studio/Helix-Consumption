"""Turn a raw billing extract into the account table the model runs on.

The source arrives one column per calendar month. Agencies that bill bi-monthly
read each account in only one month of each pair, so half those columns are NULL
for any given account - which of the two depends on the account's read cycle.
Summing each period's months collapses the file to real billing periods and is
indifferent to which cycle an account sits on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .xlsx_stream import Workbook

DAYS_RE = re.compile(r"^BILLING_DAYS_([A-Z]{3})(\d{2})$")
USAGE_RE = re.compile(r"^([A-Z]{3})_FY(\d{2})$")
BUDGET_RE = re.compile(r"^WB_([A-Z]{3})_FY(\d{2})$")

META_COLUMNS = {
    "LOCATION_NO": "location_no",
    "SERVICE_SEQ": "service_seq",
    "LOCSVC_ID": "locsvc_id",
    "LOCATION_CLASS": "location_class",
    "METER_NO": "meter_no",
    "METER_SN": "meter_sn",
    "METER_SZ": "meter_sz",
    "METER_SEQ": "meter_seq",
    "RATE": "rate_cd",
}


@dataclass
class AccountData:
    """Account attributes plus (n_accounts, n_periods) day and usage grids."""

    meta: pd.DataFrame
    days: dict[str, np.ndarray]
    usage: dict[str, np.ndarray]
    budgets: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.meta)


def _fy_label(yy: str) -> str:
    return f"FY20{yy}"


def _to_number(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _month_to_period(cfg: StudyConfig) -> dict[str, int]:
    return {m: i for i, (_, months) in enumerate(cfg.period_months.items()) for m in months}


def read_consumption(path: str | Path, cfg: StudyConfig, sheet: str,
                     header_row: int = 2, progress: int = 10_000) -> AccountData:
    """Read the billing extract and fold monthly columns into billing periods."""
    wb = Workbook(path)
    month_period = _month_to_period(cfg)
    n_periods = cfg.n_periods
    years = set(cfg.fiscal_years)

    meta_rows: list[dict] = []
    days_rows: dict[str, list[np.ndarray]] = {fy: [] for fy in cfg.fiscal_years}
    usage_rows: dict[str, list[np.ndarray]] = {fy: [] for fy in cfg.fiscal_years}

    layout: dict[str, tuple[str, str, int]] = {}  # header -> (kind, fy, period)
    seen = 0

    for record in wb.table(sheet, header_row=header_row):
        if not layout:
            for label in record:
                if m := DAYS_RE.match(label):
                    month, yy = m.groups()
                    if _fy_label(yy) in years and month in month_period:
                        layout[label] = ("days", _fy_label(yy), month_period[month])
                elif m := USAGE_RE.match(label):
                    month, yy = m.groups()
                    if _fy_label(yy) in years and month in month_period:
                        layout[label] = ("usage", _fy_label(yy), month_period[month])
            _check_layout(layout, cfg)

        meta_rows.append({dest: str(record.get(src, "")).strip()
                          for src, dest in META_COLUMNS.items()})

        d = {fy: np.zeros(n_periods) for fy in cfg.fiscal_years}
        u = {fy: np.zeros(n_periods) for fy in cfg.fiscal_years}
        for label, (kind, fy, period) in layout.items():
            value = record.get(label)
            if value is None:
                continue
            (d if kind == "days" else u)[fy][period] += _to_number(value)

        for fy in cfg.fiscal_years:
            days_rows[fy].append(d[fy])
            usage_rows[fy].append(u[fy])

        seen += 1
        if progress and seen % progress == 0:
            print(f"  ...{seen:,} accounts", flush=True)

    meta = pd.DataFrame(meta_rows)
    meta["meter_sz"] = meta["meter_sz"].str.strip()
    meta["rate_cd"] = meta["rate_cd"].str.strip()
    meta["cust_class"] = meta["rate_cd"].map(cfg.rate_code_map)
    meta["dwelling_units"] = 1.0

    return AccountData(
        meta=meta,
        days={fy: np.vstack(days_rows[fy]) for fy in cfg.fiscal_years},
        usage={fy: np.vstack(usage_rows[fy]) for fy in cfg.fiscal_years},
    )


def consolidate_services(accounts: AccountData, cfg: StudyConfig) -> pd.DataFrame:
    """Roll multiple meter rows up to one row per service (LOCSVC_ID).

    A meter changeout splits a service's year across two rows, so usage is summed.
    Billing days are NOT summed - both rows describe the same elapsed period, and
    adding them would double the days and inflate every day-prorated tier
    allotment. The maximum is taken instead.

    Returns a report of services whose rows disagree on meter size.
    """
    meta = accounts.meta
    order = np.lexsort((
        pd.to_numeric(meta["meter_seq"], errors="coerce").fillna(0).to_numpy(),
        sum(accounts.usage[fy].sum(axis=1) for fy in cfg.fiscal_years),
    ))[::-1]

    codes, uniques = pd.factorize(meta["locsvc_id"], sort=False)
    n_groups = len(uniques)

    summed: dict[str, np.ndarray] = {}
    maxed: dict[str, np.ndarray] = {}
    for fy in cfg.fiscal_years:
        summed[fy] = _group_reduce(accounts.usage[fy], codes, n_groups, "sum")
        maxed[fy] = _group_reduce(accounts.days[fy], codes, n_groups, "max")

    # Representative attributes: the row with the most usage wins, then highest seq.
    winner = np.full(n_groups, -1, dtype=int)
    for row in order:
        g = codes[row]
        if winner[g] == -1:
            winner[g] = row

    conflicts = (
        meta.assign(_g=codes)
            .groupby("_g")["meter_sz"].nunique()
            .pipe(lambda s: s[s > 1])
    )

    accounts.meta = meta.iloc[winner].reset_index(drop=True)
    accounts.usage = summed
    accounts.days = maxed

    return pd.DataFrame({
        "locsvc_id": uniques[conflicts.index.to_numpy()],
        "distinct_meter_sizes": conflicts.to_numpy(),
    })


def _group_reduce(grid: np.ndarray, codes: np.ndarray, n_groups: int, how: str) -> np.ndarray:
    out = np.zeros((n_groups, grid.shape[1]))
    if how == "sum":
        np.add.at(out, codes, grid)
    elif how == "max":
        np.maximum.at(out, codes, grid)
    else:
        raise ValueError(how)
    return out


def _check_layout(layout: dict, cfg: StudyConfig) -> None:
    """Fail loudly if the extract does not cover every period of every year."""
    expected = {(kind, fy, p) for kind in ("days", "usage")
                for fy in cfg.fiscal_years for p in range(cfg.n_periods)}
    missing = expected - set(layout.values())
    if missing:
        gaps = sorted(f"{fy} {cfg.periods[p]} {kind}" for kind, fy, p in missing)
        raise ValueError(
            "billing extract is missing columns for: " + ", ".join(gaps[:12])
            + (f" (+{len(gaps) - 12} more)" if len(gaps) > 12 else "")
        )


def read_budgets(path: str | Path, cfg: StudyConfig, sheet: str,
                 header_row: int = 1) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read per-account water budgets, folded into the same billing periods.

    Budgets already reflect each account's billing days, so unlike volumetric
    tier widths they are used as-is rather than day-prorated.
    """
    wb = Workbook(path)
    month_period = _month_to_period(cfg)
    years = set(cfg.fiscal_years)

    layout: dict[str, tuple[str, int]] = {}
    keys: list[str] = []
    rows: dict[str, list[np.ndarray]] = {fy: [] for fy in cfg.fiscal_years}

    for record in wb.table(sheet, header_row=header_row):
        if not layout:
            for label in record:
                if m := BUDGET_RE.match(label):
                    month, yy = m.groups()
                    if _fy_label(yy) in years and month in month_period:
                        layout[label] = (_fy_label(yy), month_period[month])

        key = str(record.get("LOCSVC_ID", "")).strip()
        if not key:
            continue
        keys.append(key)

        b = {fy: np.zeros(cfg.n_periods) for fy in cfg.fiscal_years}
        for label, (fy, period) in layout.items():
            b[fy][period] += _to_number(record.get(label))
        for fy in cfg.fiscal_years:
            rows[fy].append(b[fy])

    index = pd.DataFrame({"locsvc_id": keys})
    return index, {fy: np.vstack(rows[fy]) if rows[fy] else np.zeros((0, cfg.n_periods))
                   for fy in cfg.fiscal_years}


def attach_budgets(accounts: AccountData, budget_index: pd.DataFrame,
                   budget_grids: dict[str, np.ndarray], cfg: StudyConfig) -> pd.DataFrame:
    """Align budgets to the account table; return a per-account coverage report."""
    position = {key: i for i, key in enumerate(budget_index["locsvc_id"])}
    take = accounts.meta["locsvc_id"].map(position)
    found = take.notna().to_numpy()
    take_idx = take.fillna(0).astype(int).to_numpy()

    aligned: dict[str, np.ndarray] = {}
    for fy, grid in budget_grids.items():
        if len(grid) == 0:
            aligned[fy] = np.zeros((len(accounts), cfg.n_periods))
            continue
        picked = grid[take_idx]
        picked[~found] = 0.0
        aligned[fy] = picked
    accounts.budgets = aligned

    return pd.DataFrame({
        "locsvc_id": accounts.meta["locsvc_id"],
        "cust_class": accounts.meta["cust_class"],
        "in_budget_table": found,
    })


def derive_year(accounts: AccountData, cfg: StudyConfig) -> None:
    """Add each configured derived series (e.g. a multi-year average) in place."""
    for spec in cfg.derived_years:
        day_stack = np.stack([accounts.days[fy] for fy in spec.source_years])
        use_stack = np.stack([accounts.usage[fy] for fy in spec.source_years])
        accounts.days[spec.name] = _apply_rounding(day_stack.mean(axis=0), spec.days)
        accounts.usage[spec.name] = _apply_rounding(use_stack.mean(axis=0), spec.usage)
        if accounts.budgets:
            budget_stack = np.stack([accounts.budgets[fy] for fy in spec.source_years])
            accounts.budgets[spec.name] = _apply_rounding(budget_stack.mean(axis=0), spec.usage)


def _apply_rounding(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "roundup":
        return np.ceil(values)
    if mode == "rounddown":
        return np.floor(values)
    return excel_round(values)


def excel_round(values: np.ndarray, digits: int = 0) -> np.ndarray:
    """Excel's ROUND: half away from zero, not numpy's half-to-even."""
    scale = 10.0 ** digits
    scaled = np.asarray(values, dtype=float) * scale
    return np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) / scale
