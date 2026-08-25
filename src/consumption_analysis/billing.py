"""Price each account's usage under a rate schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RateSchedule
from .tiers import Allocation


@dataclass
class Bills:
    """Per-account, per-period charges under one rate schedule."""

    fixed: np.ndarray       # (n_accounts, 1) meter charge, constant across periods
    commodity: np.ndarray   # (n_accounts, n_periods)

    @property
    def total(self) -> np.ndarray:
        return self.fixed + self.commodity

    def annual(self) -> np.ndarray:
        return self.total.sum(axis=1)


def meter_charges(meter_sizes: pd.Series, schedule: RateSchedule) -> np.ndarray:
    """Look up the fixed charge for each account's meter size.

    A size absent from the schedule prices at zero rather than raising - early in
    a study the revised schedule is deliberately blank.
    """
    lookup = schedule.meter_charges
    return meter_sizes.map(lambda s: lookup.get(s, 0.0)).fillna(0.0).to_numpy()[:, None]


def compute(alloc: Allocation, meter_sizes: pd.Series, schedule: RateSchedule) -> Bills:
    rates = np.asarray(schedule.commodity_rates, dtype=float)
    commodity = np.tensordot(rates, alloc.tiers, axes=(0, 0))
    return Bills(fixed=meter_charges(meter_sizes, schedule), commodity=commodity)


@dataclass
class BillImpact:
    """Existing vs revised, per account."""

    existing: Bills
    revised: Bills

    @property
    def delta_period(self) -> np.ndarray:
        return self.revised.total - self.existing.total

    @property
    def delta_annual(self) -> np.ndarray:
        return self.revised.annual() - self.existing.annual()

    @property
    def pct_annual(self) -> np.ndarray:
        base = self.existing.annual()
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(base != 0, self.delta_annual / base, 0.0)
        return np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RepresentativeBill:
    """One customer's bill for a single billing period, broken out by component."""

    meter_size: str
    usage: float
    days: float
    units: float
    fixed: float
    tier_usage: list[float]
    tier_charges: list[float]

    @property
    def variable(self) -> float:
        return float(sum(self.tier_charges))

    @property
    def total(self) -> float:
        return float(self.fixed + self.variable)

    def tier_table(self, rates: list[float], units_label: str) -> pd.DataFrame:
        rows = []
        for i, (used, charge) in enumerate(zip(self.tier_usage, self.tier_charges)):
            if used <= 0 and charge <= 0:
                continue
            rows.append({"Tier": f"Tier {i + 1}", f"Usage ({units_label})": used,
                         "Rate ($)": rates[i], "Charge ($)": charge})
        return pd.DataFrame(rows)


def representative_bill(usage: float, days: float, units: float, meter_size: str,
                        schedule, cfg, budget: float | None = None) -> RepresentativeBill:
    """Price a single representative bill through the same allocation engine.

    Routing this through `allocate` rather than re-implementing the cascade means
    the illustrative bill can never drift from the population-level analysis.
    """
    from .tiers import allocate

    alloc = allocate(
        np.array([[float(usage)]]), np.array([[float(days)]]), np.array([float(units)]),
        schedule, cfg,
        budgets=None if budget is None else np.array([[float(budget)]]),
    )
    tier_usage = [float(alloc.tiers[t][0, 0]) for t in range(alloc.n_tiers)]
    rates = schedule.commodity_rates
    return RepresentativeBill(
        meter_size=meter_size, usage=float(usage), days=float(days), units=float(units),
        fixed=float(schedule.meter_charges.get(meter_size, 0.0)),
        tier_usage=tier_usage,
        tier_charges=[u * r for u, r in zip(tier_usage, rates)],
    )


def bucket_counts(values: np.ndarray, edges: list[float],
                  label_format: str = "${:,.0f}") -> pd.DataFrame:
    """Count values falling in (edge[i-1], edge[i]], with an open final bucket.

    Mirrors the workbook's COUNTIFS histograms, which are exclusive at the lower
    bound and inclusive at the upper.

    `label_format` must suit the units being bucketed: percentage buckets are
    fractions, so formatting them as integers collapses every label to "0 - 0".
    """
    flat = np.asarray(values).ravel()
    fmt = label_format.format
    rows = []
    for i, upper in enumerate(edges):
        if i == 0:
            n = int((flat <= upper).sum())
            label = f"<= {fmt(upper)}"
        else:
            lower = edges[i - 1]
            n = int(((flat > lower) & (flat <= upper)).sum())
            label = f"{fmt(lower)} - {fmt(upper)}"
        rows.append({"range": label, "count": n})
    rows.append({"range": f"> {fmt(edges[-1])}", "count": int((flat > edges[-1]).sum())})
    frame = pd.DataFrame(rows)
    total = frame["count"].sum()
    frame["share"] = frame["count"] / total if total else 0.0
    return frame
