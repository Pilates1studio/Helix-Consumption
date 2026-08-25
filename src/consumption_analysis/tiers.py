"""Flow each account's usage through a rate structure, one billing period at a time.

Tiers cascade: an account fills Tier 1 up to its allotment, spills the remainder
into Tier 2, and so on. Two things make the allotment account-specific rather
than a flat breakpoint:

  * billing days - a 63-day period earns a proportionally larger allotment than
    a 57-day one, so the tier boundary moves with each account's read cycle
  * dwelling units - multi-unit premises get the allotment once per unit

Budget-based classes replace the Tier 1 calculation with a per-account, per-period
allotment supplied from a water-budget table. Those budgets already reflect
billing days, so they are used as given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import N_TIERS, CustomerClass, RateSchedule, StudyConfig
from .ingest import excel_round


@dataclass
class Allocation:
    """Usage split by tier: `tiers` has shape (n_tiers, n_accounts, n_periods)."""

    tiers: np.ndarray
    usage: np.ndarray
    allotments: np.ndarray
    budget_substituted: np.ndarray | None = None

    @property
    def n_tiers(self) -> int:
        return self.tiers.shape[0]

    def total(self) -> np.ndarray:
        return self.tiers.sum(axis=0)


def tier_allotment(width: float | None, days: np.ndarray, units: np.ndarray,
                   days_per_period: int) -> np.ndarray:
    """ROUND(width / days_per_period x days, 0) x units - the Excel breakpoint."""
    if width is None:
        raise ValueError("volumetric tier is missing a width")
    scaled = excel_round(float(width) / days_per_period * days)
    return scaled * units[:, None]


def allocate(usage: np.ndarray, days: np.ndarray, units: np.ndarray,
             schedule: RateSchedule, cfg: StudyConfig,
             budgets: np.ndarray | None = None) -> Allocation:
    """Cascade usage through the tiers.

    The final tier takes whatever remains rather than applying its own cap, so
    the tiers always sum back to total usage even if the structure is misspecified.
    """
    usage = np.asarray(usage, dtype=float)
    tiers = np.zeros((N_TIERS, *usage.shape))
    allotments = np.zeros((N_TIERS, *usage.shape))
    remaining = usage.copy()

    substituted = None
    if budgets is not None:
        budgets, substituted = _fill_budget_gaps(budgets, usage, cfg.budget_gap_policy)

    for t in range(N_TIERS - 1):
        if t == 0 and budgets is not None:
            allot = budgets
        else:
            allot = tier_allotment(schedule.tier_widths[t], days, units, cfg.days_per_period)
        allot = np.maximum(allot, 0.0)
        allotments[t] = allot
        taken = np.minimum(allot, remaining)
        tiers[t] = taken
        remaining = remaining - taken

    tiers[N_TIERS - 1] = remaining
    allotments[N_TIERS - 1] = np.inf
    return Allocation(tiers=tiers, usage=usage, allotments=allotments,
                      budget_substituted=substituted)


def _fill_budget_gaps(budgets: np.ndarray, usage: np.ndarray,
                      policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Handle account-periods that show usage but carry no water budget.

    A zero budget on a period with no usage is ordinary - the account had not
    come into service yet - and is left alone. A zero budget against real usage
    is a data gap; under `cover_usage` the allotment is raised to meet the usage
    so it prices at the Tier 1 rate, the conservative assumption for revenue.
    """
    gap = (usage > 0) & (budgets <= 0)
    if policy == "cover_usage":
        budgets = np.where(gap, usage, budgets)
    elif policy != "as_is":
        raise ValueError(f"unknown budget_gap_policy {policy!r}")
    return budgets, gap


def allocate_class(klass: CustomerClass, schedule: RateSchedule, usage: np.ndarray,
                   days: np.ndarray, units: np.ndarray, cfg: StudyConfig,
                   budgets: np.ndarray | None = None) -> Allocation:
    if klass.is_budget_based:
        if budgets is None:
            raise ValueError(f"{klass.name} is budget-based but no budgets were supplied")
        return allocate(usage, days, units, schedule, cfg, budgets=budgets)
    return allocate(usage, days, units, schedule, cfg)
