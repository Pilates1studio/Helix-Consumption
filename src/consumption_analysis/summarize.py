"""Roll per-account tier detail up into the tables a rate study reports on.

The distinction that matters here is between *usage by tier* and *usage stopped
in tier*. Usage by tier asks where each unit of water was priced. Usage stopped
in tier asks where each customer's bill ended - all of a customer's usage,
attributed to the highest tier they reached. The second is what supports
statements like "Tier 3 customers average 45 hcf a period", because it groups
customers by behaviour rather than slicing every bill across tiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import N_TIERS, StudyConfig
from .tiers import Allocation

TIER_LABELS = [f"Tier {i + 1}" for i in range(N_TIERS)]


@dataclass
class ClassSummary:
    """Every table the model produces for a single customer class."""

    name: str
    year: str
    periods: list[str]
    n_accounts: int
    usage_by_tier: pd.DataFrame
    usage_stopped_in_tier: pd.DataFrame
    contributing_accounts: pd.DataFrame
    usage_per_account: pd.DataFrame
    meter_counts: pd.DataFrame
    total_usage: pd.Series
    no_usage_accounts: pd.Series

    def check(self) -> pd.DataFrame:
        """Reconciliations that must hold if the allocation is sound.

        Every unit of water is priced in exactly one tier and attributed to
        exactly one stopping tier, so those two totals must agree. And in each
        billing period every account either contributes to some tier or has no
        usage, so the two counts must add back to the class.
        """
        by_tier = self.usage_by_tier.loc["Total"]
        stopped = self.usage_stopped_in_tier.loc["Total"]
        accounts = self.contributing_accounts.loc["Total"]
        # Counts are per period; the "Total" column sums account-periods, so the
        # head-count reconciliation is only meaningful period by period.
        reconciles = {p: accounts[p] + self.no_usage_accounts[p] == self.n_accounts
                      for p in self.periods}
        reconciles["Total"] = all(reconciles.values())
        return pd.DataFrame({
            "usage by tier": by_tier,
            "usage stopped in tier": stopped,
            "tiers reconcile": np.isclose(by_tier, stopped),
            "accounts + no-usage": accounts + self.no_usage_accounts,
            "accounts reconcile": pd.Series(reconciles),
        })


def _frame(matrix: np.ndarray, periods: list[str], index: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=index, columns=periods)
    frame["Total"] = frame.sum(axis=1)
    frame.loc["Total"] = frame.sum(axis=0)
    return frame


def stopped_masks(alloc: Allocation) -> np.ndarray:
    """(n_tiers, n_accounts, n_periods) - True where a bill's last tier is t.

    A bill "stops" in the highest tier it reaches: tier t carries usage while
    tier t+1 is empty. The top tier stops wherever it has any usage at all.
    """
    tiers = alloc.tiers
    masks = np.zeros_like(tiers, dtype=bool)
    for t in range(alloc.n_tiers - 1):
        masks[t] = (tiers[t] > 0) & (tiers[t + 1] == 0)
    masks[-1] = tiers[-1] > 0
    return masks


def summarize_class(name: str, year: str, alloc: Allocation, meta: pd.DataFrame,
                    cfg: StudyConfig) -> ClassSummary:
    periods = cfg.periods
    tiers = alloc.tiers
    masks = stopped_masks(alloc)

    # Usage priced in each tier.
    by_tier = tiers.sum(axis=1)

    # All usage from bills that stopped in each tier - cumulative through that
    # tier, matching the workbook's SUMIF/SUMIFS pair.
    cumulative = np.cumsum(tiers, axis=0)
    stopped = np.array([(cumulative[t] * masks[t]).sum(axis=0) for t in range(alloc.n_tiers)])
    contributing = masks.sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_account = np.where(contributing > 0, stopped / contributing, 0.0)

    usage_by_tier = _frame(by_tier, periods, TIER_LABELS)
    usage_stopped = _frame(stopped, periods, TIER_LABELS)
    accounts_frame = _frame(contributing, periods, TIER_LABELS)

    # Per-account intensity is a ratio, so neither rows nor columns may be summed;
    # margins are recomputed from the underlying totals.
    per_account_frame = pd.DataFrame(per_account, index=TIER_LABELS, columns=periods)
    totals_stopped = stopped.sum(axis=1)
    totals_accounts = contributing.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_account_frame["Total"] = np.where(totals_accounts > 0,
                                              totals_stopped / totals_accounts, 0.0)
        # "Total" row: usage per account across all tiers, i.e. every account
        # that used water in that period. This is the class-level series the
        # peaking factors are built from.
        class_stopped = stopped.sum(axis=0)
        class_accounts = contributing.sum(axis=0)
        per_account_frame.loc["Total"] = np.append(
            np.where(class_accounts > 0, class_stopped / class_accounts, 0.0),
            (totals_stopped.sum() / totals_accounts.sum()) if totals_accounts.sum() else 0.0,
        )

    meter_counts = (
        meta["meter_sz"].value_counts()
            .reindex(cfg.meter_sizes, fill_value=0)
            .rename_axis("Meter Size").rename("Accounts").to_frame()
    )
    meter_counts.loc["Total"] = meter_counts.sum()

    no_usage = pd.Series((alloc.usage <= 0).sum(axis=0).astype(float), index=periods)
    no_usage["Total"] = no_usage.sum()

    total_usage = pd.Series(alloc.usage.sum(axis=0), index=periods)
    total_usage["Total"] = total_usage.sum()

    return ClassSummary(
        name=name, year=year, periods=periods, n_accounts=len(meta),
        usage_by_tier=usage_by_tier,
        usage_stopped_in_tier=usage_stopped,
        contributing_accounts=accounts_frame,
        usage_per_account=per_account_frame,
        meter_counts=meter_counts,
        total_usage=total_usage,
        no_usage_accounts=no_usage,
    )


def peaking_factors(series_by_period: pd.Series, periods: list[str],
                    peak_period: str | None = None) -> dict[str, float]:
    """Max-to-average ratio, the peaking factor a cost-of-service study consumes.

    `peak_period` pins the numerator to a chosen period - normally the system
    peak - rather than letting each class or tier peak on its own period. Pinning
    matters because a class that peaks in a different period than the system does
    not drive the system's peak-capacity costs.
    """
    values = series_by_period[periods].to_numpy(dtype=float)
    average = values.mean() if len(values) else 0.0
    if peak_period and peak_period in periods:
        peak = values[periods.index(peak_period)]
    else:
        peak = values.max(initial=0.0)
    return {"peak": float(peak), "average": float(average),
            "peaking_factor": float(peak / average) if average else 0.0}


def system_peak_period(class_usage: dict[str, pd.Series], periods: list[str]) -> str:
    totals = pd.DataFrame({k: v[periods] for k, v in class_usage.items()}).sum(axis=1)
    return str(totals.idxmax())


def active_tiers(summary: ClassSummary) -> list[str]:
    """Tiers that actually carry usage - the only ones worth a peaking factor."""
    totals = summary.usage_by_tier["Total"]
    return [t for t in TIER_LABELS if totals.get(t, 0) > 0]
