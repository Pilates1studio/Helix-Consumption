"""Run the consumption analysis end to end for a study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import billing, store, summarize
from .billing import BillImpact, Bills
from .config import StudyConfig
from .ingest import AccountData
from .summarize import ClassSummary
from .tiers import Allocation, allocate_class

# Bucket edges carry different units: dollar changes per bill, and annual
# percentage changes expressed as fractions.
DOLLAR_LABEL = "${:,.0f}"
PERCENT_LABEL = "{:.0%}"


@dataclass
class ClassResult:
    name: str
    year: str
    meta: pd.DataFrame
    existing: Allocation
    revised: Allocation
    bills: BillImpact
    summary_existing: ClassSummary
    summary_revised: ClassSummary
    days: np.ndarray | None = None
    budgets: np.ndarray | None = None

    def account_detail(self, periods: list[str]) -> pd.DataFrame:
        """One row per account: usage, tier split, and both bills."""
        out = self.meta[["locsvc_id", "meter_sz", "rate_cd", "cust_class",
                         "dwelling_units"]].copy()
        out["usage"] = self.existing.usage.sum(axis=1)
        for t in range(self.existing.n_tiers):
            out[f"existing_tier_{t + 1}"] = self.existing.tiers[t].sum(axis=1)
            out[f"revised_tier_{t + 1}"] = self.revised.tiers[t].sum(axis=1)
        out["bill_existing"] = self.bills.existing.annual()
        out["bill_revised"] = self.bills.revised.annual()
        out["bill_change"] = self.bills.delta_annual
        out["bill_change_pct"] = self.bills.pct_annual
        return out

    def profile(self, meter_size: str | None = None,
                extra_mask: "np.ndarray | None" = None) -> dict[str, float]:
        """Average per-period usage, billing days and budget for a meter-size cohort.

        This is the basis for the illustrative bill: the average billing period of
        the customers actually in that class and meter size, rather than a figure
        picked by hand. Averages are taken over account-periods that carry a bill,
        so vacant periods and closed accounts do not drag the representative
        customer below what a real one looks like.

        ``extra_mask`` narrows the cohort further — e.g. to the accounts in one
        Census tract — so the representative customer can be localised without
        changing how the average is defined.
        """
        mask = (np.ones(len(self.meta), dtype=bool) if not meter_size
                else (self.meta["meter_sz"] == meter_size).to_numpy())
        if extra_mask is not None:
            mask = mask & np.asarray(extra_mask, dtype=bool)
        n = int(mask.sum())
        empty = {"accounts": 0, "usage": 0.0, "annual_usage": 0.0,
                 "days": 0.0, "units": 1.0, "budget": 0.0}
        if not n:
            return empty

        usage = self.existing.usage[mask]
        billed = usage > 0
        if not billed.any():
            return empty | {"accounts": n}

        days = self.days[mask] if self.days is not None else None
        day_values = days[billed] if days is not None else np.array([])
        day_values = day_values[day_values > 0]

        budget = 0.0
        if self.budgets is not None:
            b = self.budgets[mask][billed]
            b = b[b > 0]
            budget = float(b.mean()) if b.size else 0.0

        return {
            "accounts": n,
            "usage": float(usage[billed].mean()),
            "annual_usage": float(usage.sum(axis=1)[usage.sum(axis=1) > 0].mean()),
            "days": float(day_values.mean()) if day_values.size else 0.0,
            "units": float(self.meta.loc[mask, "dwelling_units"].mean()),
            "budget": budget,
        }

    def meter_sizes_present(self) -> list[str]:
        counts = self.meta["meter_sz"].value_counts()
        return [str(s) for s in counts.index]

    def impact_tables(self, cfg: StudyConfig) -> dict[str, pd.DataFrame]:
        buckets = cfg.impact_buckets or {}
        dollars = (buckets.get("bill_dollars") or {}).get(self.name)
        percents = buckets.get("account_percent")
        out: dict[str, pd.DataFrame] = {}
        if dollars:
            out["bill_impacts"] = billing.bucket_counts(
                self.bills.delta_period, dollars, DOLLAR_LABEL)
        if percents:
            out["account_impacts"] = billing.bucket_counts(
                self.bills.pct_annual, percents, PERCENT_LABEL)
        return out


@dataclass
class StudyResult:
    cfg: StudyConfig
    year: str
    classes: dict[str, ClassResult]

    @property
    def class_names(self) -> list[str]:
        return list(self.classes)

    def usage_by_class(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {name: r.summary_existing.total_usage for name, r in self.classes.items()}
        ).T
        frame.loc["Total"] = frame.sum(axis=0)
        return frame

    def tier_summary(self, revised: bool = False) -> pd.DataFrame:
        """Every class's usage-by-tier stacked into one table."""
        frames = []
        for name, result in self.classes.items():
            summary = result.summary_revised if revised else result.summary_existing
            frame = summary.usage_by_tier.drop(index="Total").copy()
            frame.insert(0, "Customer Class", name)
            frame.index.name = "Tier"
            frames.append(frame.reset_index())
        return pd.concat(frames, ignore_index=True)

    def system_peak_period(self) -> str:
        usage = {n: r.summary_existing.total_usage for n, r in self.classes.items()}
        return summarize.system_peak_period(usage, self.cfg.periods)

    def peaking(self, basis: str = "per_account", pin_to_system: bool = True
                ) -> pd.DataFrame:
        """Class-level peaking factors.

        The default basis is usage per contributing account, which measures how
        much a typical customer's demand swells in the peak period. Total class
        usage conflates that with growth or attrition in the customer count, so
        it is the weaker signal for allocating peak-capacity cost.
        """
        periods = self.cfg.periods
        peak_period = self.system_peak_period() if pin_to_system else None
        rows = []
        for name, res in self.classes.items():
            summary = res.summary_existing
            series = (summary.usage_per_account.loc["Total"] if basis == "per_account"
                      else summary.total_usage)
            stats = summarize.peaking_factors(series, periods, peak_period)
            rows.append({
                "Customer Class": name,
                "Total Usage": summary.total_usage["Total"],
                "Peak Period": peak_period or max(periods, key=lambda p: series[p]),
                **stats,
            })
        return pd.DataFrame(rows)

    def tier_peaking(self, class_name: str, basis: str = "per_account",
                     pin_to_system: bool = True) -> pd.DataFrame:
        """Peaking factors per tier, for classes with a tiered structure.

        Higher tiers are seasonal demand - outdoor irrigation, mostly - so they
        peak far harder than Tier 1, which is closer to year-round indoor use.
        That spread is the point: it is what justifies allocating peaking costs
        disproportionately to the upper tiers.
        """
        periods = self.cfg.periods
        summary = self.classes[class_name].summary_existing
        peak_period = self.system_peak_period() if pin_to_system else None
        rows = []
        for tier in summarize.active_tiers(summary):
            series = (summary.usage_per_account.loc[tier] if basis == "per_account"
                      else summary.usage_by_tier.loc[tier])
            stats = summarize.peaking_factors(series, periods, peak_period)
            rows.append({
                "Tier": tier,
                "Usage": summary.usage_by_tier.loc[tier, "Total"],
                "Accounts": summary.contributing_accounts.loc[tier, "Total"],
                "Peak Period": peak_period or max(periods, key=lambda p: series[p]),
                **stats,
            })
        return pd.DataFrame(rows)

    def peak_contribution(self) -> pd.DataFrame:
        """Each class's peak responsibility measured against the system as a whole.

        A class's own peaking factor says how much its demand swells, but not
        whether that swelling is unusual: if every class peaked identically, none
        of them would be responsible for more of the system's peak than its share
        of annual volume. What matters for cost allocation is *deviation from the
        system*, so each class's factor is normalised by the system's own:

            relative factor = class peaking factor / system peaking factor

        1.000 means the class swells exactly as the system does and carries its
        volume share of peak cost; above 1.000 it drives the peak harder than
        average and should carry more.

        Two identities make this defensible and easy to explain:
          * relative factor == (share of peak-period usage) / (share of annual usage)
          * allocating on class volume x relative factor reduces exactly to each
            class's share of peak-period usage

        Everything is measured on usage per contributing account, so a class is
        compared with the system's typical customer rather than with its own
        volume. The peak period itself is the system's actual peak - the period
        that delivered the most water - since that is the physical event the
        capacity has to meet.

        Peak cost is then allocated on class volume weighted by that relative
        factor, so a class carries its volume share adjusted for how much harder
        or softer than average its customers swell.
        """
        periods = self.cfg.periods
        if not self.classes:
            return pd.DataFrame()

        volume, accounts = {}, {}
        for name, res in self.classes.items():
            summary = res.summary_existing
            volume[name] = summary.total_usage[periods].to_numpy(dtype=float)
            accounts[name] = summary.contributing_accounts.loc["Total", periods].to_numpy(dtype=float)

        system_volume = sum(volume.values())
        system_accounts = sum(accounts.values())
        peak_index = int(system_volume.argmax())

        with np.errstate(divide="ignore", invalid="ignore"):
            system_series = np.where(system_accounts > 0,
                                     system_volume / system_accounts, 0.0)
        system_peak = float(system_series[peak_index])
        system_average = float(system_series.mean())
        system_factor = system_peak / system_average if system_average else 0.0
        system_total = float(system_volume.sum())

        rows = []
        for name in self.classes:
            with np.errstate(divide="ignore", invalid="ignore"):
                series = np.where(accounts[name] > 0, volume[name] / accounts[name], 0.0)
            average = float(series.mean())
            at_peak = float(series[peak_index])
            own = at_peak / average if average else 0.0
            relative = own / system_factor if system_factor else 0.0
            total = float(volume[name].sum())
            rows.append({
                "Customer Class": name,
                "Accounts at Peak": float(accounts[name][peak_index]),
                "Peak per Account": at_peak,
                "Average per Account": average,
                "Class Peaking Factor": own,
                "Relative Peaking Factor": relative,
                "Total Usage": total,
                "Share of Usage": total / system_total if system_total else 0.0,
                # Carried as a column, not just an intermediate: it is what makes
                # the allocation auditable by hand - weighted usage over the sum
                # of weighted usage reproduces the percentage.
                "Weighted Usage": total * relative,
            })

        frame = pd.DataFrame(rows)
        weighted = frame["Weighted Usage"].sum()
        frame["Peak Cost Allocation"] = (frame["Weighted Usage"] / weighted if weighted
                                         else 0.0)
        frame.attrs.update(
            peak_period=periods[peak_index], system_factor=system_factor,
            system_peak=system_peak, system_average=system_average,
            system_accounts=float(system_accounts[peak_index]),
        )
        return frame

    def peak_reconciliation(self) -> pd.DataFrame:
        """The two checks a reviewer will run against the relative-factor framing.

        1. Under a volume-weighted, renormalised allocation the relative factor
           changes nothing: a constant divisor applied to every class cancels. The
           relative framing therefore *confirms* the conventional allocation rather
           than replacing it, which is the stronger position to argue from.

        2. Under a base-extra capacity formulation the "- 1" is load-bearing and
           the divisor no longer cancels. Each class's own factor recovers
           essentially the whole system excess; the relative factors sum to
           approximately zero, because they average to 1.000 by construction, and
           so cannot allocate a positive extra-capacity pool at all. Extra capacity
           must be struck on the class's own factor.
        """
        frame = self.peak_contribution()
        if frame.empty:
            return frame
        system_factor = frame.attrs["system_factor"]
        volume = frame["Total Usage"]
        own = frame["Class Peaking Factor"]
        relative = frame["Relative Peaking Factor"]

        own_weighted = volume * own
        rel_weighted = volume * relative
        own_alloc = own_weighted / own_weighted.sum() if own_weighted.sum() else 0.0
        rel_alloc = rel_weighted / rel_weighted.sum() if rel_weighted.sum() else 0.0

        out = pd.DataFrame({
            "Customer Class": frame["Customer Class"],
            "Own Factor": own,
            "Relative Factor": relative,
            "Allocation on Own Factor": own_alloc,
            "Allocation on Relative Factor": rel_alloc,
            "Difference": own_alloc - rel_alloc,
            "Extra Capacity on Own Factor": volume * (own - 1.0),
            "Extra Capacity on Relative Factor": volume * (relative - 1.0),
        })
        system_excess = float(volume.sum() * (system_factor - 1.0))
        own_excess = float((volume * (own - 1.0)).sum())
        rel_excess = float((volume * (relative - 1.0)).sum())
        out.attrs.update(
            system_factor=system_factor,
            max_allocation_difference=float(out["Difference"].abs().max()),
            system_excess=system_excess,
            own_excess=own_excess,
            relative_excess=rel_excess,
            own_excess_share=own_excess / system_excess if system_excess else 0.0,
            relative_excess_share=rel_excess / system_excess if system_excess else 0.0,
        )
        return out

    def tier_peak_contribution(self, class_name: str) -> pd.DataFrame:
        """Distribute a class's assigned peak cost across its tiered subclasses.

        Second step of a sequential allocation. Step one has already assigned each
        class its share of system peak cost, so the frame of reference here is the
        class alone - dividing by the system again would re-apply an adjustment
        already made.

        Tier usage is not measured against its own pattern through the year. Upper
        tiers *are* peak usage by construction, and judging them against their own
        annual average would let a tier that is consistently high register as not
        peaking at all. Instead the peak period is the only period examined, and
        each tier is compared with the class average in that same period:

            factor = subclass usage per account in the peak period
                     / class usage per account in the peak period

        The accounts that stopped in a tier during the peak period are that
        tier's subclass, so the subclasses partition the class exactly. Two
        consequences: the account-weighted average of the factors is exactly
        1.000, and the resulting allocation reduces to each subclass's share of
        class usage in the peak period.
        """
        summary = self.classes[class_name].summary_existing
        peak_period = self.peak_contribution().attrs["peak_period"]

        class_usage = float(summary.total_usage[peak_period])
        class_accounts = float(summary.contributing_accounts.loc["Total", peak_period])
        class_per_account = class_usage / class_accounts if class_accounts else 0.0

        rows = []
        for tier in summarize.active_tiers(summary):
            accounts = float(summary.contributing_accounts.loc[tier, peak_period])
            usage = float(summary.usage_stopped_in_tier.loc[tier, peak_period])
            per_account = usage / accounts if accounts else 0.0
            rows.append({
                "Tier": tier,
                "Accounts in Peak": accounts,
                "Usage in Peak": usage,
                "Usage per Account": per_account,
                "Factor vs Class": per_account / class_per_account if class_per_account else 0.0,
                "Annual Usage": float(summary.usage_by_tier.loc[tier, "Total"]),
            })

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        # accounts x factor renormalised; equals share of peak-period usage.
        weights = frame["Accounts in Peak"] * frame["Factor vs Class"]
        total = weights.sum()
        frame["Share of Class Peak Cost"] = weights / total if total else 0.0
        annual = frame["Annual Usage"].sum()
        frame["Share of Annual Usage"] = frame["Annual Usage"] / annual if annual else 0.0
        frame.attrs.update(class_name=class_name, peak_period=peak_period,
                           class_per_account=class_per_account,
                           class_usage=class_usage, class_accounts=class_accounts)
        return frame

    def seasonal_index_by_period(self) -> pd.DataFrame:
        """How each class's per-account demand moves against the system, per period.

            index = (class per account / class average) / (system per account / system average)

        Both sides are expressed relative to their own average, so differences in
        account size cancel - a master-metered multifamily account and a single
        home are compared on shape, not level. 1.000 means the class moved exactly
        with the system that period; above 1.000 it ran hotter than the system did.

        The value in the system's peak period is the Relative Peaking Factor, so
        this table is that headline number extended across the whole year.
        """
        periods = self.cfg.periods
        volume, accounts = {}, {}
        for name, res in self.classes.items():
            summary = res.summary_existing
            volume[name] = summary.total_usage[periods].to_numpy(dtype=float)
            accounts[name] = summary.contributing_accounts.loc["Total", periods].to_numpy(dtype=float)

        def per_account(vol, acc):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(acc > 0, vol / np.where(acc > 0, acc, 1), 0.0)

        system_series = per_account(sum(volume.values()), sum(accounts.values()))
        system_average = system_series.mean()
        system_shape = (system_series / system_average if system_average
                        else np.zeros_like(system_series))

        rows = {}
        for name in self.classes:
            series = per_account(volume[name], accounts[name])
            average = series.mean()
            shape = series / average if average else np.zeros_like(series)
            with np.errstate(divide="ignore", invalid="ignore"):
                rows[name] = np.where(system_shape > 0, shape / system_shape, 0.0)
        return pd.DataFrame(rows, index=periods).T

    def has_tiers(self, class_name: str) -> bool:
        return len(summarize.active_tiers(self.classes[class_name].summary_existing)) > 1

    def combined_impacts(self) -> dict[str, np.ndarray]:
        """Bill changes pooled across every class, for a whole-system view."""
        return {
            "delta_period": np.concatenate(
                [r.bills.delta_period.ravel() for r in self.classes.values()]),
            "pct_annual": np.concatenate(
                [r.bills.pct_annual.ravel() for r in self.classes.values()]),
            "existing_annual": sum(r.bills.existing.annual().sum()
                                   for r in self.classes.values()),
            "revised_annual": sum(r.bills.revised.annual().sum()
                                  for r in self.classes.values()),
            "accounts": sum(r.summary_existing.n_accounts for r in self.classes.values()),
        }

    def checks(self) -> pd.DataFrame:
        frames = []
        for name, result in self.classes.items():
            frame = result.summary_existing.check()
            frame.insert(0, "Customer Class", name)
            frames.append(frame.reset_index(names="Tier"))
        return pd.concat(frames, ignore_index=True)


def run(cfg: StudyConfig, accounts: AccountData, year: str | None = None,
        classes: list[str] | None = None) -> StudyResult:
    year = year or cfg.selected_year
    if year not in accounts.usage:
        raise KeyError(f"no usage series for {year!r}; have {sorted(accounts.usage)}")

    wanted = classes or list(cfg.customer_classes)
    results: dict[str, ClassResult] = {}

    for name in wanted:
        klass = cfg.customer_classes[name]
        mask = (accounts.meta["cust_class"] == name).to_numpy()
        if not mask.any():
            continue
        meta = accounts.meta[mask].reset_index(drop=True)
        usage = accounts.usage[year][mask]
        days = accounts.days[year][mask]
        units = meta["dwelling_units"].to_numpy(dtype=float)
        budgets = None
        if klass.is_budget_based:
            if not accounts.budgets or year not in accounts.budgets:
                raise ValueError(f"{name} is budget-based but no budgets are loaded for {year}")
            budgets = accounts.budgets[year][mask]

        existing = allocate_class(klass, klass.existing, usage, days, units, cfg, budgets)
        revised = allocate_class(klass, klass.revised, usage, days, units, cfg, budgets)

        impact = BillImpact(
            existing=billing.compute(existing, meta["meter_sz"], klass.existing),
            revised=billing.compute(revised, meta["meter_sz"], klass.revised),
        )

        results[name] = ClassResult(
            name=name, year=year, meta=meta, existing=existing, revised=revised,
            bills=impact, days=days, budgets=budgets,
            summary_existing=summarize.summarize_class(name, year, existing, meta, cfg),
            summary_revised=summarize.summarize_class(name, year, revised, meta, cfg),
        )

    return StudyResult(cfg=cfg, year=year, classes=results)


def load_study(config_path: str | Path, cache_path: str | Path
               ) -> tuple[StudyConfig, AccountData]:
    cfg = StudyConfig.load(config_path)
    return cfg, store.load(cfg, cache_path)
