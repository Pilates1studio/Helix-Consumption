"""Study configuration: everything that changes from one agency to the next."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

FISCAL_MONTHS = ["JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
                 "JAN", "FEB", "MAR", "APR", "MAY", "JUN"]

N_TIERS = 5


@dataclass(frozen=True)
class RateSchedule:
    """One side of the comparison - the existing rates, or the revised ones."""

    tier_widths: list[float | None]
    commodity_rates: list[float]
    meter_charges: dict[str, float]

    @classmethod
    def from_dict(cls, d: dict) -> "RateSchedule":
        widths = list(d.get("tier_widths", []))
        rates = list(d.get("commodity_rates", []))
        # Pad short definitions so every class carries the same tier count.
        widths += [None] * (N_TIERS - len(widths))
        rates += [0.0] * (N_TIERS - len(rates))
        return cls(
            tier_widths=widths[:N_TIERS],
            commodity_rates=[float(r) for r in rates[:N_TIERS]],
            meter_charges={str(k): float(v) for k, v in (d.get("meter_charges") or {}).items()},
        )

    def is_empty(self) -> bool:
        """True when no rates have been entered yet (revised, early in a study)."""
        return not any(self.commodity_rates) and not any(self.meter_charges.values())


@dataclass(frozen=True)
class CustomerClass:
    name: str
    allocation: str  # "volumetric" | "budget"
    existing: RateSchedule
    revised: RateSchedule

    @property
    def is_budget_based(self) -> bool:
        return self.allocation == "budget"


@dataclass(frozen=True)
class DerivedYear:
    name: str
    source_years: list[str]
    days: str = "roundup"
    usage: str = "rounddown"


@dataclass
class StudyConfig:
    agency: str
    title: str
    units: str
    days_per_period: int
    period_months: dict[str, list[str]]
    fiscal_years: list[str]
    derived_years: list[DerivedYear]
    selected_year: str
    rate_code_map: dict[str, str]
    excluded_rate_codes: list[str]
    meter_sizes: list[str]
    customer_classes: dict[str, CustomerClass]
    budget_gap_policy: str
    impact_buckets: dict = field(default_factory=dict)
    affordability: dict = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def periods(self) -> list[str]:
        return list(self.period_months)

    @property
    def n_periods(self) -> int:
        return len(self.period_months)

    @property
    def year_options(self) -> list[str]:
        """Fiscal years plus any derived series, in the order they are offered."""
        return list(self.fiscal_years) + [d.name for d in self.derived_years]

    @classmethod
    def load(cls, path: str | Path) -> "StudyConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf8"))
        study = raw.get("study", {})
        billing = raw["billing"]

        period_months = {k: [m.upper() for m in v] for k, v in billing["period_months"].items()}
        _validate_periods(period_months)

        classes = {}
        for name, spec in raw["customer_classes"].items():
            classes[name] = CustomerClass(
                name=name,
                allocation=spec.get("allocation", "volumetric"),
                existing=RateSchedule.from_dict(spec.get("existing", {})),
                revised=RateSchedule.from_dict(spec.get("revised", {})),
            )

        cfg = cls(
            agency=study.get("agency", "Agency"),
            title=study.get("title", "Consumption Analysis"),
            units=study.get("units", "hcf"),
            days_per_period=int(billing["days_per_period"]),
            period_months=period_months,
            fiscal_years=list(raw["fiscal_years"]),
            derived_years=[DerivedYear(**d) for d in raw.get("derived_years", [])],
            selected_year=raw["selected_year"],
            rate_code_map={str(k).strip(): v for k, v in raw["rate_code_map"].items()},
            excluded_rate_codes=[str(c).strip() for c in raw.get("excluded_rate_codes", [])],
            meter_sizes=list(raw["meter_sizes"]),
            customer_classes=classes,
            budget_gap_policy=raw.get("budget_gap_policy", "cover_usage"),
            impact_buckets=raw.get("impact_buckets", {}),
            affordability=raw.get("affordability", {}) or {},
            source_path=path,
        )

        if cfg.selected_year not in cfg.year_options:
            raise ValueError(
                f"selected_year {cfg.selected_year!r} is not one of {cfg.year_options}"
            )
        for derived in cfg.derived_years:
            missing = set(derived.source_years) - set(cfg.fiscal_years)
            if missing:
                raise ValueError(
                    f"derived year {derived.name!r} references unknown fiscal years: {sorted(missing)}"
                )
        unmapped = set(cfg.rate_code_map.values()) - set(cfg.customer_classes)
        if unmapped:
            raise ValueError(f"rate_code_map points at undefined customer classes: {sorted(unmapped)}")
        return cfg


def _validate_periods(period_months: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for period, months in period_months.items():
        for m in months:
            if m not in FISCAL_MONTHS:
                raise ValueError(f"{period}: {m!r} is not a fiscal month {FISCAL_MONTHS}")
            if m in seen:
                raise ValueError(f"month {m} appears in both {seen[m]} and {period}")
            seen[m] = period
    missing = [m for m in FISCAL_MONTHS if m not in seen]
    if missing:
        raise ValueError(f"period_months does not cover every month; missing {missing}")
