"""Bill burden by Census geography.

Joins the modelled bill to ACS household income and expresses the result as a
percent of income, by ZCTA or by tract. This is a *supplemental* analysis: it
carries no Proposition 218 burden, is not part of the cost-to-charge nexus, and
must never be represented as rate-setting justification. It exists to tell a
district where a proposed bill lands hardest, so an assistance programme can be
aimed rather than guessed at.

Three burden measures are produced, deliberately:

``burden_mhi``
    Annual bill divided by tract/ZCTA **median** household income. The
    convention every board and regulator recognises, and the basis for the
    California State Water Board's 1.5%-of-MHI screening threshold.

``burden_lq``
    Annual bill divided by the **mean household income of the lowest quintile**
    (ACS B19081_001). A bill that is 1.5% of the median household's income is
    typically 5-6% of the income of a household at the bottom of the
    distribution. Assistance programmes serve *those* households, so targeting
    off the median systematically misses poor households inside affluent
    geographies. This mirrors the low-income anchoring in EPA's 2023 Financial
    Capability Assessment and Teodoro's AR20.

``hours_min_wage``
    Annual bill divided by the state minimum wage. Unitless of income
    distribution entirely, and the measure lay audiences grasp fastest.

Report all three. The median measure is what a board expects; the quintile
measure is what actually identifies who needs help; where the two disagree is
the interesting part of the exhibit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Screening thresholds, as fractions of household income. Every one of these is
# a *convention*, not a legal standard, and the exhibit must say so.
THRESHOLDS = {
    # California State Water Board, Drinking Water Needs Assessment: annual
    # system-wide average residential charges for 6 HCF/month against annual
    # MHI. The on-point number for a California retail water agency.
    "ca_needs_assessment": 0.015,
    # US EPA drinking-water affordability criterion (Safe Drinking Water Act
    # small-system analysis), unchanged since the 1990s.
    "epa_water": 0.025,
    # EPA wastewater residential indicator, 1997 CSO Financial Capability
    # Assessment guidance.
    "epa_wastewater": 0.020,
    # Practitioner convention for combined water + wastewater. Only meaningful
    # for a combined utility; a water-only district should not use it.
    "epa_combined": 0.045,
}

CA_MINIMUM_WAGE_2026 = 16.90


@dataclass
class AffordabilityConfig:
    geography: str = "zcta"                    # "zcta" or "tract"
    crosswalk_key: str = "location_no"         # account column the ZIP/tract file joins on
    basis_classes: tuple[str, ...] = ("Residential",)
    bill_basis: str = "actual"                 # "actual" or "fixed_usage"
    fixed_usage_per_month: float = 6.0         # HCF/month; CA Needs Assessment convention
    per_unit_classes: tuple[str, ...] = ("Multifamily",)
    min_accounts: int = 25                     # suppress thin geographies
    minimum_wage: float = CA_MINIMUM_WAGE_2026
    # ACS dollars -> current dollars. ACS 5-year income is expressed in the
    # vintage's final-year dollars (2023 for the 2019-2023 release); the bill it
    # is compared against is a current-year bill, so left unindexed the burden
    # is overstated by cumulative inflation since the vintage. base/current are
    # CPI-U index levels for the metro area; enabled=False falls back to raw
    # ACS dollars.
    income_index: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=lambda: dict(THRESHOLDS))
    primary_threshold: str = "ca_needs_assessment"


# --------------------------------------------------------------------------
# 1. account-level bills
# --------------------------------------------------------------------------

def account_bills(result, classes: list[str] | None = None) -> pd.DataFrame:
    """One row per account: identity, class, and the annual bill under both
    schedules. Pulled from ``meta`` rather than ``account_detail`` so the
    service-location key survives for the geographic join.
    """
    frames = []
    for name, klass in result.classes.items():
        if classes and name not in classes:
            continue
        cols = [c for c in ("location_no", "locsvc_id", "meter_sz", "cust_class",
                            "dwelling_units") if c in klass.meta.columns]
        out = klass.meta[cols].copy()
        out["bill_existing"] = klass.bills.existing.annual()
        out["bill_revised"] = klass.bills.revised.annual()
        out["annual_usage"] = klass.existing.usage.sum(axis=1)
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def per_unit(bills: pd.DataFrame, cfg: AffordabilityConfig) -> pd.DataFrame:
    """Divide master-metered bills by dwelling units.

    A 40-unit master-metered building has one enormous bill and forty
    households behind it. Left undivided it reads as an unaffordable account
    and drags its geography's average to nonsense.
    """
    bills = bills.copy()
    units = bills.get("dwelling_units")
    if units is None:
        return bills
    mask = bills["cust_class"].isin(cfg.per_unit_classes) & (units > 1)
    for col in ("bill_existing", "bill_revised", "annual_usage"):
        bills.loc[mask, col] = bills.loc[mask, col] / units[mask]
    return bills


# --------------------------------------------------------------------------
# 2. geographic join
# --------------------------------------------------------------------------

def attach_geography(bills: pd.DataFrame, crosswalk: pd.DataFrame,
                     cfg: AffordabilityConfig) -> tuple[pd.DataFrame, dict]:
    """Join accounts to a geography and report the join quality.

    The report is not optional. An unmatched share above a few percent means
    the crosswalk is stale or keyed differently, and every geography average
    downstream is computed on a biased subset.
    """
    key = cfg.crosswalk_key
    if key not in bills.columns:
        raise KeyError(f"account table has no column {key!r} to join the crosswalk on")
    if key not in crosswalk.columns or "geoid" not in crosswalk.columns:
        raise KeyError(f"crosswalk must have columns {key!r} and 'geoid'")

    cross = crosswalk[[key, "geoid"]].copy()
    cross[key] = cross[key].astype(str).str.strip()
    cross["geoid"] = cross["geoid"].astype(str).str.strip()
    if cfg.geography == "zcta":
        cross["geoid"] = cross["geoid"].str[:5].str.zfill(5)
    cross = cross.drop_duplicates(subset=[key])

    joined = bills.copy()
    joined[key] = joined[key].astype(str).str.strip()
    joined = joined.merge(cross, on=key, how="left")

    matched = joined["geoid"].notna()
    report = {
        "accounts": int(len(joined)),
        "matched": int(matched.sum()),
        "unmatched": int((~matched).sum()),
        "match_rate": float(matched.mean()) if len(joined) else 0.0,
        "geographies": int(joined.loc[matched, "geoid"].nunique()),
    }
    return joined, report


# --------------------------------------------------------------------------
# 3. representative bill per geography
# --------------------------------------------------------------------------

def representative_by_geography(joined: pd.DataFrame,
                                cfg: AffordabilityConfig) -> pd.DataFrame:
    """Collapse accounts to one representative annual bill per geography.

    The **median** is used, not the mean. Annual bills are right-skewed — one
    estate or one mis-classified irrigation account moves a mean materially and
    a median not at all. The mean is carried alongside so the skew is visible
    rather than hidden.
    """
    frame = joined[joined["geoid"].notna()]
    if cfg.basis_classes:
        frame = frame[frame["cust_class"].isin(cfg.basis_classes)]
    billed = frame[frame["annual_usage"] > 0]

    agg = billed.groupby("geoid").agg(
        accounts=("bill_revised", "size"),
        bill_existing=("bill_existing", "median"),
        bill_revised=("bill_revised", "median"),
        bill_revised_mean=("bill_revised", "mean"),
        bill_revised_p25=("bill_revised", lambda s: s.quantile(0.25)),
        bill_revised_p75=("bill_revised", lambda s: s.quantile(0.75)),
        usage=("annual_usage", "median"),
    ).reset_index()

    agg["suppressed"] = agg["accounts"] < cfg.min_accounts
    agg["bill_change"] = agg["bill_revised"] - agg["bill_existing"]
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["bill_change_pct"] = np.where(
            agg["bill_existing"] > 0,
            agg["bill_change"] / agg["bill_existing"], np.nan)
    return agg


def fixed_usage_bill(cfg_study, schedule, usage_per_month: float,
                     meter_size: str) -> float:
    """Annual bill for a standard customer at a fixed monthly usage.

    The California Needs Assessment defines its affordability screen on 6
    HCF/month precisely so that income, not consumption, is the variable being
    compared across places. Use this basis when the exhibit's claim is "the
    same customer, in different neighbourhoods"; use the actual-bill basis when
    the claim is "what customers here actually pay".
    """
    from . import billing

    per_period = usage_per_month * cfg_study.period_months
    bill = billing.representative_bill(
        usage=per_period, days=cfg_study.days_per_period, units=1.0,
        meter_size=meter_size, schedule=schedule, cfg=cfg_study)
    return bill.total * cfg_study.n_periods


# --------------------------------------------------------------------------
# 4. burden
# --------------------------------------------------------------------------

def burden(agg: pd.DataFrame, acs: pd.DataFrame,
           cfg: AffordabilityConfig) -> pd.DataFrame:
    """Attach income and express the bill as a percent of it.

    Margins of error are carried through. At tract and ZCTA level an ACS median
    income routinely has a +/-10-20% band at 90% confidence; a geography whose
    burden band straddles the threshold has not been shown to exceed it, and
    the exhibit should not colour it as though it had.
    """
    acs = acs.copy()
    acs["geoid"] = acs["geoid"].astype(str).str.strip()
    if cfg.geography == "zcta":
        acs["geoid"] = acs["geoid"].str[:5].str.zfill(5)

    # Index income (and its margins of error — a ratio scales both) from ACS
    # vintage dollars to current dollars. The raw ACS figure is kept alongside
    # under an _acs suffix so the export shows both and the factor is auditable.
    idx = cfg.income_index or {}
    factor = (float(idx["current"]) / float(idx["base"])
              if idx.get("enabled") and idx.get("base") and idx.get("current")
              else 1.0)
    income_cols = [c for c in ("mhi", "mhi_moe", "lq_mean_income", "lq_upper_limit",
                               "mhi_owner", "mhi_renter", "mean_income")
                   if c in acs.columns]
    if factor != 1.0:
        for c in income_cols:
            acs[f"{c}_acs"] = acs[c]
            acs[c] = pd.to_numeric(acs[c], errors="coerce") * factor

    keep = [c for c in ("geoid", "NAME", "mhi", "mhi_moe", "mhi_acs", "mean_income",
                        "income_skew", "lq_mean_income",
                        "lq_mean_income_moe", "lq_upper_limit", "mhi_owner",
                        "mhi_renter", "population", "households",
                        "avg_household_size", "poverty_rate")
            if c in acs.columns]
    out = agg.merge(acs[keep], on="geoid", how="left")

    def ratio(numer, denom):
        denom = pd.to_numeric(denom, errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, numer / denom, np.nan)

    out["burden_mhi"] = ratio(out["bill_revised"], out.get("mhi"))
    out["burden_mhi_existing"] = ratio(out["bill_existing"], out.get("mhi"))
    if "lq_mean_income" in out:
        out["burden_lq"] = ratio(out["bill_revised"], out["lq_mean_income"])
    if "mean_income" in out:
        # Reported, but never the headline: mean household income exceeds the
        # median almost everywhere, so a burden measured against it is
        # systematically the friendlier number.
        out["burden_mean"] = ratio(out["bill_revised"], out["mean_income"])
    if "mhi_renter" in out:
        out["burden_renter"] = ratio(out["bill_revised"], out["mhi_renter"])
    out["hours_min_wage"] = out["bill_revised"] / cfg.minimum_wage

    # 90% confidence band on the median-income burden.
    if "mhi_moe" in out:
        lo_income = pd.to_numeric(out["mhi"], errors="coerce") + out["mhi_moe"]
        hi_income = pd.to_numeric(out["mhi"], errors="coerce") - out["mhi_moe"]
        out["burden_mhi_lo"] = ratio(out["bill_revised"], lo_income)
        out["burden_mhi_hi"] = ratio(out["bill_revised"], hi_income)

    # Thin geographies are blanked before any flag is derived from them, so a
    # suppressed row can never be reported as clearing or failing a threshold.
    thin = out["suppressed"].to_numpy()
    for col in ("burden_mhi", "burden_lq", "burden_mean", "burden_renter",
                "burden_mhi_lo", "burden_mhi_hi", "hours_min_wage"):
        if col in out.columns:
            out.loc[thin, col] = np.nan

    threshold = cfg.thresholds[cfg.primary_threshold]
    out["exceeds"] = (out["burden_mhi"] > threshold).astype("boolean")
    out.loc[out["burden_mhi"].isna(), "exceeds"] = pd.NA
    if {"burden_mhi_lo", "burden_mhi_hi"} <= set(out.columns):
        # A geography is only reported as exceeding when the whole 90% band
        # clears the threshold; one that straddles it is "uncertain", and the
        # map should say so rather than colour it as a finding.
        out["exceeds_confident"] = (out["burden_mhi_lo"] > threshold).astype("boolean")
        out["uncertain"] = ((out["burden_mhi_lo"] <= threshold)
                            & (out["burden_mhi_hi"] >= threshold)).astype("boolean")
        for col in ("exceeds_confident", "uncertain"):
            out.loc[out["burden_mhi"].isna(), col] = pd.NA
    return out


def summary(burden_table: pd.DataFrame, cfg: AffordabilityConfig) -> pd.DataFrame:
    """Headline counts for the tab, one row per threshold."""
    live = burden_table[~burden_table["suppressed"]]
    rows = []
    for name, value in cfg.thresholds.items():
        over = live["burden_mhi"] > value
        rows.append({
            "Threshold": name.replace("_", " "),
            "% of income": value,
            "Geographies over": int(over.sum()),
            "of": int(live["burden_mhi"].notna().sum()),
            "Accounts affected": int(live.loc[over, "accounts"].sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. inputs
# --------------------------------------------------------------------------

def load_crosswalk(path: str, key: str) -> pd.DataFrame:
    """Read the account -> geography crosswalk.

    Accepts whatever the agency sent: the key column may be named for the
    account, the service location, or the premise, and the geography column may
    be a ZIP, a ZCTA, or a tract GEOID. Only the two columns are kept, and both
    are read as text so leading zeros survive — a ZIP read as an integer is the
    single most common way this join silently loses a whole city.
    """
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [c.strip().lower() for c in frame.columns]
    key = key.lower()
    geo_candidates = ["geoid", "zcta", "zip", "zip_code", "zipcode",
                      "postal_code", "tract", "tract_geoid"]
    geo_col = next((c for c in geo_candidates if c in frame.columns), None)
    if geo_col is None:
        raise KeyError(f"{path}: no geography column found "
                       f"(looked for {', '.join(geo_candidates)})")
    if key not in frame.columns:
        raise KeyError(f"{path}: no key column {key!r}; has {list(frame.columns)}")
    out = frame[[key, geo_col]].rename(columns={geo_col: "geoid"})
    return out.dropna(subset=["geoid"])


def load_acs(path: str) -> pd.DataFrame:
    """Read the cached ACS pull written by ``tools/fetch_census.py``."""
    frame = pd.read_csv(path, dtype={"geoid": str})
    frame["geoid"] = frame["geoid"].astype(str).str.strip()
    return frame


def load_geometry(path: str) -> dict:
    """Read tract/ZCTA boundaries as GeoJSON.

    Cartographic-boundary files from the Census (the ``cb_*`` series) are the
    right source: they are generalised for mapping and an order of magnitude
    smaller than the full TIGER/Line files, which carry shoreline detail no
    board exhibit will ever resolve.
    """
    import json
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
