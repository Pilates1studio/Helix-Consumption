"""Staff review build — Helix consumption analysis (Cons-Bill Calc-Hosting).

    streamlit run src/consumption_analysis/app.py

Locked-down variant of the internal app for named agency reviewers (e.g.
Jennifer and Timothy at Helix) to sign off on rates before the Board sees
them. Full parity with the internal tool for exploration — usage filtering,
tiers, peaking, bill impact, and the full affordability heat map — but the
Rates tab's entry grids, "Apply to session," and "Download config block"
do not exist in this build. Rate and rate-structure changes stay with Beeb;
this is view-only by construction, not by convention. Pinned to Helix's own
config and cache — never globs clients/*/config.yaml, so no other agency's
data can ever appear here even if this repo is later reused as a template.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from consumption_analysis import (affordability_tab, billing, model, report,
                                      store, summarize, theme)
    from consumption_analysis.affordability_tab import _paths as _aff_paths
    from consumption_analysis.affordability_tab import _tract_label
    from consumption_analysis.config import StudyConfig
except ModuleNotFoundError:  # running the file directly, package not installed
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from consumption_analysis import (affordability_tab, billing, model, report,
                                      store, summarize, theme)
    from consumption_analysis.affordability_tab import _paths as _aff_paths
    from consumption_analysis.affordability_tab import _tract_label
    from consumption_analysis.config import StudyConfig

ROOT = Path(__file__).resolve().parent.parent.parent
AGENCY_SLUG = "helix"  # pinned — this build only ever serves Helix

st.set_page_config(page_title="Helix Consumption Analysis — Staff Review", layout="wide")
theme.inject(st)


def _require_passcode() -> None:
    """Gate the whole app behind a shared passcode for the named reviewers.

    Set via the PASSCODE env var on Render; never hardcoded in source so it
    isn't sitting in plaintext in the git history.
    """
    expected = os.environ.get("PASSCODE")
    if not expected:
        st.error("PASSCODE is not configured on this deployment — set it as an "
                 "env var before sharing this link.")
        st.stop()
    if st.session_state.get("authed"):
        return
    st.title("Helix Consumption Analysis — Staff Review")
    entered = st.text_input("Passcode", type="password")
    if st.button("Enter") or entered:
        if entered == expected:
            st.session_state["authed"] = True
            st.rerun()
        elif entered:
            st.error("Incorrect passcode.")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(config_path: str, cache_path: str):
    cfg = StudyConfig.load(config_path)
    return cfg, store.load(cfg, cache_path)


def _active_meter_sizes(cfg: StudyConfig, class_names: list[str]) -> list[str]:
    """Meter sizes that actually carry a fixed charge somewhere in this study.

    Structural, the same way tier suppression is: a size priced at $0 for
    every class on both the existing and revised side (10" and 12" for
    Helix — no customer holds one) is padding in the config, not a real
    meter size, so it is dropped rather than shown as a row of zeros.
    """
    return [size for size in cfg.meter_sizes
            if any(getattr(cfg.customer_classes[name], side).meter_charges.get(size, 0.0)
                   for name in class_names for side in ("existing", "revised"))]


def _fixed_charge_matrix(cfg: StudyConfig, class_names: list[str], side: str,
                         sizes: list[str]) -> pd.DataFrame:
    """Meter size x customer class grid of fixed charges."""
    data = {
        name: [getattr(cfg.customer_classes[name], side).meter_charges.get(size, 0.0)
               for size in sizes]
        for name in class_names
    }
    return pd.DataFrame(data, index=pd.Index(sizes, name="Meter Size"))


def _rate_active_tiers(rates) -> int:
    """How many tiers of a rate schedule actually carry a nonzero commodity rate.

    Rate-based rather than usage-based: this is what "how many tiers does this
    class have" means structurally, and — unlike a usage count — it reads the
    same for the revised schedule even before anyone has been billed under it.
    A class always has at least one priced tier.
    """
    return max(sum(1 for r in rates if r), 1)


def _class_active_tiers(cfg: StudyConfig, name: str) -> int:
    """Active tier count for a class: the larger of its existing/revised sides,
    so a widened structure on one side is never trimmed out of view."""
    klass = cfg.customer_classes[name]
    return max(_rate_active_tiers(klass.existing.commodity_rates),
               _rate_active_tiers(klass.revised.commodity_rates))


def _max_active_tiers(cfg: StudyConfig, class_names: list[str]) -> int:
    return max((_class_active_tiers(cfg, name) for name in class_names), default=1)


def _variable_rate_matrix(cfg: StudyConfig, class_names: list[str], side: str) -> pd.DataFrame:
    """Commodity rates by tier, trimmed to the widest active structure among
    the classes shown. A class with fewer active tiers than that leaves the
    remaining rows blank rather than showing its padding zeros."""
    n_max = _max_active_tiers(cfg, class_names)
    tiers = [f"Tier {i + 1}" for i in range(n_max)]
    data = {}
    for name in class_names:
        rates = list(getattr(cfg.customer_classes[name], side).commodity_rates)
        n_active = _class_active_tiers(cfg, name)
        data[name] = [rates[i] if i < n_active else np.nan for i in range(n_max)]
    return pd.DataFrame(data, index=pd.Index(tiers, name="Tier"))


def _tier_structure_matrix(cfg: StudyConfig, class_names: list[str], side: str) -> pd.DataFrame:
    """Usage allotment per tier per class, in plain language rather than raw widths.

    A class with only one active tier reads as "Uniform" — there is no tier
    structure to show, just a single rate. The last active tier is always the
    one that structurally absorbs everything the tiers below it don't — so
    rather than print its padding-sized width (e.g. 9999999), it is labelled
    as the catch-all it actually is. Tiers beyond a class's active count are
    blank: they carry a zero rate and never see real usage.
    """
    n_max = _max_active_tiers(cfg, class_names)
    rows = [f"Tier {i + 1}" for i in range(n_max)]
    data = {}
    for name in class_names:
        widths = getattr(cfg.customer_classes[name], side).tier_widths
        n_active = _class_active_tiers(cfg, name)
        col = []
        for i in range(n_max):
            if i >= n_active:
                col.append("—")
            elif n_active == 1:
                col.append("Uniform")
            elif i == n_active - 1:
                col.append(f"All usage > Tier {n_active - 1}")
            else:
                w = widths[i]
                col.append("Budget" if w is None else f"{float(w):,.0f} {cfg.units}")
        data[name] = col
    return pd.DataFrame(data, index=pd.Index(rows, name="Tier"))


def _change_frame(existing: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    """Interleave $ and % change so each class reads as a pair of columns."""
    out = pd.DataFrame(index=existing.index)
    for col in existing.columns:
        delta = revised[col] - existing[col]
        # A zero existing charge has no meaningful percent change; show it blank
        # rather than an infinity that reads as a real number.
        base = existing[col].where(existing[col] != 0)
        out[f"{col} $"] = delta
        out[f"{col} %"] = delta / base
    return out


def sidebar_inputs() -> tuple[str, str]:
    """Pinned to Helix — deliberately not a picker.

    A client-facing build must never glob clients/*/config.yaml the way the
    internal tool does: that would surface every agency in this repo to
    whoever holds this link. If this repo is ever reused as a template for
    another agency, change AGENCY_SLUG at the top of this file rather than
    reintroducing a selector.
    """
    config_path = ROOT / "clients" / AGENCY_SLUG / "config.yaml"
    cache_path = ROOT / "build" / f"{AGENCY_SLUG}.parquet"
    if not config_path.exists() or not cache_path.exists():
        st.error(f"Missing config or cached account table for '{AGENCY_SLUG}'.")
        st.stop()
    return str(config_path), str(cache_path)


def main() -> None:
    _require_passcode()
    config_path, cache_path = sidebar_inputs()
    cfg, accounts = _load(config_path, cache_path)

    st.sidebar.markdown("---")
    st.sidebar.header("Usage series")
    available = [y for y in cfg.year_options if y in accounts.usage]
    year = st.sidebar.selectbox(
        "Fiscal year", available, key="year",
        index=available.index(cfg.selected_year) if cfg.selected_year in available else 0)

    class_names = [n for n in cfg.customer_classes if (accounts.meta["cust_class"] == n).any()]
    st.sidebar.markdown("---")
    st.sidebar.header("Customer class")
    selected = st.sidebar.selectbox("Class", class_names, key="class")

    st.sidebar.markdown("---")
    st.sidebar.header("Peaking basis")
    basis_label = st.sidebar.radio(
        "Measure peaking on", ["Usage per contributing account", "Total class usage"],
        key="peaking_basis",
        help="Per-account isolates how much a typical customer's demand swells in "
             "the peak period; total usage conflates that with customer growth.")
    basis = "per_account" if basis_label.startswith("Usage per") else "total"
    pin = st.sidebar.radio("Peak period", ["System peak", "Each class's own peak"],
                           key="peak_period_mode") == "System peak"

    # No rate-override mechanism in this build — active_cfg is always the
    # locked config as deployed. Rate changes are Beeb's to make and redeploy.
    active_cfg = cfg

    result = model.run(active_cfg, accounts, year=year)

    st.title(f"{cfg.agency} — {cfg.title}")
    st.caption(f"{year} · {cfg.n_periods} billing periods · "
               f"{cfg.days_per_period} days per period · units: {cfg.units}")

    # Reconciliations run on every rerun but stay out of the way unless one
    # fails, in which case every number on screen is suspect and it says so.
    checks = result.checks()
    failed = checks[~checks["tiers reconcile"] | ~checks["accounts reconcile"]]
    if not failed.empty:
        st.error(f"{len(failed)} reconciliation(s) failed — figures below are not "
                 "trustworthy until this is resolved.")
        st.dataframe(failed, width='stretch')

    tabs = st.tabs(["Overview", "Rates", f"{selected} — tiers", "Peaking",
                    "Peak Contribution", "Bill Impact", "Impact distribution",
                    f"{selected} — accounts", "Affordability", "Export"])
    with tabs[0]:
        _overview(result, cfg)
    with tabs[1]:
        _rates(active_cfg, class_names, selected)
    with tabs[2]:
        _tier_detail(result, cfg, selected)
    with tabs[3]:
        _peaking(result, cfg, selected, basis, pin)
    with tabs[4]:
        _peak_contribution(result, cfg)
    with tabs[5]:
        _bill_impact(result, active_cfg, class_names, selected)
    with tabs[6]:
        _impact_distribution(result, active_cfg, class_names, selected)
    with tabs[7]:
        _accounts(result, cfg, selected)
    with tabs[8]:
        affordability_tab.render(result, cfg)
    with tabs[9]:
        _export(result, cfg, year)


def _rates(cfg: StudyConfig, class_names: list[str], default_class: str) -> None:
    """Full rate schedule: fixed charges by meter size, variable rates by class."""
    money = "${:,.2f}"

    st.subheader("Fixed charges by meter size")
    st.caption("Charged every billing period regardless of usage. These drive the "
               "fixed portion of every bill on the Bill Impact tab.")
    active_sizes = _active_meter_sizes(cfg, class_names)
    existing_fixed = _fixed_charge_matrix(cfg, class_names, "existing", active_sizes)
    revised_fixed = _fixed_charge_matrix(cfg, class_names, "revised", active_sizes)

    # A uniform fixed-charge schedule is the norm; per-class schedules are the
    # anomaly. When every class carries the same charges, the per-class columns
    # say nothing — collapse to one Rate column and let a real difference be
    # the thing that brings the class columns back.
    def _uniform(frame):
        return frame.eq(frame.iloc[:, 0], axis=0).all().all()
    if _uniform(existing_fixed) and _uniform(revised_fixed):
        existing_fixed = existing_fixed.iloc[:, [0]].set_axis(["Rate"], axis=1)
        revised_fixed = revised_fixed.iloc[:, [0]].set_axis(["Rate"], axis=1)

    # width='content' — not 'stretch' — so this table sizes to its own columns
    # instead of filling the page. A Meter Size + Rate (+ Change) grid stretched
    # to full width leaves the Rate column stranded far from its row labels.
    view = st.radio("Show", ["Existing", "Proposed", "Change"], horizontal=True,
                    key="rates_fixed_view", label_visibility="collapsed")
    if view == "Existing":
        st.dataframe(existing_fixed.style.format(money), width='content')
    elif view == "Proposed":
        if revised_fixed.to_numpy().sum() == 0:
            st.warning("No proposed fixed charges entered yet.")
        st.dataframe(revised_fixed.style.format(money), width='content')
    else:
        change = _change_frame(existing_fixed, revised_fixed)
        fmt = {c: (money if c.endswith("$") else "{:+.1%}") for c in change.columns}
        st.dataframe(change.style.format(fmt, na_rep="—"), width='content')

    st.markdown("---")
    st.subheader("Variable rates by customer class")
    st.caption("Per-unit commodity rates. Helix's are blended — each includes the "
               "$3.91/hcf San Diego County Water Authority pass-through.")
    existing_var = _variable_rate_matrix(cfg, class_names, "existing")
    revised_var = _variable_rate_matrix(cfg, class_names, "revised")

    var_view = st.radio("Show", ["Existing", "Proposed", "Change"], horizontal=True,
                        key="rates_var_view", label_visibility="collapsed")
    if var_view == "Existing":
        st.dataframe(existing_var.style.format(money, na_rep="—"), width='stretch')
    elif var_view == "Proposed":
        st.dataframe(revised_var.style.format(money, na_rep="—"), width='stretch')
    else:
        change = _change_frame(existing_var, revised_var)
        fmt = {c: (money if c.endswith("$") else "{:+.1%}") for c in change.columns}
        st.dataframe(change.style.format(fmt, na_rep="—"), width='stretch')

    st.markdown("---")
    st.subheader("Usage tiers by customer class")
    st.caption("How usage is allotted within each class's variable rate. A class "
               "with a single active tier bills one uniform rate; the last tier "
               "shown always catches all remaining usage, so it reads as such "
               "rather than as a placeholder width.")
    existing_tiers = _tier_structure_matrix(cfg, class_names, "existing")
    revised_tiers = _tier_structure_matrix(cfg, class_names, "revised")
    if var_view == "Existing":
        st.dataframe(existing_tiers, width='stretch')
    elif var_view == "Proposed":
        st.dataframe(revised_tiers, width='stretch')
    elif existing_tiers.equals(revised_tiers):
        st.dataframe(existing_tiers, width='stretch')
        st.caption("Tier boundaries are unchanged between existing and proposed "
                   "rates — only the commodity rates above change.")
    else:
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Existing**")
            st.dataframe(existing_tiers, width='stretch')
        with t2:
            st.markdown("**Proposed**")
            st.dataframe(revised_tiers, width='stretch')

    st.markdown("---")
    st.caption("Rates and rate structure are locked in this review build. To review a "
               "different alternative, ask Beeb to deploy it.")


def _overview(result: model.StudyResult, cfg: StudyConfig) -> None:
    usage = result.usage_by_class()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total usage", f"{usage.loc['Total', 'Total']:,.0f} {cfg.units}")
    c2.metric("Accounts", f"{sum(r.summary_existing.n_accounts for r in result.classes.values()):,}")
    c3.metric("Classes", len(result.classes))

    st.subheader("Total usage by customer class")
    st.altair_chart(theme.total_bars(usage, "Customer Class", cfg.units),
                    width='stretch')
    with st.expander("Usage by billing period"):
        st.dataframe(usage.style.format("{:,.0f}"), width='stretch')


def _tier_detail(result: model.StudyResult, cfg: StudyConfig, name: str) -> None:
    if name not in result.classes:
        st.info(f"No accounts in {name}.")
        return
    summary = result.classes[name].summary_existing
    klass = cfg.customer_classes[name]
    if not result.has_tiers(name):
        st.info(f"{name} is on a uniform rate — there is only one tier, so "
                "there is nothing to break out here.")
        return
    st.caption("Budget-based: Tier 1 is each account's water-budget allotment."
               if klass.is_budget_based else
               f"Volumetric: tier widths prorated over {cfg.days_per_period} days "
               "and multiplied by dwelling units.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts", f"{summary.n_accounts:,}")
    c2.metric("Usage", f"{summary.total_usage['Total']:,.0f} {cfg.units}")
    c3.metric("Meters", f"{summary.meter_counts.loc['Total', 'Accounts']:,.0f}")

    # Trimmed to the tiers that actually carry usage — a class's unused
    # padding tiers (always zero here) add nothing but noise to every table
    # below.
    active = summarize.active_tiers(summary)
    rows = active + ["Total"]

    st.subheader(f"Total usage by tier ({cfg.units})")
    st.caption("Where each unit of water was priced.")
    st.altair_chart(theme.total_bars(summary.usage_by_tier.loc[active], "Tier", cfg.units),
                    width='stretch')
    with st.expander("Usage by tier and billing period"):
        st.dataframe(summary.usage_by_tier.loc[rows].style.format("{:,.0f}"), width='stretch')

    st.subheader(f"Usage stopped in tier ({cfg.units})")
    st.caption("All usage from bills that ended in each tier — this is what groups "
               "customers into Tier 1 / Tier 2 / Tier 3 cohorts.")
    st.dataframe(summary.usage_stopped_in_tier.loc[rows].style.format("{:,.0f}"),
                 width='stretch')

    left, right = st.columns(2)
    with left:
        st.subheader("Contributing accounts")
        st.dataframe(summary.contributing_accounts.loc[rows].style.format("{:,.0f}"),
                     width='stretch')
    with right:
        st.subheader(f"Usage per contributing account ({cfg.units})")
        st.dataframe(summary.usage_per_account.loc[rows].style.format("{:,.1f}"),
                     width='stretch')


def _peaking(result: model.StudyResult, cfg: StudyConfig, name: str,
             basis: str, pin: bool) -> None:
    label = ("usage per contributing account" if basis == "per_account"
             else "total class usage")
    st.caption(f"Peaking factor = peak period ÷ average period, measured on {label}. "
               + ("Numerator pinned to the system peak period."
                  if pin else "Each class or tier peaks on its own period."))

    st.subheader("By customer class")
    peaking = result.peaking(basis=basis, pin_to_system=pin).set_index("Customer Class")
    display = peaking.rename(columns={"peak": "Peak", "average": "Average",
                                      "peaking_factor": "Peaking Factor"})
    st.dataframe(
        display.style.format({"Total Usage": "{:,.0f}", "Peak": "{:,.1f}",
                              "Average": "{:,.1f}", "Peaking Factor": "{:.3f}"}),
        width='stretch')
    st.altair_chart(
        theme.peaking_bars(display.reset_index(), "Customer Class", "Peaking Factor"),
        width='stretch')

    st.subheader(f"By tier — {name}")
    if name not in result.classes:
        st.info(f"No accounts in {name}.")
        return
    if not result.has_tiers(name):
        st.info(f"{name} is on a uniform rate — there are no tiers to compare. "
                "Select a tiered class in the sidebar.")
        return

    tiers = result.tier_peaking(name, basis=basis, pin_to_system=pin).set_index("Tier")
    tier_display = tiers.rename(columns={"peak": "Peak", "average": "Average",
                                         "peaking_factor": "Peaking Factor"})
    st.dataframe(
        tier_display.style.format({"Usage": "{:,.0f}", "Accounts": "{:,.0f}",
                                   "Peak": "{:,.1f}", "Average": "{:,.1f}",
                                   "Peaking Factor": "{:.3f}"}),
        width='stretch')
    st.altair_chart(theme.peaking_bars(tier_display.reset_index(), "Tier", "Peaking Factor"),
                    width='stretch')
    st.caption("Upper tiers are seasonal demand and peak harder than Tier 1, which is "
               "closer to year-round indoor use. That spread is what supports "
               "allocating peaking costs disproportionately to the upper tiers.")


def _peak_contribution(result: model.StudyResult, cfg: StudyConfig) -> None:
    """Peak responsibility measured as deviation from the system, not self-reference."""
    frame = result.peak_contribution()
    if frame.empty:
        st.info("No classes to compare.")
        return
    meta = frame.attrs

    st.caption(
        "A class's own peaking factor says how much its demand swells, but not "
        "whether that swelling is unusual. If every class peaked identically, none "
        "would be responsible for more of the system peak than its share of volume. "
        "Here each class's factor is divided by the system's own, so **1.000 means "
        "the class moves exactly with the system** and carries its volume share of "
        "peak cost. Above 1.000 it drives the peak harder and should carry more. "
        "All figures are usage per contributing account.")

    c1, c2, c3 = st.columns(3)
    c1.metric("System peak period", meta["peak_period"])
    c2.metric("System peaking factor", f"{meta['system_factor']:.3f}",
              help=f"{meta['system_peak']:,.1f} {cfg.units} per account at peak vs "
                   f"{meta['system_average']:,.1f} on average")
    c3.metric("Accounts at peak", f"{meta['system_accounts']:,.0f}")

    st.subheader("Relative peaking factor by class")
    chart = frame[["Customer Class", "Relative Peaking Factor"]].copy()
    st.altair_chart(
        theme.peaking_bars(chart, "Customer Class", "Relative Peaking Factor"),
        width='stretch')
    st.caption("The dashed line is the system at 1.000.")

    st.subheader("Peak responsibility")
    display = frame.set_index("Customer Class")
    st.dataframe(display.style.format({
        "Accounts at Peak": "{:,.0f}",
        "Peak per Account": "{:,.1f}",
        "Average per Account": "{:,.1f}",
        "Class Peaking Factor": "{:.3f}",
        "Relative Peaking Factor": "{:.3f}",
        "Total Usage": "{:,.0f}",
        "Share of Usage": "{:.2%}",
        "Weighted Usage": "{:,.0f}",
        "Peak Cost Allocation": "{:.2%}",
    }), width='stretch')
    st.caption(
        f"**Weighted Usage** = Total Usage × Relative Peaking Factor. Dividing it by "
        f"the column total ({frame['Weighted Usage'].sum():,.0f} {cfg.units}) reproduces "
        f"Peak Cost Allocation, so the percentage can be checked by hand.")

    st.subheader("Who gains and who loses")
    st.caption(
        "What each class would carry on volume alone, against what it carries once "
        "peak responsibility is weighted in. A class below 1.000 uses water more "
        "evenly through the year and does not drive the peak as hard, so it pays a "
        "smaller share of peak-related cost than its volume would suggest. Only the "
        "peak-related component moves — this is not a shift in the whole bill.")
    st.altair_chart(
        theme.share_comparison_bars(
            frame[["Customer Class", "Share of Usage", "Peak Cost Allocation"]],
            "Customer Class", "Share of Usage", "Peak Cost Allocation"),
        width='stretch')
    st.caption(
        "**Peak Cost Allocation** = (class usage × its relative factor) ÷ the sum of "
        "that product across all classes. So it is the volume share, re-weighted by "
        "how hard each class pushes the peak, then renormalised to 100%. Compare it "
        "against Share of Usage to see who gains and who loses under this basis.")

    shift = ((display["Peak Cost Allocation"] - display["Share of Usage"]) * 100)
    movers = shift.abs().sort_values(ascending=False)
    if len(movers):
        top = movers.index[0]
        st.info(f"Largest shift: **{top}** moves "
                f"{shift[top]:+.2f} percentage points against a straight volume "
                f"allocation ({display.loc[top, 'Share of Usage']:.2%} of usage → "
                f"{display.loc[top, 'Peak Cost Allocation']:.2%} of peak cost).")

    st.markdown("---")
    with st.expander("Reconciliation — what a reviewer will ask about the relative factor"):
        recon = result.peak_reconciliation()
        rmeta = recon.attrs
        st.markdown(
            "**Check 1 — does the relative factor change the allocation?** No. Under a "
            "volume-weighted, renormalised allocation a constant divisor applied to every "
            "class cancels out. The relative framing **confirms** the conventional "
            "allocation rather than replacing it, which is the stronger position to argue "
            "from: present it as the nexus narrative and a fairness test, not as a "
            "different set of dollars.")
        st.dataframe(recon.set_index("Customer Class")[[
            "Own Factor", "Relative Factor", "Allocation on Own Factor",
            "Allocation on Relative Factor", "Difference"]].style.format({
                "Own Factor": "{:.4f}", "Relative Factor": "{:.4f}",
                "Allocation on Own Factor": "{:.6%}",
                "Allocation on Relative Factor": "{:.6%}",
                "Difference": "{:.8%}"}), width='stretch')
        st.caption(f"Largest difference: **{rmeta['max_allocation_difference']:.10%}**.")
        # Check 2 (extra-capacity guard: relative factors must not be used for
        # base-extra capacity, since the −1 is load-bearing there) is computed
        # in model.peak_reconciliation and documented in the model log, but is
        # not displayed — dashboard shows Check 1 only, per Beeb.

    st.markdown("---")
    st.subheader("Step 2 — distributing a class's peak cost across its tiered subclasses")
    st.caption(
        "Step 1 assigned each class its share of system peak cost. Inside the class "
        "the frame of reference is the class itself. Upper tiers **are** peak usage "
        "by construction, so they are not judged against their own pattern through "
        "the year — that would let a consistently high tier register as not peaking "
        "at all. Only the peak period is examined, and each tier is compared with "
        "the class average in that same period. The accounts stopping in a tier are "
        "that tier's subclass, so the subclasses partition the class exactly.")

    tiered = [n for n in result.class_names if result.has_tiers(n)]
    if not tiered:
        st.info("No class has more than one active tier.")
    else:
        pick = st.selectbox("Customer class", tiered, key="tier_peak_class")
        tiers = result.tier_peak_contribution(pick)
        meta = tiers.attrs
        relative = (result.peak_contribution().set_index("Customer Class")
                    .loc[pick, "Relative Peaking Factor"])
        st.caption(
            f"If every {pick} account used the same amount of water — the uniform case "
            f"behind {pick}'s relative peak of **{relative:.4f}** — each account would "
            f"use **{meta['class_per_account']:,.5f}** {cfg.units} in "
            f"{meta['peak_period']} ({meta['class_usage']:,.0f} ÷ "
            f"{meta['class_accounts']:,.0f} contributing accounts). But {pick} accounts "
            f"in {meta['peak_period']} include customers falling in each tier. Grouping "
            f"accounts by the tier they reached gives each tier's share of overall "
            f"{meta['peak_period']} demand, allocating {pick}'s peak costs to tiers in "
            f"relation to how each contributes to the {pick} total in {meta['peak_period']}.")
        st.dataframe(tiers.set_index("Tier").style.format({
            "Accounts in Peak": "{:,.0f}",
            "Usage in Peak": "{:,.0f}",
            "Usage per Account": "{:,.2f}",
            "Factor vs Class": "{:.3f}",
            "Annual Usage": "{:,.0f}",
            "Share of Class Peak Cost": "{:.2%}",
            "Share of Annual Usage": "{:.2%}",
        }), width='stretch')
        st.altair_chart(
            theme.peaking_bars(tiers[["Tier", "Factor vs Class"]],
                               "Tier", "Factor vs Class"),
            width='stretch')
        weighted = ((tiers["Accounts in Peak"] * tiers["Factor vs Class"]).sum()
                    / tiers["Accounts in Peak"].sum())
        st.caption(
            f"Account-weighted average of the factors = **{weighted:.6f}** — exactly "
            f"1.000 by construction, since the subclasses partition the class. "
            f"**Share of Class Peak Cost** is accounts × factor renormalised, which "
            f"reduces to each subclass's share of class usage in {meta['peak_period']}. "
            f"Compare it against Share of Annual Usage to see the effect of allocating "
            f"on peak-period behaviour rather than annual volume.")

        if cfg.customer_classes[pick].is_budget_based:
            st.info(
                f"**Budget-based class — expect factors near 1.000.** Tier 1 is each "
                f"account's water budget, and the budget is itself seasonal: it rises "
                f"and falls with weather so that a customer irrigating normally stays "
                f"inside it all year. Within-budget demand therefore moves with the "
                f"class rather than against it, and the tier split separates "
                f"within-budget from over-budget usage rather than light users from "
                f"heavy ones. Compressed factors are the structure working, not an "
                f"anomaly.\n\n"
                f"Read \"near 1.000\" as *proportionate*, not *exempt*. Within-budget "
                f"usage is the most seasonal water in the class in absolute terms — the "
                f"system still has to build for it — so it carries a full proportionate "
                f"share of peak cost, not a reduced one.")

    st.markdown("---")
    st.subheader("Seasonal index by period")
    st.caption(
        "Each class's per-account demand relative to the system's, with both "
        "expressed against their own average so account size cancels out. "
        "1.000 means the class moved exactly with the system that period. The "
        f"value in **{meta['peak_period']}** is the relative peaking factor above.")
    st.dataframe(result.seasonal_index_by_period().style.format("{:.3f}")
                 .map(theme.deviation_shading),
                 width='stretch')
    st.caption(
        "Shading reads distance from 1.000, deepening with the gap: "
        "**:green[green] = ran hotter than the system** that period, "
        "**grey = ran cooler**, unshaded = moved with it. It encodes nothing the "
        "numbers do not already say — it is there to make each class's seasonal "
        "shape visible across a row at a glance.")


def _bill_segments(bills: dict) -> pd.DataFrame:
    """Long frame for the bill chart: fixed charge, then one band per priced tier.

    A tier is banded only where it carries a charge in at least one scenario, so a
    structure that never reaches Tier 3 does not draw an empty band or add a dead
    legend entry.
    """
    charged = sorted({
        t for bill in bills.values()
        for t, amount in enumerate(bill.tier_charges) if amount > 0
    })
    rows = []
    for scenario, bill in bills.items():
        rows.append({"Scenario": scenario, "Segment": "Fixed charge",
                     "Kind": "Fixed charge", "Amount": bill.fixed})
        for t in charged:
            rows.append({"Scenario": scenario, "Segment": f"Tier {t + 1}",
                         "Kind": "Variable charge", "Amount": bill.tier_charges[t]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _tract_lookup(crosswalk_path: str) -> dict:
    """location_no -> tract GEOID from the affordability crosswalk, plus labels.

    Cached on path: the 6 MB crosswalk should be read once per session, not on
    every widget interaction.
    """
    frame = pd.read_csv(crosswalk_path, dtype=str)
    frame.columns = [c.strip().lower() for c in frame.columns]
    if "location_no" not in frame.columns or "geoid" not in frame.columns:
        return {}
    frame = frame[frame["geoid"].str.len() == 11].drop_duplicates("location_no")
    return dict(zip(frame["location_no"], frame["geoid"]))


@st.cache_data(show_spinner=False)
def _tract_mhi_table(acs_path: str, factor: float) -> dict:
    """geoid -> CPI-indexed median household income, from the ACS cache."""
    frame = pd.read_csv(acs_path, dtype={"geoid": str})
    if "mhi" not in frame.columns:
        return {}
    mhi = pd.to_numeric(frame["mhi"], errors="coerce") * factor
    return {g: v for g, v in zip(frame["geoid"].str.strip(), mhi) if pd.notna(v)}


def _tract_mhi(cfg: StudyConfig, geoid: str) -> float | None:
    """Indexed median household income for one tract, or None when absent."""
    paths = _aff_paths(cfg, "tract")
    acs_path = paths.get("acs")
    if not acs_path or not os.path.exists(acs_path):
        return None
    idx = (cfg.affordability or {}).get("income_index") or {}
    factor = (float(idx["current"]) / float(idx["base"])
              if idx.get("enabled") and idx.get("base") and idx.get("current")
              else 1.0)
    return _tract_mhi_table(acs_path, factor).get(str(geoid))


def _tract_median_bill(result: model.StudyResult, cfg: StudyConfig,
                       geoid: str, lookup: dict) -> float | None:
    """Median proposed annual bill of the tract's burden-basis accounts.

    Mirrors the Affordability tab exactly — same basis classes, billed
    accounts only, median — so "Avg Burden of Tract" here equals "Burden"
    there for the same tract, whatever usage the illustrative bill is set to.
    """
    aff_cfg = cfg.affordability or {}
    options = aff_cfg.get("class_options") or {}
    classes = (list(next(iter(options.values())))
               if options else list(aff_cfg.get("basis_classes") or ["Residential"]))
    bills = []
    for name in classes:
        res = result.classes.get(name)
        if res is None:
            continue
        geoids = res.meta["location_no"].astype(str).map(lookup)
        mask = ((geoids == geoid)
                & (res.existing.usage.sum(axis=1) > 0)).to_numpy()
        if mask.any():
            bills.append(pd.Series(res.bills.revised.annual()[mask]))
    if not bills:
        return None
    return float(pd.concat(bills).median())


def _bill_impact(result: model.StudyResult, cfg: StudyConfig,
                 class_names: list[str], default_class: str) -> None:
    st.caption("Illustrative bill for one customer over a single billing period, "
               "priced under both rate structures.")

    # Tract filter — the bridge to the Affordability tab: the representative
    # customer can be localised to one tract, so a burden seen on the map can
    # be priced here on that neighbourhood's own average usage.
    tract_lookup = {}
    tract_paths = _aff_paths(cfg, "tract")
    if tract_paths.get("crosswalk") and os.path.exists(tract_paths["crosswalk"]):
        tract_lookup = _tract_lookup(tract_paths["crosswalk"])

    c1, c2, c3 = st.columns(3)
    klass_name = c1.selectbox("Customer type", class_names,
                              index=class_names.index(default_class)
                              if default_class in class_names else 0,
                              key="bi_class")
    res = result.classes[klass_name]
    sizes = res.meter_sizes_present()
    if not sizes:
        st.info(f"No accounts in {klass_name}.")
        return
    meter_size = c2.selectbox("Meter size", sizes, key="bi_size")

    DISTRICT = "Entire district"
    tract_mask = None
    tract_pick = DISTRICT
    if tract_lookup:
        geoids = res.meta["location_no"].astype(str).map(tract_lookup)
        present = sorted(geoids.dropna().unique())
        tract_pick = c3.selectbox("Census tract", [DISTRICT] + present,
                                  key="bi_tract", format_func=lambda g:
                                  g if g == DISTRICT else _tract_label(g))
        if tract_pick != DISTRICT:
            tract_mask = (geoids == tract_pick).to_numpy()

    profile = res.profile(meter_size, extra_mask=tract_mask)
    if not profile["accounts"]:
        st.info("No accounts with that meter size"
                + ("" if tract_pick == DISTRICT else " in that tract") + ".")
        return

    avg_usage = round(profile["usage"], 1)
    avg_days = round(profile["days"]) or cfg.days_per_period
    klass = cfg.customer_classes[klass_name]

    key = f"bi_usage_{klass_name}_{meter_size}_{tract_pick}"
    if key not in st.session_state:
        st.session_state[key] = float(avg_usage)

    st.markdown("---")
    u1, u2, u3, u4 = st.columns([2, 2, 2, 1.4])
    usage = u1.number_input(f"Usage per period ({cfg.units})", min_value=0.0,
                            step=1.0, key=key)
    days = u2.number_input("Billing days", min_value=1, max_value=200,
                           value=int(avg_days), step=1, key=f"bi_days_{klass_name}_{meter_size}")
    units = u3.number_input("Dwelling units", min_value=1.0, step=1.0,
                            value=float(round(profile["units"]) or 1),
                            key=f"bi_units_{klass_name}_{meter_size}")
    u4.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)

    def revert_usage() -> None:
        # Must run as a callback: Streamlit forbids writing to a widget's
        # session_state entry after that widget has been instantiated in the
        # same run, which is what raised on the button press.
        st.session_state[key] = float(avg_usage)

    u4.button("Revert to average", width='stretch', on_click=revert_usage,
              key=f"bi_revert_{klass_name}_{meter_size}")

    budget = None
    if klass.is_budget_based:
        budget = st.number_input(
            f"Tier 1 water budget for the period ({cfg.units})", min_value=0.0, step=1.0,
            value=float(round(profile["budget"], 1)),
            help="Budget-based classes take Tier 1 from the water-budget table rather "
                 "than a tier width. Defaults to the average for this selection.")

    where = ("" if tract_pick == DISTRICT
             else f" in {_tract_label(tract_pick)}")
    note = (f"{profile['accounts']:,} accounts on a {meter_size} meter{where} · "
            f"average usage {avg_usage:,.1f} {cfg.units} per period "
            f"over {avg_days} billing days")
    if abs(usage - avg_usage) > 1e-9:
        note += "  ·  **usage overridden**"
    st.caption(note)

    existing = billing.representative_bill(usage, days, units, meter_size,
                                           klass.existing, cfg, budget)
    proposed = billing.representative_bill(usage, days, units, meter_size,
                                           klass.revised, cfg, budget)

    delta = proposed.total - existing.total
    pct = (delta / existing.total) if existing.total else 0.0

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Existing bill", f"${existing.total:,.2f}")
    m2.metric("Proposed bill", f"${proposed.total:,.2f}")
    m3.metric("Difference", f"${delta:,.2f}", delta=f"${delta:,.2f}")
    m4.metric("Percent change", f"{pct:+.1%}", delta=f"{pct:+.1%}")

    frame = _bill_segments({"Existing": existing, "Proposed": proposed})
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(theme.bill_comparison(frame, height=470), width='stretch')
    with right:
        st.subheader("Components")
        comparison = pd.DataFrame({
            "Existing": [existing.fixed, existing.variable, existing.total],
            "Proposed": [proposed.fixed, proposed.variable, proposed.total],
        }, index=["Fixed charge", "Variable charge", "Total"])
        # Change columns are shown for the Total only, deliberately: rate
        # design often shifts recovery between the fixed charge and the
        # variable rates, and a large per-component swing is a design
        # decision, not a customer impact. The customer's impact is the
        # total — that is the number this table should argue from,
        # especially alongside the affordability figures below. The change
        # cells are pre-formatted strings so the component rows render as
        # genuinely blank cells, not a placeholder.
        total_delta = proposed.total - existing.total
        comparison["$ Change"] = ["", "", f"${total_delta:,.2f}"]
        comparison["% Change"] = ["", "",
                                  f"{total_delta / existing.total:+.1%}"
                                  if existing.total else ""]
        st.table(comparison.style.format({"Existing": "${:,.2f}",
                                          "Proposed": "${:,.2f}"}))

        # Affordability context for the selected tract: the illustrative bill
        # annualised against that tract's CPI-indexed median household income
        # — the same denominator the Affordability map uses, so the two tabs
        # can never disagree about what a percent means. Four columns, to sit
        # flush under the four-column Components table above.
        if tract_pick != DISTRICT:
            mhi = _tract_mhi(cfg, tract_pick)
            if mhi:
                st.subheader("Affordability Review")
                annual_bill = proposed.total * cfg.n_periods
                median_bill = _tract_median_bill(result, cfg, tract_pick,
                                                 tract_lookup)
                # Header titles carry explicit line breaks (rendered via
                # white-space: pre-line in the theme CSS) so each wraps at
                # its natural phrase boundary instead of wherever width
                # happens to cut it.
                row = {
                    "Tract": _tract_label(tract_pick),
                    "Proposed\nAnnual Bill": annual_bill,
                    "Household\nIncome": mhi,
                    # This customer's bill against the tract's income. Differs
                    # from the tract burden whenever the illustrative usage
                    # differs from the tract's median account.
                    "Bill as %\nof Income": annual_bill / mhi,
                    "Avg Burden\nof Tract": (median_bill / mhi
                                            if median_bill else None),
                }
                # Static table rather than the interactive grid: the grid
                # truncates long headers and cannot right-align string cells.
                # The tract name sits in the row index so it renders as a
                # left-aligned row label, mirroring the Components table's
                # Fixed/Variable/Total labels, while every data cell obeys
                # the app-wide right alignment.
                afford = (pd.DataFrame([row]).set_index("Tract")
                          .rename_axis(None))
                st.table(afford.style.format({
                    "Proposed\nAnnual Bill": "${:,.0f}",
                    "Household\nIncome": "${:,.0f}",
                    "Bill as %\nof Income": "{:.2%}",
                    "Avg Burden\nof Tract": "{:.2%}"}, na_rep="—"))

    st.subheader("Tier detail")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Existing**")
        st.dataframe(existing.tier_table(klass.existing.commodity_rates, cfg.units)
                     .style.format({f"Usage ({cfg.units})": "{:,.1f}",
                                    "Rate ($)": "${:,.2f}", "Charge ($)": "${:,.2f}"}),
                     width='stretch', hide_index=True)
    with t2:
        st.markdown("**Proposed**")
        st.dataframe(proposed.tier_table(klass.revised.commodity_rates, cfg.units)
                     .style.format({f"Usage ({cfg.units})": "{:,.1f}",
                                    "Rate ($)": "${:,.2f}", "Charge ($)": "${:,.2f}"}),
                     width='stretch', hide_index=True)

    annual = st.checkbox("Show annualised figures "
                         f"({cfg.n_periods} periods per year)", value=False)
    if annual:
        st.info(f"Annual existing **${existing.total * cfg.n_periods:,.2f}** · "
                f"proposed **${proposed.total * cfg.n_periods:,.2f}** · "
                f"difference **${delta * cfg.n_periods:,.2f}**")


def _impact_distribution(result: model.StudyResult, cfg: StudyConfig,
                         class_names: list[str], default_class: str) -> None:
    """How bill changes are spread across customers, per class or system-wide."""
    ALL = "All accounts (system total)"
    options = [ALL] + class_names
    scope = st.selectbox("Scope", options,
                         index=options.index(default_class)
                         if default_class in options else 0,
                         key="impact_scope")

    buckets = cfg.impact_buckets or {}
    percents = buckets.get("account_percent")

    if scope == ALL:
        pooled = result.combined_impacts()
        st.markdown(f"### {ALL}")
        st.caption(f"{pooled['accounts']:,} accounts across "
                   f"{len(result.classes)} customer classes.")
        existing_rev, revised_rev = pooled["existing_annual"], pooled["revised_annual"]
        pcts = pooled["pct_annual"]
    else:
        res = result.classes[scope]
        st.markdown(f"### {scope}")
        st.caption(f"{res.summary_existing.n_accounts:,} accounts.")
        if cfg.customer_classes[scope].revised.is_empty():
            st.warning("No revised rates entered for this class, so every bill change "
                       "is just the removal of the existing bill.")
        existing_rev = res.bills.existing.annual().sum()
        revised_rev = res.bills.revised.annual().sum()
        pcts = res.bills.pct_annual

    change = revised_rev - existing_rev
    c1, c2, c3 = st.columns(3)
    c1.metric("Existing annual revenue", f"${existing_rev:,.0f}")
    c2.metric("Proposed annual revenue", f"${revised_rev:,.0f}")
    c3.metric("Change", f"${change:,.0f}",
              delta=f"{change / existing_rev:+.1%}" if existing_rev else None)

    # Bill impacts by dollar amount are dropped from this client-facing build —
    # account impacts by percent change is the one distribution shown here.
    for title, values, edges, axis, fmt, kind in (
        ("Account impacts (annual %)", pcts, percents, "Change in annual bill (%)",
         model.PERCENT_LABEL, "percent"),
    ):
        if not edges:
            continue
        with st.expander(title, expanded=True):
            # The bucket edges are editable in place. The config supplies the
            # default; an edit lives for the session and never writes to disk,
            # the same contract as the rate-entry grids. Percent edges are
            # entered as percents (-20, -10, 0, 10) and stored as fractions.
            default = (", ".join(f"{e * 100:g}" for e in edges) if kind == "percent"
                       else ", ".join(f"{e:g}" for e in edges))
            raw_edges = st.text_input(
                "Ranges (comma-separated upper bounds, ascending)", value=default,
                key=f"impact_edges_{kind}_{scope}",
                help="Each value is a bucket's upper bound; a final open bucket "
                     "catches everything above the last. Percent ranges are "
                     "entered as percents.")
            try:
                parsed = [float(v) for v in raw_edges.replace("$", "")
                          .replace("%", "").split(",") if v.strip()]
                if parsed != sorted(parsed) or not parsed:
                    raise ValueError
                use_edges = ([v / 100 for v in parsed] if kind == "percent"
                             else parsed)
            except ValueError:
                st.warning("Ranges must be ascending numbers separated by "
                           "commas — using the configured defaults.")
                use_edges = edges
            table = billing.bucket_counts(values, use_edges, fmt)
            table_col, chart_col = st.columns([2, 3])
            with table_col:
                st.dataframe(table.style.format({"count": "{:,.0f}",
                                                 "share": "{:.1%}"}),
                             width='stretch', hide_index=True)
            with chart_col:
                st.altair_chart(theme.distribution_bars(table, axis),
                                width='stretch')


def _accounts(result: model.StudyResult, cfg: StudyConfig, name: str) -> None:
    if name not in result.classes:
        st.info(f"No accounts in {name}.")
        return
    detail = result.classes[name].account_detail(cfg.periods)
    # account_detail() always builds tier_1..5 columns (RateSchedule pads every
    # class to 5 tiers); drop the ones beyond this class's active tiers so the
    # table and its download don't carry columns that are always zero.
    n_active = len(summarize.active_tiers(result.classes[name].summary_existing))
    unused_cols = [f"{side}_tier_{i + 1}" for i in range(n_active, 5)
                   for side in ("existing", "revised")]
    detail = detail.drop(columns=[c for c in unused_cols if c in detail.columns])
    st.caption(f"{len(detail):,} accounts. Filter, sort, then download.")
    c1, c2 = st.columns(2)
    sizes = c1.multiselect("Meter size", sorted(detail["meter_sz"].unique()))
    min_usage = c2.number_input("Minimum usage", value=0.0, step=10.0)

    view = detail
    if sizes:
        view = view[view["meter_sz"].isin(sizes)]
    if min_usage:
        view = view[view["usage"] >= min_usage]

    st.dataframe(view.head(2000), width='stretch', height=420)
    st.download_button("Download filtered accounts (CSV)",
                       view.to_csv(index=False).encode("utf8"),
                       file_name=f"{name.lower()}_accounts_{result.year}.csv",
                       mime="text/csv")


def _export(result: model.StudyResult, cfg: StudyConfig, year: str) -> None:
    st.write("Write the full study — dashboard, summary, one sheet per class, "
             "peaking, and checks — to a formatted workbook.")
    if st.button("Build Excel workbook"):
        tmp = ROOT / "build" / f"_ui_{year}.xlsx"
        report.write(result, tmp, account_detail=False)
        buffer = io.BytesIO(tmp.read_bytes())
        st.download_button(
            "Download workbook", buffer.getvalue(),
            file_name=f"{cfg.agency.replace(' ', '_')}_{year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success(f"Built ({tmp.stat().st_size / 1e6:.2f} MB).")


if __name__ == "__main__":
    main()
