"""Streamlit view for the affordability analysis.

Kept out of ``app.py`` because it is the one tab with external inputs â€” a
crosswalk, an ACS vintage, a geometry file â€” and the one tab whose output is
*not* part of the rate study. Separating it keeps that boundary visible in the
file layout, not just in a footnote.
"""

from __future__ import annotations

import logging
import os

import altair as alt
import pandas as pd
import streamlit as st

from . import affordability as aff
from . import theme

log = logging.getLogger(__name__)

# A single-hue sequential ramp: light is affordable, deep is a heavier burden.
# Deliberately not red/green â€” the firm does not editorialise on an exhibit, and
# a diverging palette implies a good/bad verdict the analysis does not support.
RAMP = ["#EAF2F5", "#BBD7DF", "#8ABBC8", "#4E93A5", theme.TEAL, "#0A4A5A"]

GEO_LABELS = {
    "tract": "Census tract",
    "zcta": "ZIP code (ZCTA approximation)",
}

# Ordered by defensibility, best first â€” the selector offers them in this order
# so the sound choice is the one already selected.
GEO_ORDER = ["tract", "zcta"]

# One screening basis on screen: median household income â€” the measure the
# regulatory conventions (CA 1.5%, EPA 2.5%/2.0%) are defined on. The others
# are computed and exported but deliberately not offered as views:
# - Mean: exceeds the median almost everywhere, most where the low-income
#   population is largest â€” systematically the friendlier number. Kept as the
#   mean/median skew ratio in the table.
# - Lowest quintile (B19081): removed from the display per Beeb; still in the
#   CSV export for targeting work.
# - Renter median (B25119_003): the median for ALL renter-occupied households
#   in the tract, regardless of structure â€” ACS publishes no income table by
#   tenure AND structure type. On a Single-Family basis it mostly reflects
#   households in complexes, i.e. the wrong population. Its correct use is a
#   per-unit Multi-Family burden at an agency with dwelling-unit counts;
#   surface it only there.
BASES = {
    "Median household income": "burden_mhi",
}

# One clause per convention, keyed to the threshold names in
# affordability.THRESHOLDS â€” assembled into the "Screening thresholds"
# caption below for whichever ones a study's config actually carries. Keeping
# them per-threshold means a build narrowed to a single convention (a
# water-only district screening on EPA alone, say) never describes
# conventions it isn't showing.
_THRESHOLD_CLAUSES = {
    "ca_needs_assessment": "The California State Water Board's Drinking Water "
                           "Needs Assessment screens at 1.5% of MHI",
    "epa_water": "EPA's drinking-water criterion is 2.5%",
    "epa_wastewater": "EPA's wastewater residential indicator is 2.0%",
    "epa_combined": "The 4.5% combined figure applies only to a combined water "
                    "and wastewater utility â€” a water-only district should not "
                    "screen against it",
}


def _threshold_caption(cfg: aff.AffordabilityConfig) -> str:
    clauses = [_THRESHOLD_CLAUSES[name] for name in cfg.thresholds
              if name in _THRESHOLD_CLAUSES]
    body = ". ".join(clauses)
    return ("Each of these is a **convention**, not a legal standard."
            + (f" {body}." if body else ""))


def _settings(cfg, geography: str | None = None) -> aff.AffordabilityConfig:
    raw = dict(getattr(cfg, "affordability", {}) or {})
    known = {f for f in aff.AffordabilityConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if geography:
        kwargs["geography"] = geography
        per_geo = (raw.get("geographies") or {}).get(geography) or {}
        for k, v in per_geo.items():
            if k in known:
                kwargs[k] = v
    for key in ("basis_classes", "per_unit_classes"):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(kwargs[key])
    return aff.AffordabilityConfig(**kwargs)


def _resolver(cfg):
    base = cfg.source_path.parent if cfg.source_path else None

    def resolve(value):
        if not value:
            return None
        if os.path.isabs(value) or base is None:
            return value
        candidate = os.path.join(base, value)
        if os.path.exists(candidate):
            return candidate
        return value if os.path.exists(value) else candidate
    return resolve


def _paths(cfg, geography: str) -> dict:
    """Input paths for one geography.

    A ``geographies:`` block in the config holds one entry per geography; a flat
    ``crosswalk``/``acs``/``geometry`` at the top level is the older single-
    geography form and still works.
    """
    raw = dict(getattr(cfg, "affordability", {}) or {})
    per_geo = (raw.get("geographies") or {}).get(geography)
    source = per_geo if per_geo is not None else (
        raw if raw.get("geography", "zcta") == geography else {})
    resolve = _resolver(cfg)
    return {k: resolve(source.get(k)) for k in ("crosswalk", "acs", "geometry")}


def _available(cfg) -> list[str]:
    """Geographies whose crosswalk and ACS cache both exist on disk.

    Configured-but-absent geographies are not offered rather than offered and
    broken: before the addresses are geocoded there is no tract crosswalk, and
    the tab should say so once rather than fail on selection.
    """
    raw = dict(getattr(cfg, "affordability", {}) or {})
    names = list(raw.get("geographies") or {}) or [raw.get("geography", "zcta")]
    ordered = ([g for g in GEO_ORDER if g in names]
               + [g for g in names if g not in GEO_ORDER])
    out = []
    for g in ordered:
        p = _paths(cfg, g)
        if (p.get("crosswalk") and p.get("acs")
                and os.path.exists(p["crosswalk"]) and os.path.exists(p["acs"])):
            out.append(g)
    return out


def render(result, cfg) -> None:
    raw = dict(getattr(cfg, "affordability", {}) or {})
    available = _available(cfg)
    configured = list(raw.get("geographies") or {}) or [raw.get("geography", "zcta")]

    if not available:
        st.info(
            "Affordability inputs are not configured yet. Add an "
            "`affordability:` block to the agency config with, per geography:\n\n"
            "- `crosswalk` â€” CSV of account key to ZIP or tract GEOID\n"
            "- `acs` â€” the ACS pull from `tools/fetch_census.py`\n"
            "- `geometry` â€” Census cartographic-boundary GeoJSON (optional; "
            "without it the tables render and the map does not)")
        return

    default = raw.get("geography") if raw.get("geography") in available else available[0]
    if len(available) > 1:
        geography = st.radio(
            "Geography", available, horizontal=True, key="aff_geography",
            index=available.index(default),
            format_func=lambda g: GEO_LABELS.get(g, g),
            help="Tract is the analytic unit. ZIP is a communication and "
                 "cross-check view: ZIP Codes are USPS delivery routes rather "
                 "than areas, and ZCTA boundaries ignore the service-area "
                 "boundary, so the income attached includes non-customers.")
    else:
        geography = default
        absent = [g for g in configured if g not in available]
        if absent:
            # Not user-facing: the reviewer doesn't need build instructions on
            # screen. It lands in the server console/log instead.
            log.info("affordability: showing %s; %s configured but inputs not "
                     "on disk (geocode_addresses.py, then fetch_census.py "
                     "--tracts)", geography, ", ".join(absent))

    if geography == "zcta":
        st.caption("ZIP basis â€” income joins are ZCTA-wide and extend past the "
                   "service-area boundary.")

    settings = _settings(cfg, geography)
    paths = _paths(cfg, geography)

    # Which customers the burden is measured on. Configured per agency:
    # Helix carries a single Single-Family option (no dwelling-unit counts on
    # master-metered complexes, so a Multifamily bill cannot be split per
    # household); an agency with real unit counts lists Multi-Family and a
    # combined option and gets a selector.
    class_options = {str(k): list(v)
                     for k, v in (raw.get("class_options") or {}).items() if v}
    if len(class_options) > 1:
        label = st.radio("Customer basis", list(class_options),
                         horizontal=True, key="aff_classes")
        settings.basis_classes = tuple(class_options[label])
    elif class_options:
        label = next(iter(class_options))
        settings.basis_classes = tuple(class_options[label])
        st.caption(f"{label} accounts")

    crosswalk = aff.load_crosswalk(paths["crosswalk"], settings.crosswalk_key)
    acs = aff.load_acs(paths["acs"])

    bills = aff.per_unit(aff.account_bills(result), settings)
    joined, join_report = aff.attach_geography(bills, crosswalk, settings)

    # The join report leads. Every number below it is computed on the matched
    # subset, so a poor match rate is not a footnote â€” it is the headline.
    rate = join_report["match_rate"]
    line = (f"{join_report['matched']:,} of {join_report['accounts']:,} accounts "
            f"matched ({rate:.1%}), across {join_report['geographies']} "
            f"{'tracts' if geography == 'tract' else 'ZIPs'}.")
    (st.warning if rate < 0.95 else st.success)(
        line + ("  Below 95% â€” check the crosswalk key and vintage before "
                "reading anything into these averages." if rate < 0.95 else ""))

    agg = aff.representative_by_geography(joined, settings)
    table = aff.burden(agg, acs, settings)

    st.subheader("Screening thresholds")
    st.caption(_threshold_caption(settings))
    st.dataframe(
        aff.summary(table, settings).style.format(
            {"% of income": "{:.1%}", "Accounts affected": "{:,.0f}"}),
        width="stretch", hide_index=True)

    st.subheader("Burden by geography")
    choices = [k for k, v in BASES.items() if v in table.columns]
    if len(choices) > 1:
        basis = st.radio("Income basis", choices, horizontal=True,
                         key="aff_basis")
    else:
        basis = choices[0]
    column = BASES[basis]

    live = table[~table["suppressed"] & table[column].notna()]
    if live.empty:
        st.warning("No geography has both an income estimate and enough accounts "
                   f"to report (minimum {settings.min_accounts}).")
        return

    threshold = settings.thresholds[settings.primary_threshold]
    geometry = paths.get("geometry")
    resolve = _resolver(cfg)
    boundary_path = resolve(raw.get("boundary"))
    clicked = None
    if geometry and os.path.exists(geometry):
        boundary = (aff.load_geometry(boundary_path)
                    if boundary_path and os.path.exists(boundary_path) else None)
        spec = _map(aff.load_geometry(geometry), live, column, cfg,
                    threshold=threshold, boundary=boundary)
        try:
            event = st.vega_lite_chart(spec, width='stretch',
                                       key=f"aff_map_{geography}",
                                       on_select="rerun")
            clicked = _clicked_geoid(event)
        except TypeError:
            # Older Streamlit without on_select. Say so rather than letting
            # the click die silently â€” the fix is a one-line update.
            st.vega_lite_chart(spec, width='stretch')
            st.caption(f"Map click needs Streamlit 1.35+ (installed: "
                       f"{st.__version__}). Update with "
                       f"`py -m pip install -U streamlit`; the menu below "
                       f"works meanwhile.")
    else:
        log.info("affordability: no geometry file for %s â€” ranked table only",
                 geography)

    _accounts_panel(joined, live, crosswalk_path=paths["crosswalk"],
                    settings=settings, geography=geography, clicked=clicked,
                    column=column)

    st.altair_chart(_ranked_bars(live, column, threshold), width='stretch')

    show = live.sort_values(column, ascending=False).copy()
    show["label"] = show["geoid"].map(_tract_label)
    cols = {"label": "Tract" if geography == "tract" else "ZIP",
            "accounts": "Accounts",
            "bill_existing": "Existing Annual Bill",
            "bill_revised": "Proposed Annual Bill",
            "bill_change_pct": "Change", "mhi": "Median income",
            "mean_income": "Mean income", "income_skew": "Skew",
            column: "Burden"}
    present = {k: v for k, v in cols.items() if k in show.columns}
    st.dataframe(
        show[list(present)].rename(columns=present).style.format({
            "Existing Annual Bill": "${:,.0f}",
            "Proposed Annual Bill": "${:,.0f}",
            "Change": "{:+.1%}", "Median income": "${:,.0f}",
            "Mean income": "${:,.0f}", "Skew": "{:.2f}",
            "Burden": "{:.2%}", "Accounts": "{:,.0f}"}),
        width="stretch", hide_index=True)

    if "income_skew" in live.columns and live["income_skew"].notna().any():
        skew = live["income_skew"]
        worst = live.loc[skew.idxmax()]
        st.caption(
            f"**Skew** is mean Ã· median household income â€” a tail indicator, "
            f"not a burden. Range here {skew.min():.2f}â€“{skew.max():.2f}. A high "
            f"value means a long upper income tail, so the median understates "
            f"how hard the bill lands on that geography's lower-income "
            f"households. Highest here: {worst['geoid']} at "
            f"{worst['income_skew']:.2f}.")

    if "uncertain" in table.columns:
        n = int(table["uncertain"].fillna(False).sum())
        if n:
            st.caption(
                f"{n} geographies have a 90% confidence band on median income "
                f"that straddles the {threshold:.1%} threshold. They are not "
                f"shown to exceed it â€” the ACS estimate is not precise enough "
                f"at this geography to say either way.")

    suppressed = int(table["suppressed"].sum())
    if suppressed:
        st.caption(f"{suppressed} geographies suppressed for having fewer than "
                   f"{settings.min_accounts} accounts.")

    st.download_button(
        "Download burden table (CSV)", table.to_csv(index=False).encode(),
        file_name=f"{cfg.agency.replace(' ', '_')}_burden_{geography}.csv",
        mime="text/csv")


def _clicked_geoid(event) -> str | None:
    """GEOID of the polygon clicked on the map, if any.

    The event shape has varied across Streamlit releases (attribute vs mapping
    access, list of records vs dict of lists), so every plausible shape is
    tried before giving up â€” a shape mismatch must degrade to "no click", not
    to a swallowed exception that leaves the click silently dead.
    """
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if sel is None:
        return None
    points = (getattr(sel, "geo_click", None)
              or (sel.get("geo_click") if hasattr(sel, "get") else None))
    if not points:
        return None
    # Two shapes exist in the wild: a list of records [{field: value}], and a
    # dict of lists {field: [value, ...]}. Normalise to one record either way,
    # and unwrap single-element lists â€” the earlier parser filtered list
    # values out entirely, which silently discarded the dict-of-lists shape.
    first = points[0] if isinstance(points, (list, tuple)) else points
    try:
        record = dict(first)
    except (TypeError, ValueError):
        return None
    for value in record.values():
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, (dict, list, tuple)) or value in (None, ""):
            continue
        return str(value)
    return None


def _tract_label(geoid: str) -> str:
    """Human name for a geography: 'Tract 154.03' from an 11-digit GEOID,
    the raw value otherwise (ZIPs are already readable)."""
    g = str(geoid)
    if len(g) == 11 and g.isdigit():
        code = int(g[5:])
        return f"Tract {code // 100}.{code % 100:02d}" if code % 100 \
            else f"Tract {code // 100}"
    return g


def _operator_label(operator: str) -> str:
     """Human-readable label for burden filter operator."""
     return "greater than" if operator == "gt" else "less than"


def _filter_by_burden(joined: pd.DataFrame, filter_type: str, operator: str, 
                      threshold: float, column: str) -> pd.DataFrame:
     """Filter accounts by burden % criteria.
     
     Args:
         joined: DataFrame with account details and burden columns
         filter_type: "geography" (filter by geography's burden) or "individual" (by account's burden)
         operator: "gt" (greater than) or "lt" (less than)
         threshold: burden % threshold (as decimal, e.g., 0.025 for 2.5%)
         column: burden column name (e.g., "burden_mhi")
     
     Returns:
         Filtered DataFrame
     """
     if filter_type == "geography":
         # Filter by geography's burden threshold
         frame = joined.copy()
         # Create a mapping of geoid to its burden value
         geo_burden = frame.groupby("geoid")[column].first()
         if operator == "gt":
             valid_geos = geo_burden[geo_burden > threshold].index
         else:  # "lt"
             valid_geos = geo_burden[geo_burden < threshold].index
         frame = frame[frame["geoid"].isin(valid_geos)]
     else:  # "individual"
         # Filter by individual account's burden
         frame = joined.copy()
         # Each account has their own burden % (bill/income)
         # Use the tract/geography's median household income for consistency
         pct_income = frame["bill_revised"] / frame["mhi"]
         if operator == "gt":
             frame = frame[pct_income > threshold]
         else:  # "lt"
             frame = frame[pct_income < threshold]
     
     return frame



def _accounts_panel(joined: pd.DataFrame, live: pd.DataFrame, crosswalk_path: str,
                    settings, geography: str, clicked: str | None, 
                    column: str) -> None:
     """Dual-mode account filter: by tract or by burden %."""
     if "aff_filter_mode" not in st.session_state:
         st.session_state["aff_filter_mode"] = "tract"
     
     st.subheader("Account Filter")
     c1, c2 = st.columns([1, 2])
     with c1:
         fm = st.radio("Filter by", ["tract", "burden"],
                      format_func=lambda x: "Tract" if x == "tract" else "Burden %",
                      horizontal=True, key="aff_filter_mode")
     
     # BURDEN FILTER MODE
     if fm == "burden":
         with c2:
             st.markdown("")
         b1, b2, b3 = st.columns(3)
         with b1:
             bt = st.radio("Type", ["geography", "individual"],
                          format_func=lambda x: "Geography" if x == "geography" else "Account",
                          horizontal=True, key="aff_burden_type")
         with b2:
             op = st.radio("Operator", ["gt", "lt"],
                          format_func=lambda x: ">" if x == "gt" else "<",
                          horizontal=True, key="aff_burden_operator")
         with b3:
             tp = st.number_input("Threshold (%)", min_value=0.0, max_value=100.0,
                                 value=2.5, step=0.1, key="aff_burden_threshold")
         
         frame = _filter_by_burden(joined, bt, op, tp/100, column)
         frame = frame[frame["annual_usage"] > 0].copy()
         if settings.basis_classes:
             frame = frame[frame["cust_class"].isin(settings.basis_classes)]
         
         st.info(f"**{len(frame):,}** accounts with burden {_operator_label(op)} {tp:.1f}%")
         
         try:
             extra = pd.read_csv(crosswalk_path, dtype=str)
             extra.columns = [c.strip().lower() for c in extra.columns]
             k = settings.crosswalk_key.lower()
             if k in extra.columns and "city" in extra.columns:
                 extra = extra.drop_duplicates(subset=[k])
                 frame[settings.crosswalk_key] = frame[settings.crosswalk_key].astype(str)
                 frame = frame.merge(extra[[k, "city"]], how="left",
                                    left_on=settings.crosswalk_key, right_on=k)
         except Exception:
             pass
         
         cols = {"location_no": "Account", "city": "City",
                 "meter_sz": "Meter", "annual_usage": "Annual usage",
                 "bill_existing": "Existing Annual Bill",
                 "bill_revised": "Proposed Annual Bill",
                 "bill_change_pct": "Change",
                 "pct_income": "Burden %"}
         
         frame["bill_change_pct"] = (frame["bill_revised"] - frame["bill_existing"]) \
             / frame["bill_existing"].where(frame["bill_existing"] != 0)
         frame["pct_income"] = frame["bill_revised"] / frame["mhi"]
         
         present = {k: v for k, v in cols.items() if k in frame.columns}
         st.dataframe(
             frame.sort_values("bill_revised", ascending=False)[list(present)]
             .rename(columns=present).style.format({
                 "Annual usage": "{:,.0f}", "Existing Annual Bill": "${:,.0f}",
                 "Proposed Annual Bill": "${:,.0f}", "Change": "{:+.1%}",
                 "Burden %": "{:.2%}"}),
             width="stretch", hide_index=True, height=420)
     
     # TRACT FILTER MODE
     else:
         options = live.sort_values("burden_mhi", ascending=False)["geoid"].tolist()
         if not options:
             st.warning("No geographies available.")
             return
         
         st.markdown("")
         # The selection event persists across reruns, so the click is applied only
         # when it differs from the last one seen. Otherwise the map's most recent
         # click would override the pulldown on every rerun and the pulldown would
         # go dead after the first click.
         if clicked in options and clicked != st.session_state.get("aff_last_click"):
             st.session_state["aff_focus"] = clicked
             st.session_state["aff_last_click"] = clicked
         if st.session_state.get("aff_focus") not in options:
             st.session_state.pop("aff_focus", None)
         focus = st.selectbox(
             "Click a tract on the map, or pick one here", options,
             key="aff_focus", format_func=_tract_label)

         frame = joined[joined["geoid"] == focus]
         if settings.basis_classes:
             frame = frame[frame["cust_class"].isin(settings.basis_classes)]
         frame = frame[frame["annual_usage"] > 0].copy()

         # City rides along from the crosswalk file when it carries one. Street
         # addresses are deliberately NOT shown — account number plus city is
         # enough to work a list without putting a household's address on screen.
         try:
             extra = pd.read_csv(crosswalk_path, dtype=str)
             extra.columns = [c.strip().lower() for c in extra.columns]
             key = settings.crosswalk_key.lower()
             carry = [c for c in ("city",) if c in extra.columns]
             if carry and key in extra.columns:
                 extra = extra.drop_duplicates(subset=[key])
                 frame[settings.crosswalk_key] = frame[settings.crosswalk_key].astype(str)
                 frame = frame.merge(extra[[key] + carry], how="left",
                                     left_on=settings.crosswalk_key, right_on=key)
         except Exception:                                  # noqa: BLE001
             carry = []

         row = live.loc[live["geoid"] == focus].iloc[0]
         bits = [f"{len(frame):,} accounts", f"median bill ${row['bill_revised']:,.0f}"]
         if pd.notna(row.get("burden_mhi")):
             bits.append(f"burden {row['burden_mhi']:.2%}")
         if pd.notna(row.get("mhi")):
             bits.append(f"median income ${row['mhi']:,.0f}")
         st.caption(f"**{_tract_label(focus)}** — " + " · ".join(bits))

         cols = {"location_no": "Account", "city": "City",
                 "meter_sz": "Meter", "annual_usage": "Annual usage",
                 "bill_existing": "Existing Annual Bill",
                 "bill_revised": "Proposed Annual Bill",
                 "bill_change_pct": "Change",
                 "pct_income": "Proposed Bill as % of Income"}
         frame["bill_change_pct"] = (frame["bill_revised"] - frame["bill_existing"]) \
             / frame["bill_existing"].where(frame["bill_existing"] != 0)
         # Each account's own bill against the geography's median household income
         # (CPI-indexed) — the same denominator the map uses, so the account list
         # and the map agree about what a percent means.
         mhi = row.get("mhi")
         frame["pct_income"] = (frame["bill_revised"] / mhi
                                if pd.notna(mhi) and mhi else pd.NA)
         present = {k: v for k, v in cols.items() if k in frame.columns}
         st.dataframe(
             frame.sort_values("bill_revised", ascending=False)[list(present)]
             .rename(columns=present).style.format({
                 "Annual usage": "{:,.0f}", "Existing Annual Bill": "${:,.0f}",
                 "Proposed Annual Bill": "${:,.0f}", "Change": "{:+.1%}",
                 "Proposed Bill as % of Income": "{:.2%}"}),
             width="stretch", hide_index=True, height=420)
def _map(geojson: dict, table: pd.DataFrame, column: str, cfg,
         threshold: float = 0.015, boundary: dict | None = None) -> dict:
    """Choropleth of the service area, colored on threshold-anchored bins.

    Bins are fixed around the primary affordability screen (default the
    California 1.5%-of-MHI convention) rather than quantiles, so color answers
    "which side of the threshold, and by how much" â€” the question the exhibit
    exists for â€” instead of "how does this polygon rank against its neighbours".
    The two darkest steps begin AT the threshold, so crossing it is a visible
    break in the ramp.

    The Census polygons are pre-clipped to the district boundary by
    ``tools/clip_to_boundary.py``, and the boundary itself is drawn on top, so
    the map reads as the district's service area. Income joined to a clipped
    polygon is still the estimate for the whole Census geography.

    The GEOID property name differs by vintage and geography â€” ZCTA files carry
    ``ZCTA5CE20`` or ``GEOID20``, tract files ``GEOID``. Rather than hard-code
    one, the first recognised property on the first feature is used.
    """
    props = geojson["features"][0]["properties"]
    key = next((k for k in ("GEOID", "GEOID20", "GEOID10", "ZCTA5CE20",
                            "ZCTA5CE10") if k in props), list(props)[0])
    # Tract geometry from TIGERweb carries a readable NAME ("Census Tract
    # 27.03"); the stripped ZCTA file does not, so the code itself is shown.
    name_key = "NAME" if "NAME" in props else key

    # Bin edges anchored on the threshold: three steps below it, the threshold
    # itself, then 4/3 and 5/3 of it above. At the 1.5% default this yields
    # 1.0 / 1.25 / 1.5 / 2.0 / 2.5.
    edges = [round(threshold * f, 6)
             for f in (2 / 3, 5 / 6, 1.0, 4 / 3, 5 / 3)]

    data = alt.Data(values=geojson,
                    format=alt.DataFormat(property="features", type="json"))
    fields = ["geoid", column, "bill_revised", "accounts", "mhi"]
    fields = [f for f in fields if f in table.columns or f == "geoid"]
    lookup = alt.LookupData(data=table[fields], key="geoid", fields=fields[1:])

    # The GEOID lives nested under datum.properties, and Vega-Lite point
    # selections cannot reliably extract nested fields into the selection
    # tuple â€” the click event arrives as an empty record ({}). Flattening the
    # id onto the datum with a calculate transform and selecting on the flat
    # field is the standard fix; the click then carries the GEOID.
    click = alt.selection_point(name="geo_click", on="click",
                                fields=["geoid_sel"])
    fills = (
        alt.Chart(data)
        .mark_geoshape(stroke="white", strokeWidth=1.1)
        .transform_calculate(geoid_sel=f"datum.properties['{key}']")
        .add_params(click)
        .transform_lookup(lookup=f"properties.{key}", from_=lookup)
        .encode(
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.55)),
            # A polygon with no burden value â€” suppressed for thinness, or with
            # no income estimate â€” draws in neutral grey rather than taking the
            # bottom of the ramp, which would read as "affordable".
            color=alt.condition(
                f"isValid(datum['{column}'])",
                alt.Color(f"{column}:Q",
                          title=["Bill Ã· income", f"(screen {threshold:.2%})"],
                          scale=alt.Scale(type="threshold", domain=edges,
                                          range=RAMP),
                          legend=alt.Legend(format=".2%", orient="right")),
                alt.value(theme.NEUTRAL)),
            tooltip=[
                alt.Tooltip(f"properties.{name_key}:N", title="Geography"),
                alt.Tooltip(f"{column}:Q", format=".2%", title="Burden"),
                alt.Tooltip("bill_revised:Q", format="$,.0f", title="Proposed bill"),
                alt.Tooltip("accounts:Q", format=",.0f", title="Accounts"),
                alt.Tooltip("properties.in_district_share:Q", format=".0%",
                            title="Share inside district")],
        )
    )

    # Layer order is load-bearing: the boundary goes UNDER the fills. Drawn on
    # top, its (unfilled) geoshape still hit-tests its interior in this Vega
    # version and steals every click, so the selection fires with an empty
    # datum. Underneath, the fills receive the click and the outline still
    # reads at the district edge.
    layers = []
    if boundary:
        bdata = alt.Data(values=boundary,
                         format=alt.DataFormat(property="features", type="json"))
        layers.append(
            alt.Chart(bdata)
            .mark_geoshape(filled=False, stroke=theme.SLATE, strokeWidth=1.8))
    layers.append(fills)

    chart = (
        alt.layer(*layers)
        .properties(height=560,
                    title=f"{cfg.agency} â€” proposed annual bill as a percent of "
                          f"household income")
        .project(type="mercator")
        .configure_view(stroke=None)
    )

    # Altair hoists selection params to the top level of a layered spec, and
    # this Vega-Lite version then compiles the selection into EVERY layer â€”
    # "Duplicate signal name" â€” leaving the selection dead. Vega-Lite requires
    # selections inside a unit spec, so move the param back into the fills
    # layer (the one carrying the calculate transform). Verified against the
    # live renderer: with the param in-unit and the boundary underneath, a
    # click returns the tract GEOID.
    spec = chart.to_dict()
    params = spec.pop("params", None)
    if params:
        for layer in spec.get("layer", []):
            transforms = layer.get("transform", [])
            if any("calculate" in t for t in transforms):
                layer["params"] = params
                break
        else:
            spec["params"] = params   # single-layer fallback: put it back
    return spec


def _ranked_bars(table: pd.DataFrame, column: str, threshold: float) -> alt.Chart:
    """Ranked bars with the threshold drawn on.

    Kept alongside the map because it reads faster at small counts and is the
    honest presentation when there are only a handful of geographies â€” nine
    polygons is a bar chart wearing a map. At tract level the map earns its
    place and this becomes the sortable companion, so the labels drop out and
    the bars thin once the count gets large.
    """
    frame = table.sort_values(column, ascending=False).copy()
    frame["label"] = frame["geoid"].map(_tract_label)
    many = len(frame) > 30
    bars = (
        alt.Chart(frame, height=max(220, (11 if many else 26) * len(frame)))
        # A white outline separates adjacent bars â€” at tract counts the bars
        # sit nearly edge to edge and same-bin neighbours otherwise fuse into
        # one block.
        .mark_bar(size=9 if many else 18, color=theme.TEAL,
                  stroke="white", strokeWidth=1)
        .encode(
            y=alt.Y("label:N", sort="-x", title=None,
                    axis=alt.Axis(labels=not many)),
            x=alt.X(f"{column}:Q", title="Bill Ã· household income",
                    axis=alt.Axis(format=".1%")),
            tooltip=[alt.Tooltip("label:N", title="Geography"),
                     alt.Tooltip(f"{column}:Q", format=".2%", title="Burden"),
                     alt.Tooltip("accounts:Q", format=",.0f", title="Accounts")])
    )
    rule = (alt.Chart(pd.DataFrame({"t": [threshold]}))
            .mark_rule(color=theme.SLATE, strokeDash=[5, 3])
            .encode(x="t:Q"))
    return (bars + rule).configure_view(stroke=None)

