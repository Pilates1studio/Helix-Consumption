"""IMC palette and chart styling.

Colours and their roles follow the firm standard in
`imc-vault/firm/excel-branding.md`, so screen output reads as the same family as
the Excel models and report tables.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

EMERALD = "#01824E"      # primary, dark headers, positive results
TEAL = "#0F6C81"         # inputs, secondary accents
SLATE = "#3E454E"        # dark bars, section headers
SAGE = "#E6EDE6"
EMERALD_XLIGHT = "#E8F1EC"
EMERALD_LIGHT = "#CFE7DB"
EMERALD_MID = "#7FB79E"
TEAL_XLIGHT = "#EAF2F5"
TEAL_LIGHT = "#DCEAEF"
NEUTRAL = "#E9ECEE"
MUTED = "#8A9499"
OFF_WHITE = "#F6F7F8"
NEAR_BLACK = "#231F20"

# Tier ramp: emerald deepening as usage climbs the tiers.
TIER_COLORS = [EMERALD_MID, EMERALD, "#015C37", TEAL, SLATE]

# Bill components. Fixed is the muted slate segment, variable the emerald one.
FIXED_COLOR = SLATE
VARIABLE_COLOR = EMERALD

# Emerald at decreasing opacity, one step per tier. The variable block still
# reads as a single emerald mass; the banding shows how it splits across tiers
# without breaking the reported amount out tier by tier.
EMERALD_RGB = "1,130,78"
VARIABLE_TIER_ALPHA = [1.0, 0.74, 0.52, 0.36, 0.24]
VARIABLE_TIER_COLORS = [f"rgba({EMERALD_RGB},{a})" for a in VARIABLE_TIER_ALPHA]

CSS = f"""
<style>
  .stApp {{ background-color: {OFF_WHITE}; }}
  h1, h2, h3 {{ color: {NEAR_BLACK}; }}
  h1 {{ border-bottom: 3px solid {EMERALD}; padding-bottom: .3rem; }}
  h3 {{ color: {SLATE}; }}
  section[data-testid="stSidebar"] {{
      background-color: {SAGE};
      border-right: 1px solid {EMERALD_LIGHT};
  }}
  section[data-testid="stSidebar"] h2 {{
      color: {EMERALD}; font-size: 1.02rem; text-transform: uppercase;
      letter-spacing: .06em; margin-bottom: .2rem;
  }}
  /* Inputs carry the teal cue used for inputs in the Excel models. */
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] .stNumberInput input {{
      color: {TEAL} !important; font-weight: 600;
  }}
  div[data-testid="stMetric"] {{
      background-color: white; border: 1px solid {EMERALD_LIGHT};
      border-left: 4px solid {EMERALD}; border-radius: 3px; padding: .7rem .9rem;
  }}
  div[data-testid="stMetricValue"] {{ color: {EMERALD}; font-size: 1.6rem; }}
  div[data-testid="stMetricLabel"] {{ color: {SLATE}; }}
  button[data-baseweb="tab"] {{ color: {SLATE}; }}
  button[data-baseweb="tab"][aria-selected="true"] {{
      color: {EMERALD}; border-bottom-color: {EMERALD} !important;
  }}
  .stButton button, .stDownloadButton button {{
      background-color: {EMERALD}; color: white; border: none; font-weight: 600;
  }}
  .stButton button:hover, .stDownloadButton button:hover {{
      background-color: {TEAL}; color: white;
  }}
  /* Static tables (st.table): match the interactive tables' look. Styler
     CSS loses fights with Streamlit's own table styles, so this is set
     app-wide with priority — centered wrapped headers on the grey the
     dataframes use, white body cells, numbers right-aligned. Row-label
     cells (tbody th) left-align, which is why single-row display tables
     put their label column in the index. */
  div[data-testid="stTable"] thead th {{
      background-color: {NEUTRAL} !important; color: {SLATE} !important;
      text-align: center !important; white-space: pre-line !important;
      font-weight: 600; padding: .35rem .6rem !important;
      border: 1px solid {EMERALD_XLIGHT} !important;
  }}
  /* Streamlit nests header text inside div > p, each carrying its own
     white-space: normal, which collapses explicit newlines in column
     titles. Style EVERY descendant so titles break where they say, not
     where the width falls — verified against the live DOM (the p, not the
     div, holds the text). Box properties stay off the inner elements so
     borders don't double. */
  div[data-testid="stTable"] thead th * {{
      white-space: pre-line !important; text-align: center !important;
      background: transparent !important; border: none !important;
      margin: 0 !important; color: inherit !important;
      font-weight: inherit !important;
  }}
  div[data-testid="stTable"] tbody th {{
      background-color: white !important; color: {SLATE} !important;
      text-align: left !important; font-weight: 600;
      padding: .3rem .6rem !important;
      border: 1px solid {EMERALD_XLIGHT} !important;
  }}
  div[data-testid="stTable"] tbody td {{
      background-color: white !important; text-align: right !important;
      padding: .3rem .6rem !important;
      border: 1px solid {EMERALD_XLIGHT} !important;
  }}
</style>
"""


def inject(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def deviation_shading(value: float, midpoint: float = 1.0, span: float = 0.35) -> str:
    """Cell background shading a value by how far it sits from a midpoint.

    Emerald above the midpoint, slate below, deepening with distance. Written by
    hand rather than via `Styler.background_gradient`, which pulls in matplotlib
    for what is a two-colour ramp - and fails at render time if it is absent.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not span:
        return ""
    strength = min(abs(value - midpoint) / span, 1.0) * 0.55
    if strength < 0.02:
        return ""
    rgb = EMERALD_RGB if value >= midpoint else "62,69,78"
    return f"background-color: rgba({rgb},{strength:.3f})"


def _base(chart: alt.Chart, height: int) -> alt.Chart:
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=SLATE, titleColor=SLATE, grid=False,
                            domainColor=NEUTRAL, tickColor=NEUTRAL)
            .configure_legend(labelColor=SLATE, titleColor=SLATE))


def total_bars(frame: pd.DataFrame, category: str, units: str,
               height: int | None = None) -> alt.Chart:
    """Horizontal bars of a total by category, each labelled with value and share.

    `frame` is indexed by category with a "Total" column; any "Total" row is dropped
    so the bars show only the parts, and the share is each part of their sum.
    """
    data = (frame.loc[[i for i in frame.index if str(i) != "Total"], ["Total"]]
            .reset_index(names=category)
            .rename(columns={"Total": "Usage"}))
    total = data["Usage"].sum()
    data["Share"] = data["Usage"] / total if total else 0.0
    # A class rounding to "0.0%" reads as nothing at all; show it as below the
    # threshold instead so a small class is still visibly present.
    data["Label"] = [f"{v:,.0f}   " + ("<0.1%" if 0 < s < 0.001 else f"{s:.1%}")
                     for v, s in zip(data["Usage"], data["Share"])]

    ceiling = float(data["Usage"].max()) if len(data) else 0.0
    height = height or max(140, 42 * len(data) + 40)

    bars = alt.Chart(data).mark_bar(size=26).encode(
        y=alt.Y(f"{category}:N", title=None, sort="-x",
                axis=alt.Axis(labelFontSize=12, labelColor=SLATE)),
        x=alt.X("Usage:Q", title=units,
                scale=alt.Scale(domain=[0, ceiling * 1.28 if ceiling else 1])),
        color=alt.Color(f"{category}:N", scale=alt.Scale(range=TIER_COLORS),
                        legend=None),
        tooltip=[category, alt.Tooltip("Usage:Q", format=",.0f"),
                 alt.Tooltip("Share:Q", format=".1%")],
    )
    labels = alt.Chart(data).mark_text(
        align="left", dx=7, color=SLATE, fontWeight="bold", fontSize=12,
    ).encode(
        y=alt.Y(f"{category}:N", sort="-x"),
        x=alt.X("Usage:Q"),
        text="Label:N",
    )
    return _base(bars + labels, height)


def bill_comparison(frame: pd.DataFrame, height: int = 360,
                    width: int | None = None) -> alt.Chart:
    """Stacked bars splitting each scenario's bill into fixed and variable.

    `frame` has columns Scenario, Segment, Kind and Amount, where Kind is
    "Fixed charge" or "Variable charge" and Segment names the drawn band — the
    fixed charge, or one tier of the variable charge.

    Variable tiers are drawn as bands of the same emerald at decreasing opacity,
    so the split across tiers is visible while the variable charge still reads as
    one block. Labels are placed per Kind, not per band: the variable band carries
    a single total rather than a figure per tier.

    Segment extents are computed here rather than left to Vega's stacking, so the
    in-bar labels are guaranteed to sit on the block they describe.
    """
    kind_order = {"Fixed charge": 0, "Variable charge": 1}
    data = frame.copy()
    data["_kind"] = data["Kind"].map(kind_order)
    data = data.sort_values(["Scenario", "_kind", "Segment"], kind="stable")
    data["y1"] = data.groupby("Scenario")["Amount"].cumsum()
    data["y0"] = data["y1"] - data["Amount"]

    # One label per Kind, carrying that Kind's total. The variable label sits on
    # the boundary between the first two tier bands rather than the middle of the
    # block, so it cannot be misread as belonging to Tier 1 alone.
    rows = []
    for (scenario, kind), group in data.groupby(["Scenario", "Kind"], sort=False):
        low, high = float(group["y0"].min()), float(group["y1"].max())
        bands = group.sort_values("y0")
        on_boundary = kind == "Variable charge" and len(bands) > 1
        rows.append({
            "Scenario": scenario, "Kind": kind,
            "y": float(bands.iloc[0]["y1"]) if on_boundary else (low + high) / 2,
            "y0": low, "y1": high, "on_boundary": on_boundary,
            "Label": f"${group['Amount'].sum():,.2f}",
        })
    blocks = pd.DataFrame(rows)
    # A block too thin to hold text would otherwise show an overlapping label.
    span = float(data["y1"].max()) if len(data) else 0.0
    if span:
        blocks = blocks[(blocks["y1"] - blocks["y0"]) > span * 0.06]

    totals = (data.groupby("Scenario", as_index=False)["Amount"].sum()
              .rename(columns={"Amount": "Total"}))
    totals["Label"] = [f"${v:,.2f}" for v in totals["Total"]]
    ceiling = float(totals["Total"].max()) if len(totals) else 0.0

    # Fixed first, then tiers in order, so the opacity ramp reads bottom-up.
    segments = ["Fixed charge"] + sorted(
        s for s in data["Segment"].unique() if s != "Fixed charge")
    colors = [FIXED_COLOR] + VARIABLE_TIER_COLORS[:len(segments) - 1]

    # labelAngle=0 keeps the scenario names horizontal; Vega tilts band labels
    # on its own once they crowd, which reads as noise on a two-bar chart.
    scenario = alt.X("Scenario:N", title=None, sort=["Existing", "Proposed"],
                     axis=alt.Axis(labelFontSize=13, labelColor=SLATE,
                                   labelAngle=0, labelPadding=8))
    y_axis = alt.Y("y0:Q", title="Bill ($)",
                   scale=alt.Scale(domain=[0, ceiling * 1.12 if ceiling else 1]))
    color = alt.Color("Segment:N", scale=alt.Scale(domain=segments, range=colors),
                      legend=alt.Legend(orient="bottom", title=None, columns=4))
    tooltip = ["Scenario", "Segment", alt.Tooltip("Amount:Q", format="$,.2f")]

    # Drawn as two layers so the white rule falls only between the fixed charge
    # and the variable block. Tier bands are separated by their shade alone: a
    # rule between them would run straight through the variable total, which sits
    # on that boundary.
    fixed = alt.Chart(data[data["Kind"] == "Fixed charge"]).mark_bar(
        size=96, stroke="white", strokeWidth=1.5,
    ).encode(x=scenario, y=y_axis, y2=alt.Y2("y1:Q"), color=color, tooltip=tooltip)
    variable = alt.Chart(data[data["Kind"] == "Variable charge"]).mark_bar(
        size=96,
    ).encode(x=scenario, y=y_axis, y2=alt.Y2("y1:Q"), color=color, tooltip=tooltip)

    inside = alt.Chart(blocks).mark_text(
        color="white", fontWeight="bold", fontSize=13,
    ).encode(x=scenario, y=alt.Y("y:Q"), text="Label:N")
    layers = [variable, fixed, inside]

    on_top = alt.Chart(totals).mark_text(
        dy=-9, color=SLATE, fontWeight="bold", fontSize=14,
    ).encode(x=scenario, y=alt.Y("Total:Q"), text="Label:N")

    layers.append(on_top)
    chart = alt.layer(*layers)
    # Streamlit stretches the chart to its container; a standalone render has no
    # container, and without a width the fixed-size bars overlap.
    if width:
        chart = chart.properties(width=width)
    return _base(chart, height)


def distribution_bars(frame: pd.DataFrame, x_title: str, height: int = 300,
                      width: int | None = None) -> alt.Chart:
    """Share of customers falling in each impact bucket.

    `frame` has the columns produced by `billing.bucket_counts`: range, count,
    share. Bucket order carries meaning, so the x axis is pinned to the order the
    buckets were built in rather than sorted.
    """
    data = frame.copy()
    data["Label"] = [f"{s:.1%}" if s > 0 else "" for s in data["share"]]
    order = list(data["range"])
    ceiling = float(data["share"].max()) if len(data) else 0.0

    x = alt.X("range:N", title=x_title, sort=order,
              axis=alt.Axis(labelAngle=0, labelFontSize=11, labelColor=SLATE,
                            labelPadding=6))
    bars = alt.Chart(data).mark_bar(size=44, color=EMERALD).encode(
        x=x,
        y=alt.Y("share:Q", title="% of accounts",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, ceiling * 1.18 if ceiling else 1])),
        tooltip=[alt.Tooltip("range:N", title="Range"),
                 alt.Tooltip("count:Q", title="Count", format=",.0f"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")],
    )
    labels = alt.Chart(data).mark_text(
        dy=-8, color=SLATE, fontWeight="bold", fontSize=12,
    ).encode(x=x, y=alt.Y("share:Q"), text="Label:N")

    chart = bars + labels
    if width:
        chart = chart.properties(width=width)
    return _base(chart, height)


def share_comparison_bars(frame: pd.DataFrame, category: str, left: str, right: str,
                          height: int = 340, width: int | None = None) -> alt.Chart:
    """Paired bars comparing two shares per category.

    Built for the "who gains, who loses" exhibit: what a class would carry on
    volume alone against what it carries once peak responsibility is weighted in.
    The first series is drawn in slate as the neutral reference, the second in
    emerald as the result.
    """
    data = frame.melt(id_vars=category, value_vars=[left, right],
                      var_name="Measure", value_name="Share")
    ceiling = float(data["Share"].max()) if len(data) else 0.0

    x = alt.X(f"{category}:N", title=None, sort=None,
              axis=alt.Axis(labelAngle=0, labelFontSize=12, labelColor=SLATE,
                            labelPadding=8))
    offset = alt.XOffset("Measure:N", sort=[left, right])
    color = alt.Color("Measure:N", sort=[left, right],
                      scale=alt.Scale(domain=[left, right], range=[SLATE, EMERALD]),
                      legend=alt.Legend(orient="bottom", title=None))

    bars = alt.Chart(data).mark_bar(size=30).encode(
        x=x, xOffset=offset,
        y=alt.Y("Share:Q", title="Share", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, ceiling * 1.18 if ceiling else 1])),
        color=color,
        tooltip=[category, "Measure", alt.Tooltip("Share:Q", format=".2%")],
    )
    labels = alt.Chart(data).mark_text(
        dy=-7, color=SLATE, fontWeight="bold", fontSize=11,
    ).encode(x=x, xOffset=offset, y=alt.Y("Share:Q"),
             text=alt.Text("Share:Q", format=".1%"))

    chart = bars + labels
    if width:
        chart = chart.properties(width=width)
    return _base(chart, height)


def peaking_bars(frame: pd.DataFrame, label_col: str, value_col: str,
                 height: int = 300) -> alt.Chart:
    ceiling = float(frame[value_col].max()) if len(frame) else 0.0
    x = alt.X(f"{label_col}:N", title=None, sort=None,
              axis=alt.Axis(labelFontSize=12, labelColor=SLATE,
                            labelAngle=0, labelPadding=8))
    bars = alt.Chart(frame).mark_bar(size=42, color=EMERALD).encode(
        x=x,
        y=alt.Y(f"{value_col}:Q", title="Peaking factor",
                scale=alt.Scale(domain=[0, ceiling * 1.15 if ceiling else 1])),
        tooltip=[label_col, alt.Tooltip(f"{value_col}:Q", format=".3f")],
    )
    labels = alt.Chart(frame).mark_text(
        dy=-9, color=SLATE, fontWeight="bold", fontSize=13,
    ).encode(x=x, y=alt.Y(f"{value_col}:Q"),
             text=alt.Text(f"{value_col}:Q", format=".3f"))
    # 1.0 is flat demand; anything above it is the seasonal swing.
    rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return _base(bars + rule + labels, height)
