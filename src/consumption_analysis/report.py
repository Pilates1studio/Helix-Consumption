"""Write the study out as a formatted Excel workbook.

Only summary-level tables go into the workbook. Account-level detail - tens of
thousands of rows per class - is written alongside as Parquet, which is what
keeps the deliverable small enough to open and email.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import StudyConfig
from .model import StudyResult

TITLE_ROW = 0


class _Formats:
    def __init__(self, book):
        self.title = book.add_format({"bold": True, "font_size": 14})
        self.subtitle = book.add_format({"italic": True, "font_color": "#555555"})
        self.header = book.add_format({"bold": True, "bg_color": "#1F3864",
                                       "font_color": "white", "border": 1,
                                       "text_wrap": True, "valign": "vcenter"})
        self.section = book.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        self.label = book.add_format({"bold": True})
        self.number = book.add_format({"num_format": "#,##0"})
        self.decimal = book.add_format({"num_format": "#,##0.00"})
        self.money = book.add_format({"num_format": "$#,##0.00"})
        self.percent = book.add_format({"num_format": "0.0%"})
        self.factor = book.add_format({"num_format": "0.000"})
        self.input = book.add_format({"font_color": "#0000FF", "num_format": "#,##0.00"})
        self.total = book.add_format({"bold": True, "top": 1, "num_format": "#,##0"})


def _write_table(sheet, fmts, frame: pd.DataFrame, row: int, col: int, title: str,
                 number_format=None, index_header: str = "") -> int:
    """Write a DataFrame with its index as the first column. Returns the next free row."""
    sheet.write(row, col, title, fmts.section)
    row += 1
    sheet.write(row, col, index_header or (frame.index.name or ""), fmts.header)
    for j, name in enumerate(frame.columns):
        sheet.write(row, col + 1 + j, str(name), fmts.header)
    row += 1
    for label, values in frame.iterrows():
        is_total = str(label).strip().lower() == "total"
        sheet.write(row, col, str(label), fmts.label if is_total else None)
        for j, value in enumerate(values):
            fmt = fmts.total if is_total else (number_format or fmts.number)
            if isinstance(value, (bool,)):
                sheet.write(row, col + 1 + j, "yes" if value else "NO")
            elif pd.isna(value):
                sheet.write_blank(row, col + 1 + j, None)
            else:
                sheet.write_number(row, col + 1 + j, float(value), fmt)
        row += 1
    return row + 1


def write(result: StudyResult, path: str | Path, account_detail: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = result.cfg

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        book = writer.book
        fmts = _Formats(book)
        _write_dashboard(book, fmts, cfg, result)
        _write_summary(book, fmts, cfg, result)
        _write_peak_contribution(book, fmts, cfg, result)
        for name in result.class_names:
            _write_class(book, fmts, cfg, result, name)
        _write_checks(book, fmts, result)

    if account_detail:
        detail_dir = path.parent / f"{path.stem}_detail"
        detail_dir.mkdir(exist_ok=True)
        for name, res in result.classes.items():
            dest = detail_dir / f"{name.lower()}.parquet"
            res.account_detail(cfg.periods).to_parquet(dest, index=False, compression="zstd")
    return path


def _write_dashboard(book, fmts, cfg: StudyConfig, result: StudyResult) -> None:
    sheet = book.add_worksheet("Dashboard")
    sheet.set_column(0, 0, 32)
    sheet.set_column(1, 12, 14)
    sheet.write(0, 0, cfg.agency, fmts.title)
    sheet.write(1, 0, f"{cfg.title} - {result.year}", fmts.subtitle)
    sheet.write(2, 0, f"{cfg.n_periods} billing periods per year, "
                      f"{cfg.days_per_period} days per period", fmts.subtitle)

    row = 4
    names = result.class_names
    sheet.write(row, 0, "Existing Tier Widths", fmts.section)
    for j, name in enumerate(names):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1
    for t in range(5):
        sheet.write(row, 0, f"Tier {t + 1}", fmts.label)
        for j, name in enumerate(names):
            width = cfg.customer_classes[name].existing.tier_widths[t]
            sheet.write(row, 1 + j, "budget" if width is None else float(width), fmts.input)
        row += 1

    row += 1
    sheet.write(row, 0, "Existing Commodity Rates ($/unit)", fmts.section)
    for j, name in enumerate(names):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1
    for t in range(5):
        sheet.write(row, 0, f"Tier {t + 1}", fmts.label)
        for j, name in enumerate(names):
            sheet.write(row, 1 + j,
                        cfg.customer_classes[name].existing.commodity_rates[t], fmts.money)
        row += 1

    row += 1
    sheet.write(row, 0, "Existing Fixed Meter Charges ($)", fmts.section)
    for j, name in enumerate(names):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1
    for size in cfg.meter_sizes:
        sheet.write(row, 0, size, fmts.label)
        for j, name in enumerate(names):
            sheet.write(row, 1 + j,
                        cfg.customer_classes[name].existing.meter_charges.get(size, 0.0),
                        fmts.money)
        row += 1

    blank = [n for n in names if cfg.customer_classes[n].revised.is_empty()]
    if blank:
        row += 1
        sheet.write(row, 0, "Revised rates not yet entered for: " + ", ".join(blank),
                    fmts.subtitle)


def _write_summary(book, fmts, cfg: StudyConfig, result: StudyResult) -> None:
    sheet = book.add_worksheet("Summary")
    sheet.set_column(0, 0, 24)
    sheet.set_column(1, 14, 15)
    sheet.write(0, 0, cfg.agency, fmts.title)
    sheet.write(1, 0, f"{cfg.title} - Summary - {result.year}", fmts.subtitle)

    row = 3
    row = _write_table(sheet, fmts, result.usage_by_class(), row, 0,
                       f"Usage by Customer Class ({cfg.units})",
                       index_header="Customer Class")

    tiers = result.tier_summary().set_index(["Customer Class", "Tier"])
    flat = tiers.copy()
    flat.index = [f"{c} - {t}" for c, t in tiers.index]
    row = _write_table(sheet, fmts, flat, row, 0,
                       f"Usage by Tier ({cfg.units}) - Existing Structure",
                       index_header="Class / Tier")

    _write_peaking(sheet, fmts, result, row)


def _write_peak_contribution(book, fmts, cfg: StudyConfig, result: StudyResult) -> None:
    """Peak responsibility measured as deviation from the system as a whole.

    Kept on its own sheet rather than merged into Summary: it is an alternative
    basis for allocating peak cost, not a refinement of the one there, and a
    reviewer needs to be able to compare the two side by side.
    """
    frame = result.peak_contribution()
    if frame.empty:
        return
    meta = frame.attrs

    sheet = book.add_worksheet("Peak Contribution")
    sheet.set_column(0, 0, 24)
    sheet.set_column(1, 9, 17)
    sheet.write(0, 0, f"{cfg.agency} - Peak Contribution", fmts.title)
    sheet.write(1, 0, f"{result.year} - basis: usage per contributing account",
                fmts.subtitle)

    row = 3
    for line in (
        "How much did this class contribute to the peak we actually had to serve?",
        "",
        "A class's own peaking factor says how much its demand swells, but not whether that",
        "swelling is unusual. If every class peaked identically, none would be responsible for",
        "more of the system peak than its share of volume. Each class's factor is therefore",
        "divided by the system's own:",
        "",
        "        relative factor = class peaking factor / system peaking factor",
        "",
        "1.000 means the class moves exactly with the system and carries its volume share of",
        "peak cost. Above 1.000 it drives the peak harder and should carry more. The system",
        "aggregate includes the class being measured: under a uniform variable rate the whole",
        "system is what the rate recovers against, so the whole system is the correct benchmark.",
    ):
        sheet.write(row, 0, line, fmts.subtitle if line else None)
        row += 1
    row += 1

    for label, value, fmt in (
        ("System peak period", meta["peak_period"], None),
        ("System peaking factor", meta["system_factor"], fmts.factor),
        (f"System usage per account at peak ({cfg.units})", meta["system_peak"], fmts.decimal),
        (f"System usage per account, average ({cfg.units})", meta["system_average"], fmts.decimal),
        ("Contributing accounts at peak", meta["system_accounts"], fmts.number),
    ):
        sheet.write(row, 0, label, fmts.label)
        if fmt is None:
            sheet.write(row, 1, str(value))
        else:
            sheet.write_number(row, 1, float(value), fmt)
        row += 1
    row += 1

    sheet.write(row, 0, "Peak Responsibility by Customer Class", fmts.section)
    row += 1
    columns = ["Accounts at Peak", "Peak per Account", "Average per Account",
               "Class Peaking Factor", "Relative Peaking Factor", "Total Usage",
               "Share of Usage", "Weighted Usage", "Peak Cost Allocation"]
    formats = [fmts.number, fmts.decimal, fmts.decimal, fmts.factor, fmts.factor,
               fmts.number, fmts.percent, fmts.number, fmts.percent]
    sheet.write(row, 0, "Customer Class", fmts.header)
    for j, name in enumerate(columns):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1
    for _, values in frame.iterrows():
        sheet.write(row, 0, str(values["Customer Class"]))
        for j, (name, fmt) in enumerate(zip(columns, formats)):
            sheet.write_number(row, 1 + j, float(values[name]), fmt)
        row += 1
    row += 1
    weighted_total = float(frame["Weighted Usage"].sum())
    sheet.write(row, 0, "Weighted Usage = Total Usage x Relative Peaking Factor. Divide it by "
                        f"the column total ({weighted_total:,.0f} {cfg.units}) to reproduce "
                        "Peak Cost Allocation by hand.", fmts.subtitle)
    row += 1
    sheet.write(row, 0, "Compare Peak Cost Allocation against Share of Usage: a class below "
                        "1.000 uses water more evenly through the year, does not drive the peak "
                        "as hard, and carries a smaller share of peak cost than its volume "
                        "alone would suggest. Only the peak-related component moves.",
                fmts.subtitle)
    row += 2

    row = _write_peak_reconciliation(sheet, fmts, cfg, result, row)

    row = _write_tier_step(sheet, fmts, cfg, result, row, meta["peak_period"])

    index = result.seasonal_index_by_period()
    sheet.write(row, 0, "Seasonal Index by Period", fmts.section)
    row += 1
    sheet.write(row, 0, "Class per-account demand vs the system's, each expressed against its "
                        "own average so account size cancels. 1.000 = moved with the system.",
                fmts.subtitle)
    row += 1
    sheet.write(row, 0, f"The {meta['peak_period']} column is the Relative Peaking Factor above.",
                fmts.subtitle)
    row += 2
    sheet.write(row, 0, "Customer Class", fmts.header)
    for j, period in enumerate(index.columns):
        sheet.write(row, 1 + j, str(period), fmts.header)
    row += 1
    for label, values in index.iterrows():
        sheet.write(row, 0, str(label))
        for j, value in enumerate(values):
            sheet.write_number(row, 1 + j, float(value), fmts.factor)
        row += 1


def _write_peak_reconciliation(sheet, fmts, cfg: StudyConfig, result: StudyResult,
                               row: int) -> int:
    """Pre-computed answers to the two questions a reviewer will raise."""
    frame = result.peak_reconciliation()
    if frame.empty:
        return row
    meta = frame.attrs

    sheet.write(row, 0, "Reconciliation - How the Relative Factor Behaves", fmts.section)
    row += 1
    for line in (
        "Check 1. Under a volume-weighted, renormalised allocation the relative factor changes",
        "nothing: a constant divisor applied to every class cancels out. The relative framing",
        "CONFIRMS the conventional allocation rather than replacing it. Present it as the nexus",
        "narrative and a fairness test, not as a different set of dollars.",
        "",
        "Check 2. Under a base-extra capacity formulation the '- 1' is load-bearing and the",
        "divisor no longer cancels. Each class's OWN factor recovers essentially the whole system",
        "excess. The relative factors sum to approximately zero - they average to 1.000 by",
        "construction, so classes above the system are offset by those below - and therefore",
        "cannot allocate a positive extra-capacity pool. Strike extra capacity on the own factor.",
    ):
        sheet.write(row, 0, line, fmts.subtitle if line else None)
        row += 1
    row += 1

    columns = ["Own Factor", "Relative Factor", "Allocation on Own Factor",
               "Allocation on Relative Factor", "Difference",
               "Extra Capacity on Own Factor", "Extra Capacity on Relative Factor"]
    formats = [fmts.factor, fmts.factor, fmts.percent, fmts.percent, fmts.percent,
               fmts.number, fmts.number]
    sheet.write(row, 0, "Customer Class", fmts.header)
    for j, name in enumerate(columns):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1
    for _, values in frame.iterrows():
        sheet.write(row, 0, str(values["Customer Class"]))
        for j, (col, fmt) in enumerate(zip(columns, formats)):
            sheet.write_number(row, 1 + j, float(values[col]), fmt)
        row += 1
    row += 1

    units = cfg.units
    for label, value, fmt in (
        ("Largest allocation difference (check 1 - expect zero)",
         meta["max_allocation_difference"], fmts.percent),
        (f"System excess capacity: total volume x (system factor - 1) ({units})",
         meta["system_excess"], fmts.number),
        (f"Recovered by own factors ({units})", meta["own_excess"], fmts.number),
        ("  as a share of system excess", meta["own_excess_share"], fmts.percent),
        (f"Recovered by relative factors ({units})", meta["relative_excess"], fmts.number),
        ("  as a share of system excess", meta["relative_excess_share"], fmts.percent),
    ):
        sheet.write(row, 0, label, fmts.label)
        sheet.write_number(row, 1, float(value), fmt)
        row += 1
    return row + 1


def _write_tier_step(sheet, fmts, cfg: StudyConfig, result: StudyResult, row: int,
                     peak_period: str) -> int:
    """Step 2: each class's assigned peak cost distributed to its tiered subclasses."""
    tiered = [n for n in result.class_names if result.has_tiers(n)]
    if not tiered:
        return row

    sheet.write(row, 0, "Step 2 - Peak Cost Allocation to Tiered Subclasses", fmts.section)
    row += 1
    for line in (
        "Step 1 assigned each class its share of system peak cost. If every account in a class",
        f"used the same amount of water - the uniform case behind that class's relative peak - each",
        f"account would use the class average shown below in {peak_period}. But a class's accounts in",
        f"{peak_period} include customers falling in different tiers. Grouping accounts by the tier",
        "they reached gives each tier's share of overall demand in that period, allocating the",
        "class's peak costs to tiers in relation to how each contributes to the class total.",
        "",
        "        factor = subclass usage per account in peak / class usage per account in peak",
        "",
        f"Only the peak period ({peak_period}) is examined: upper tiers ARE peak usage by",
        "construction, so judging them against their own annual pattern would let a consistently",
        "high tier register as not peaking at all. The accounts stopping in a tier are that tier's",
        "subclass, so the subclasses partition the class: the account-weighted average of the",
        "factors is exactly 1.000, and the allocation reduces to each subclass's share of class",
        "usage in the peak period.",
    ):
        sheet.write(row, 0, line, fmts.subtitle if line else None)
        row += 1
    row += 1

    columns = ["Accounts in Peak", "Usage in Peak", "Usage per Account",
               "Factor vs Class", "Annual Usage", "Share of Class Peak Cost",
               "Share of Annual Usage"]
    formats = [fmts.number, fmts.number, fmts.decimal, fmts.factor, fmts.number,
               fmts.percent, fmts.percent]
    sheet.write(row, 0, "Class / Tier", fmts.header)
    for j, name in enumerate(columns):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1

    budget_based = False
    for name in tiered:
        frame = result.tier_peak_contribution(name)
        meta = frame.attrs
        sheet.write(row, 0, f"{name} - class average in {peak_period}", fmts.label)
        sheet.write_number(row, 3, meta["class_per_account"], fmts.decimal)
        row += 1
        for _, values in frame.iterrows():
            sheet.write(row, 0, f"{name} - {values['Tier']}")
            for j, (col, fmt) in enumerate(zip(columns, formats)):
                sheet.write_number(row, 1 + j, float(values[col]), fmt)
            row += 1
        row += 1
        budget_based |= cfg.customer_classes[name].is_budget_based

    if budget_based:
        for line in (
            "Budget-based classes - expect factors near 1.000.",
            "Tier 1 is each account's water budget, and the budget is itself seasonal: it rises and",
            "falls with weather so a customer irrigating normally stays inside it all year.",
            "Within-budget demand therefore moves with the class rather than against it, and the tier",
            "split separates within-budget from over-budget usage rather than light from heavy users.",
            "Compressed factors are the structure working, not an anomaly.",
            "",
            "Read 'near 1.000' as PROPORTIONATE, not EXEMPT. Within-budget usage is the most seasonal",
            "water in the class in absolute terms - the system still has to build for it - so it",
            "carries a full proportionate share of peak cost, not a reduced one.",
        ):
            sheet.write(row, 0, line, fmts.subtitle if line else None)
            row += 1
        row += 1
    return row


def _write_peaking(sheet, fmts, result: StudyResult, row: int) -> int:
    """Peaking factors on usage per contributing account, by class and by tier."""
    peak_period = result.system_peak_period()
    sheet.write(row, 0, "Peaking Factors - basis: usage per contributing account",
                fmts.section)
    row += 1
    sheet.write(row, 0, f"Peak period pinned to the system peak ({peak_period}). "
                        "Factor = peak period / average period.", fmts.subtitle)
    row += 2

    peaking = result.peaking(basis="per_account").set_index("Customer Class")
    peaking = peaking[["Total Usage", "peak", "average", "peaking_factor"]]
    peaking.columns = ["Total Usage", "Peak per Account", "Average per Account",
                       "Peaking Factor"]
    sheet.write(row, 0, "Customer Class", fmts.header)
    for j, name in enumerate(peaking.columns):
        sheet.write(row, 1 + j, name, fmts.header)
    row += 1
    for label, values in peaking.iterrows():
        sheet.write(row, 0, str(label))
        for j, value in enumerate(values):
            fmt = fmts.factor if j == 3 else (fmts.number if j == 0 else fmts.decimal)
            sheet.write_number(row, 1 + j, float(value), fmt)
        row += 1
    row += 1

    tiered = [n for n in result.class_names if result.has_tiers(n)]
    if not tiered:
        return row
    sheet.write(row, 0, "Peaking Factors by Tier", fmts.section)
    row += 1
    sheet.write(row, 0, "Upper tiers are seasonal demand and peak harder than Tier 1.",
                fmts.subtitle)
    row += 1
    headers = ["Class / Tier", "Usage", "Contributing Accounts", "Peak per Account",
               "Average per Account", "Peaking Factor"]
    for j, name in enumerate(headers):
        sheet.write(row, j, name, fmts.header)
    row += 1
    for name in tiered:
        for _, values in result.tier_peaking(name, basis="per_account").iterrows():
            sheet.write(row, 0, f"{name} - {values['Tier']}")
            for j, (value, fmt) in enumerate((
                (values["Usage"], fmts.number),
                (values["Accounts"], fmts.number),
                (values["peak"], fmts.decimal),
                (values["average"], fmts.decimal),
                (values["peaking_factor"], fmts.factor),
            )):
                sheet.write_number(row, 1 + j, float(value), fmt)
            row += 1
    return row


def _write_class(book, fmts, cfg: StudyConfig, result: StudyResult, name: str) -> None:
    res = result.classes[name]
    sheet = book.add_worksheet(name[:31])
    sheet.set_column(0, 0, 26)
    sheet.set_column(1, 14, 15)
    sheet.write(0, 0, f"{cfg.agency} - {name}", fmts.title)

    klass = cfg.customer_classes[name]
    basis = ("budget-based (Tier 1 from the water-budget table)"
             if klass.is_budget_based
             else f"volumetric, tier widths prorated over {cfg.days_per_period} days")
    sheet.write(1, 0, f"{result.year} - {basis}", fmts.subtitle)

    summary = res.summary_existing
    row = 3
    row = _write_table(sheet, fmts, summary.meter_counts, row, 0,
                       "Meter Counts", index_header="Meter Size")
    row = _write_table(sheet, fmts, summary.usage_by_tier, row, 0,
                       f"Usage by Tier ({cfg.units})", index_header="Tier")
    row = _write_table(sheet, fmts, summary.usage_stopped_in_tier, row, 0,
                       f"Usage Stopped in Tier ({cfg.units})", index_header="Tier")
    row = _write_table(sheet, fmts, summary.contributing_accounts, row, 0,
                       "Contributing Accounts", index_header="Tier")
    row = _write_table(sheet, fmts, summary.usage_per_account, row, 0,
                       f"Usage per Contributing Account ({cfg.units})",
                       number_format=fmts.decimal, index_header="Tier")

    impacts = res.impact_tables(cfg)
    for key, title in (("bill_impacts", "Bill Impacts by Period ($ change)"),
                       ("account_impacts", "Account Impacts (annual % change)")):
        frame = impacts.get(key)
        if frame is None:
            continue
        table = frame.set_index("range")
        sheet.write(row, 0, title, fmts.section)
        row += 1
        sheet.write(row, 0, "Range", fmts.header)
        sheet.write(row, 1, "Count", fmts.header)
        sheet.write(row, 2, "Share", fmts.header)
        row += 1
        for label, values in table.iterrows():
            sheet.write(row, 0, str(label))
            sheet.write_number(row, 1, float(values["count"]), fmts.number)
            sheet.write_number(row, 2, float(values["share"]), fmts.percent)
            row += 1
        row += 1


def _write_checks(book, fmts, result: StudyResult) -> None:
    sheet = book.add_worksheet("Checks")
    sheet.set_column(0, 0, 18)
    sheet.set_column(1, 6, 24)
    sheet.write(0, 0, "Internal Consistency Checks", fmts.title)
    sheet.write(1, 0, "Usage priced by tier must equal usage attributed to a stopping "
                      "tier; contributing plus no-usage accounts must equal the class.",
                fmts.subtitle)
    checks = result.checks()
    row = 3
    for j, name in enumerate(checks.columns):
        sheet.write(row, j, str(name), fmts.header)
    row += 1
    for _, values in checks.iterrows():
        for j, value in enumerate(values):
            if isinstance(value, bool):
                sheet.write(row, j, "yes" if value else "NO")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                sheet.write_number(row, j, float(value), fmts.number)
            else:
                sheet.write(row, j, str(value))
        row += 1
