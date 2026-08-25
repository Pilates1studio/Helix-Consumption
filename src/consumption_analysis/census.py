"""ACS 5-year income pulls for affordability work.

Kept deliberately thin and *offline at analysis time*: the API is hit once by
``tools/fetch_census.py``, which writes a CSV into the agency's ``geo/`` folder.
The app and the model read that CSV. Nothing in the study path touches the
network, so a run is reproducible and a vintage is pinned to a file you can
diff.

Census requires a free API key (https://api.census.gov/data/key_signup.html).
Put it in the ``CENSUS_API_KEY`` environment variable or pass ``--key``.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://api.census.gov/data/{year}/acs/acs5"

# ACS sentinel values for suppressed / not-applicable estimates.
_NULLS = {-666666666, -999999999, -888888888, -222222222, -333333333, -555555555}

# Estimate (E) and margin of error (M) are pulled together for every variable.
# The MOE is not decoration: at tract and ZCTA level a median-income estimate
# routinely carries a +/- 10-20% band, which is wide enough to move a geography
# across an affordability threshold. Burden bands are built from it.
VARIABLES = {
    "B19013_001": "mhi",            # median household income
    "B19025_001": "aggregate_income",  # aggregate HH income; /households = mean
    "B19081_001": "lq_mean_income",  # mean household income, lowest quintile
    "B19080_001": "lq_upper_limit",  # upper limit of the lowest quintile
    "B25119_002": "mhi_owner",       # median HH income, owner-occupied
    "B25119_003": "mhi_renter",      # median HH income, renter-occupied
    "B01003_001": "population",
    "B11001_001": "households",
    "B25010_001": "avg_household_size",
    "B17001_002": "poverty_count",
    "B17001_001": "poverty_universe",
}


def _get(url: str) -> list[list[str]]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    if not body.lstrip().startswith("["):
        raise RuntimeError(f"Census API returned a non-JSON response:\n{body[:400]}")
    return json.loads(body)


def _build_url(year: int, get_cols: list[str], for_clause: str,
               in_clause: str | None, key: str | None) -> str:
    params = {"get": ",".join(get_cols), "for": for_clause}
    if in_clause:
        params["in"] = in_clause
    if key:
        params["key"] = key
    return BASE.format(year=year) + "?" + urllib.parse.urlencode(params, safe=":,*")


def _frame(rows: list[list[str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    for code, name in VARIABLES.items():
        for suffix, label in (("E", name), ("M", f"{name}_moe")):
            col = f"{code}{suffix}"
            if col not in frame.columns:
                continue
            values = pd.to_numeric(frame[col], errors="coerce")
            frame[label] = values.where(~values.isin(_NULLS))
            frame = frame.drop(columns=[col])
    return frame


def fetch_zcta(zctas: list[str], year: int = 2023,
               key: str | None = None) -> pd.DataFrame:
    """Pull ACS 5-year estimates for a list of ZIP Code Tabulation Areas.

    From ACS 2020 onward ZCTAs no longer nest inside states, so the API takes
    no ``in=state:`` predicate here — the codes are queried nationally. Passing
    a state predicate is the most common way this call fails.
    """
    key = key or os.environ.get("CENSUS_API_KEY")
    codes = ",".join(sorted({str(z).strip()[:5].zfill(5) for z in zctas}))
    get_cols = ["NAME"] + [f"{c}{s}" for c in VARIABLES for s in ("E", "M")]
    url = _build_url(year, get_cols, f"zip code tabulation area:{codes}", None, key)
    frame = _frame(_get(url))
    frame = frame.rename(columns={"zip code tabulation area": "geoid"})
    frame["geo_level"] = "zcta"
    frame["acs_year"] = year
    return frame


def fetch_tracts(state: str, counties: list[str], year: int = 2023,
                 key: str | None = None) -> pd.DataFrame:
    """Pull ACS 5-year estimates for every tract in the given counties.

    ``state`` and ``counties`` are FIPS codes as strings ("06", ["073"]).
    Tracts are the defensible geography; ZCTA is the fallback when only ZIP
    codes are available on the account file.
    """
    key = key or os.environ.get("CENSUS_API_KEY")
    get_cols = ["NAME"] + [f"{c}{s}" for c in VARIABLES for s in ("E", "M")]
    out = []
    for county in counties:
        url = _build_url(year, get_cols, "tract:*",
                         f"state:{state} county:{county}", key)
        out.append(_frame(_get(url)))
    frame = pd.concat(out, ignore_index=True)
    frame["geoid"] = frame["state"] + frame["county"] + frame["tract"]
    frame["geo_level"] = "tract"
    frame["acs_year"] = year
    return frame


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the burden calculation wants but ACS does not publish."""
    frame = frame.copy()
    universe = frame.get("poverty_universe")
    if universe is not None:
        frame["poverty_rate"] = (frame["poverty_count"] / universe).where(universe > 0)

    # ACS publishes aggregate household income, not the mean. The mean is worth
    # deriving for one reason: mean / median is a skew ratio. A geography where
    # the mean sits far above the median has a long upper tail, which means a
    # single central measure describes it badly and the median is the *safer*
    # of the two, not the interchangeable one.
    households = frame.get("households")
    if households is not None and "aggregate_income" in frame.columns:
        frame["mean_income"] = (frame["aggregate_income"] / households).where(households > 0)
        frame["income_skew"] = (frame["mean_income"] / frame["mhi"]).where(frame["mhi"] > 0)
    return frame
