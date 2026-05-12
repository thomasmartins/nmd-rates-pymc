"""
ECB yield curve data download.

Downloads AAA-rated euro area government bond zero-coupon spot rates
from the ECB Statistical Data Warehouse REST API (daily → monthly).
"""

import io
import warnings
import numpy as np
import pandas as pd
import requests

ECB_API  = "https://data-api.ecb.europa.eu/service/data"
DATASET  = "YC"

# ECB maturity codes → months
MATURITY_MAP = {
    3:   "SR_3M",
    6:   "SR_6M",
    12:  "SR_1Y",
    24:  "SR_2Y",
    36:  "SR_3Y",
    60:  "SR_5Y",
    84:  "SR_7Y",
    120: "SR_10Y",
}


def _fetch_series(mat_code: str, start: str, end: str) -> pd.Series:
    """Download one maturity series from ECB SDW (daily frequency)."""
    key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.{mat_code}"
    url = (
        f"{ECB_API}/{DATASET}/{key}"
        f"?startPeriod={start}&endPeriod={end}"
        f"&format=csvdata&detail=dataonly"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # Column names vary; find the date and value columns
    date_col  = [c for c in df.columns if "TIME" in c.upper()][0]
    value_col = [c for c in df.columns if "OBS_VALUE" in c.upper() or "VALUE" in c.upper()][-1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col)[value_col].astype(float)
    s.name = mat_code
    return s


def download_ecb_yield_curve(
    start: str = "2014-01",
    end:   str = "2024-12",
    maturities: list = None,
) -> pd.DataFrame:
    """
    Download ECB AAA euro area government bond zero-coupon spot rates, monthly.

    Parameters
    ----------
    start, end : 'YYYY-MM' strings
    maturities : list of maturities in months; default [3,6,12,24,36,60,84,120]

    Returns
    -------
    DataFrame (monthly, MS freq) with columns '3m','6m',... Values are % p.a.
    """
    if maturities is None:
        maturities = sorted(MATURITY_MAP.keys())

    series = {}
    for m in maturities:
        code = MATURITY_MAP[m]
        try:
            s = _fetch_series(code, start, end)
            series[m] = s
        except Exception as e:
            warnings.warn(f"Failed to download maturity {m}m ({code}): {e}")

    if not series:
        raise RuntimeError("No ECB data could be downloaded.")

    df = pd.DataFrame(series)
    df.index = pd.DatetimeIndex(df.index)

    # Resample to monthly mean and forward-fill any gaps
    df_monthly = df.resample("MS").mean().ffill()
    df_monthly = df_monthly.loc[start:end]
    df_monthly.columns = [f"{m}m" for m in df_monthly.columns]
    return df_monthly
