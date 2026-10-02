"""
Smog Net Data Loader
====================
Downloads, caches, and joins CPCB Delhi daily air quality observations (2015-2020)
with NASA POWER daily surface meteorology (T2M, T2M_MAX, T2M_MIN, RH2M, WS10M, WD10M, PRECTOTCORR).
Applies time-indexed linear interpolation (limit=3 days) for short pollutant gaps.
"""

import io
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import requests

CPCB_URL = "https://raw.githubusercontent.com/adityarc19/aqi-india/master/city_day.csv"
NASA_POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS10M,WD10M,PRECTOTCORR"
    "&community=RE&longitude=77.2090&latitude=28.6139"
    "&start=20150101&end=20200701&format=CSV"
)

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]
METEO_COLS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M", "WD10M", "PRECTOTCORR"]


def fetch_cpcb_data(data_dir: Path) -> pd.DataFrame:
    """Fetch CPCB air quality dataset, filtering for Delhi."""
    cpcb_cache = data_dir / "city_day.csv"
    if cpcb_cache.exists():
        print(f"[DataLoader] Loading cached CPCB data from {cpcb_cache}")
        df = pd.read_csv(cpcb_cache)
    else:
        print(f"[DataLoader] Downloading CPCB data from {CPCB_URL}...")
        resp = requests.get(CPCB_URL, timeout=45)
        resp.raise_for_status()
        with open(cpcb_cache, "wb") as f:
            f.write(resp.content)
        df = pd.read_csv(io.BytesIO(resp.content))

    delhi_df = df[df["City"] == "Delhi"].copy()
    delhi_df["Date"] = pd.to_datetime(delhi_df["Date"])
    delhi_df = delhi_df.sort_values("Date").reset_index(drop=True)
    print(f"[DataLoader] Loaded {len(delhi_df)} daily rows for Delhi ({delhi_df['Date'].min().strftime('%Y-%m-%d')} to {delhi_df['Date'].max().strftime('%Y-%m-%d')})")
    return delhi_df


def fetch_nasa_power_data(data_dir: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Fetch Delhi meteorological data from NASA POWER API.
    Caches raw CSV response to prevent re-fetching.
    Returns DataFrame and boolean indicating if synthetic fallback was used.
    """
    nasa_cache = data_dir / "nasa_power_delhi.csv"
    is_synthetic = False

    if nasa_cache.exists():
        print(f"[DataLoader] Loading cached NASA POWER meteorology from {nasa_cache}")
        with open(nasa_cache, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        print("[DataLoader] Fetching meteorology from NASA POWER API...")
        try:
            resp = requests.get(NASA_POWER_URL, timeout=45)
            resp.raise_for_status()
            raw_text = resp.text
            with open(nasa_cache, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"[DataLoader] Cached NASA POWER meteorology to {nasa_cache}")
        except Exception as e:
            print(f"[DataLoader] WARNING: NASA POWER API call failed ({e}). Falling back to synthetic weather.")
            is_synthetic = True

    if is_synthetic:
        # Fallback to clearly-labeled synthetic placeholder ONLY if API fails
        # Generates realistic seasonal temperature, humidity, wind patterns for Delhi
        dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")
        doy = dates.dayofyear.values
        t2m = 25.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        rh2m = 55.0 + 20.0 * np.sin(2 * np.pi * (doy - 200) / 365.25)
        ws10m = 3.0 + 1.2 * np.sin(2 * np.pi * doy / 365.25)
        wd10m = 270.0 + 40.0 * np.cos(2 * np.pi * doy / 365.25)
        meteo_df = pd.DataFrame({
            "Date": dates,
            "T2M": t2m,
            "T2M_MAX": t2m + 5.0,
            "T2M_MIN": t2m - 5.0,
            "RH2M": np.clip(rh2m, 20.0, 95.0),
            "WS10M": np.clip(ws10m, 0.5, 8.0),
            "WD10M": wd10m % 360.0,
            "PRECTOTCORR": np.zeros(len(dates))
        })
        return meteo_df, True

    # Parse NASA POWER CSV header
    lines = raw_text.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if "-END HEADER-" in line:
            header_idx = i
            break
    if header_idx == -1:
        csv_content = raw_text
    else:
        csv_content = "\n".join(lines[header_idx + 1:])

    raw_meteo = pd.read_csv(io.StringIO(csv_content))
    raw_meteo.replace(-999, np.nan, inplace=True)

    # Construct Date column
    raw_meteo["Date"] = pd.to_datetime(
        raw_meteo["YEAR"].astype(str) + "-" +
        raw_meteo["MO"].astype(str).str.zfill(2) + "-" +
        raw_meteo["DY"].astype(str).str.zfill(2)
    )

    keep_cols = ["Date"] + [c for c in METEO_COLS if c in raw_meteo.columns]
    meteo_df = raw_meteo[keep_cols].sort_values("Date").reset_index(drop=True)
    print(f"[DataLoader] NASA POWER loaded: {len(meteo_df)} days with variables {METEO_COLS}")
    return meteo_df, False


def load_and_preprocess_delhi_data(data_dir_str: str = "data") -> Tuple[pd.DataFrame, bool]:
    """
    Load CPCB Delhi air quality, Gurugram air quality (for spatial lags/gradients),
    and NASA POWER meteorology, merge on Date, apply time-indexed linear interpolation,
    and drop unrecoverable PM2.5 rows.
    """
    data_dir = Path(data_dir_str)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CPCB data
    cpcb_cache = data_dir / "city_day.csv"
    if cpcb_cache.exists():
        print(f"[DataLoader] Loading cached CPCB data from {cpcb_cache}")
        cpcb_raw = pd.read_csv(cpcb_cache)
    else:
        print(f"[DataLoader] Downloading CPCB data from {CPCB_URL}...")
        resp = requests.get(CPCB_URL, timeout=45)
        resp.raise_for_status()
        with open(cpcb_cache, "wb") as f:
            f.write(resp.content)
        cpcb_raw = pd.read_csv(io.BytesIO(resp.content))

    # Delhi primary observations
    delhi_df = cpcb_raw[cpcb_raw["City"] == "Delhi"].copy()
    delhi_df["Date"] = pd.to_datetime(delhi_df["Date"])
    delhi_df = delhi_df.sort_values("Date").reset_index(drop=True)

    # Gurugram spatial neighbor observations (Tier 1 spatial feature)
    gurugram_df = cpcb_raw[cpcb_raw["City"] == "Gurugram"].copy()
    gurugram_df["Date"] = pd.to_datetime(gurugram_df["Date"])
    gurugram_df = gurugram_df.sort_values("Date").reset_index(drop=True)
    gurugram_sub = gurugram_df[["Date", "PM2.5", "PM10", "NO2"]].rename(
        columns={
            "PM2.5": "GURUGRAM_PM2.5",
            "PM10": "GURUGRAM_PM10",
            "NO2": "GURUGRAM_NO2",
        }
    )

    # 2. NASA POWER Meteorology
    meteo_df, is_synthetic = fetch_nasa_power_data(data_dir)

    # 3. Merge Delhi + Gurugram + NASA POWER
    merged_df = pd.merge(delhi_df, gurugram_sub, on="Date", how="left")
    merged_df = pd.merge(merged_df, meteo_df, on="Date", how="left")
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    # Set Date as time index for time-indexed linear interpolation
    merged_df = merged_df.set_index("Date")

    # Time-indexed linear interpolation (limit=3 days) for pollutant variables & meteo
    spatial_cols = ["GURUGRAM_PM2.5", "GURUGRAM_PM10", "GURUGRAM_NO2"]
    interpolate_cols = [c for c in POLLUTANT_COLS + METEO_COLS + spatial_cols if c in merged_df.columns]
    merged_df[interpolate_cols] = merged_df[interpolate_cols].interpolate(
        method="time", limit=3, limit_direction="both"
    )

    # Drop rows where target PM2.5 itself is unrecoverable
    unrecoverable = merged_df["PM2.5"].isna().sum()
    if unrecoverable > 0:
        print(f"[DataLoader] Dropping {unrecoverable} rows where PM2.5 is unrecoverable after interpolation.")
        merged_df = merged_df.dropna(subset=["PM2.5"])

    merged_df = merged_df.reset_index()
    print(f"[DataLoader] Final merged dataset: {len(merged_df)} rows from {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"[DataLoader] Gurugram spatial neighbor joined ({merged_df['GURUGRAM_PM2.5'].notna().sum()} valid days)")

    return merged_df, is_synthetic


if __name__ == "__main__":
    df, syn = load_and_preprocess_delhi_data()
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("Head:\n", df[["Date", "PM2.5", "PM10", "T2M", "RH2M", "WS10M"]].head())
