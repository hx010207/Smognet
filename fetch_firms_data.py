"""
NASA FIRMS Active Fire Downloader & Aggregator (Parallelized)
============================================================
Queries official NASA FIRMS Area API for MODIS Standard Processing (MODIS_SP)
active fire detections across the Punjab & Haryana agricultural region
(Bounding box: 73.8°E, 27.5°N, 77.6°E, 32.5°N) for 2015-01-01 to 2020-07-01.

Aggregates daily fire counts and total Fire Radiative Power (FRP, in MW).
Caches daily results locally to data/firms_punjab_haryana_daily_2015_2020.csv.
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests

# Load NASA FIRMS API key from environment variable
MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
BBOX = "73.8,27.5,77.6,32.5"  # west, south, east, north (Punjab & Haryana)
SOURCE = "MODIS_SP"
DAY_RANGE = 5

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "firms_punjab_haryana_daily_2015_2020.csv"
RAW_DIR = DATA_DIR / "firms_raw_chunks"
RAW_DIR.mkdir(exist_ok=True)


def get_chunk(start_date_str: str) -> pd.DataFrame:
    raw_path = RAW_DIR / f"chunk_{start_date_str}.csv"
    if raw_path.exists():
        try:
            return pd.read_csv(raw_path)
        except Exception:
            pass

    if not MAP_KEY:
        raise ValueError(
            "NASA FIRMS MAP_KEY not set. Please set the 'FIRMS_MAP_KEY' environment variable. "
            "Get a free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAY_RANGE}/{start_date_str}"
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                df = pd.read_csv(io.StringIO(resp.text))
                return df
            elif resp.status_code == 400:
                return pd.DataFrame()
            else:
                time.sleep(1.0 + attempt * 0.5)
        except Exception:
            time.sleep(1.0 + attempt * 0.5)
    return pd.DataFrame()


def download_all_firms_data() -> pd.DataFrame:
    if CACHE_FILE.exists():
        print(f"[FIRMS] Loading cached daily fire dataset from {CACHE_FILE}")
        df_cached = pd.read_csv(CACHE_FILE)
        df_cached["Date"] = pd.to_datetime(df_cached["Date"])
        return df_cached

    print("[FIRMS] Connecting to NASA FIRMS Area API in parallel (max_workers=6)...")
    start_dt = pd.Timestamp("2015-01-01")
    end_dt = pd.Timestamp("2020-07-01")
    
    date_list = []
    curr = start_dt
    while curr <= end_dt:
        date_list.append(curr.strftime("%Y-%m-%d"))
        curr += pd.Timedelta(days=DAY_RANGE)

    print(f"[FIRMS] Total 5-day intervals to process: {len(date_list)}")
    all_chunks = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_date = {executor.submit(get_chunk, d): d for d in date_list}
        completed = 0
        for future in as_completed(future_to_date):
            d = future_to_date[future]
            try:
                df_chunk = future.result()
                if not df_chunk.empty and "acq_date" in df_chunk.columns:
                    all_chunks.append(df_chunk)
            except Exception as e:
                print(f"[FIRMS] Error fetching {d}: {e}")
            completed += 1
            if completed % 50 == 0 or completed == len(date_list):
                print(f"[FIRMS] Completed {completed}/{len(date_list)} intervals...")

    print(f"[FIRMS] Finished downloading all chunks.")
    if all_chunks:
        df_all = pd.concat(all_chunks, ignore_index=True).drop_duplicates()
        print(f"[FIRMS] Total unique satellite fire detections downloaded: {len(df_all)}")
        df_all["acq_date"] = pd.to_datetime(df_all["acq_date"])
        
        # Group by acquisition date
        daily = df_all.groupby("acq_date").agg(
            firms_fire_count=("frp", "count"),
            firms_frp_total=("frp", "sum"),
            firms_frp_mean=("frp", "mean"),
            firms_confidence_mean=("confidence", "mean")
        ).reset_index()
        daily = daily.rename(columns={"acq_date": "Date"})
    else:
        daily = pd.DataFrame(columns=["Date", "firms_fire_count", "firms_frp_total", "firms_frp_mean", "firms_confidence_mean"])

    full_dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")
    df_full = pd.DataFrame({"Date": full_dates}).merge(daily, on="Date", how="left")
    df_full["firms_fire_count"] = df_full["firms_fire_count"].fillna(0).astype(int)
    df_full["firms_frp_total"] = df_full["firms_frp_total"].fillna(0.0).astype(float)
    df_full["firms_frp_mean"] = df_full["firms_frp_mean"].fillna(0.0).astype(float)
    df_full["firms_confidence_mean"] = df_full["firms_confidence_mean"].fillna(0.0).astype(float)

    df_full.to_csv(CACHE_FILE, index=False)
    print(f"[FIRMS] Saved clean daily active fire data to {CACHE_FILE}")
    print(f"[FIRMS] Peak fire day: {df_full.loc[df_full['firms_frp_total'].idxmax()]['Date'].strftime('%Y-%m-%d')} with {df_full['firms_fire_count'].max()} fire pixels, FRP = {df_full['firms_frp_total'].max():.1f} MW")
    return df_full


if __name__ == "__main__":
    download_all_firms_data()
