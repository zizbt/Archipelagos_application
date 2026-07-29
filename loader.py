"""
loader.py
=========
Reads precomputed files only — zero DuckDB queries inside the running app.
All functions return pandas DataFrames or dicts ready to use.
"""

import json
import duckdb
import pandas as pd
from pathlib import Path
from config import (
    RAW_DATA, TRAJECTORY_DIR, HEATMAP_DIR, FILTER_DIR, STAT_DIR, GIS_DIR,
    ALL_TYPES, FLAG_NAMES,
)

PARQUET_GLOB = str(RAW_DATA / "**" / "*.parquet")


# ── Maritime zones ─────────────────────────────────────────────────────────────

def load_geojson(zone_key):
    """Reads a precomputed GeoJSON. Returns the dict or None."""
    path = GIS_DIR / f"{zone_key}.geojson"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Filters ────────────────────────────────────────────────────────────────────

def load_flags():
    """List of flags sorted by frequency."""
    path = FILTER_DIR / "flags.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_stats():
    """Global dataset statistics."""
    path = STAT_DIR / "stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_type_year_season_stats():
    """DataFrame: year, season, vessel_type, n_vessels, n_points."""
    path = STAT_DIR / "type_year_season.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["year", "season", "vessel_type", "n_vessels", "n_points"])
    return pd.read_parquet(path)


def load_protected_area_stats():
    """Dict keyed by year: {total_vessels_in_zone, by_type, sample_size}."""
    path = STAT_DIR / "protected_areas.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_marine_protected_areas():
    """
    Returns list of marine/coastal WDPA zones with metadata:
    id, name, iucn_cat, marine, area_km2, centroid_lat, centroid_lon, bbox
    """
    path = STAT_DIR / "marine_protected_areas.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("zones", [])


# ── Heatmaps ───────────────────────────────────────────────────────────────────

def load_heatmap(year, season, vessel_type=None):
    """
    Loads lat/lon points for a precomputed heatmap.
    vessel_type=None -> all types combined.
    """
    vtype = vessel_type if vessel_type else ALL_TYPES
    fname = f"heatmap_{year}_{season}_{vtype}.parquet"
    path = HEATMAP_DIR / fname
    if not path.exists():
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.read_parquet(path)


# ── Precomputed trajectories ────────────────────────────────────────────────────

def load_trajectories(year, month, vessel_type=None):
    """
    Loads precomputed trajectories for a year + month.
    vessel_type=None -> all types combined.
    """
    vtype = vessel_type if vessel_type else ALL_TYPES
    fname = f"traj_{year}_{month:02d}_{vtype}.parquet"
    path = TRAJECTORY_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_trajectories_filtered(year, month, vessel_type=None, flags=None):
    """
    Loads trajectories and applies a flag filter in memory.
    Much faster than DuckDB for this secondary operation.
    """
    df = load_trajectories(year, month, vessel_type)
    if df.empty:
        return df
    if flags and len(flags) > 0:
        df = df[df["flag"].isin(flags)]
    return df


def load_trajectories_date(year, month, day, vessel_type=None, flags=None):
    """
    Loads trajectories for a precise date (day within the month).
    Filter applied in memory from the precomputed month file.
    """
    df = load_trajectories(year, month, vessel_type)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.day == day]
    if flags and len(flags) > 0:
        df = df[df["flag"].isin(flags)]
    return df


def load_trajectories_range(start_date, end_date, vessel_types=None, flags=None):
    """
    Charge les trajectoires précalculées sur une plage de dates arbitraire
    (peut chevaucher plusieurs mois / plusieurs fichiers).
    vessel_types : liste de types ou None/[] -> tous les types combinés.
    flags        : liste de pavillons ou None -> pas de filtre pays.
    """
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    types_to_load = vessel_types if vessel_types else [None]

    frames = []
    for period in months:
        for vt in types_to_load:
            df = load_trajectories(period.year, period.month, vt)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    if flags:
        df = df[df["flag"].isin(flags)]

    return df


# ── Fallback DuckDB (only if precomputed file is missing) ──────────────────────

def query_fallback(year, month, day=None, vessel_type=None, flags=None, max_rows=60_000):
    """
    Direct DuckDB query — used only if the precomputed file is absent.
    Prints a warning in the terminal.
    """
    print(f"  WARNING: DuckDB fallback for year={year} month={month} -- run preprocess.py")
    conds = [f"year = {year}", f"month = {month}"]
    if day:
        conds.append(f"DAY(date) = {day}")
    if vessel_type and vessel_type != ALL_TYPES:
        conds.append(f"vessel_type = '{vessel_type}'")
    if flags:
        flags_str = ",".join(f"'{f}'" for f in flags)
        conds.append(f"flag IN ({flags_str})")
    where = " AND ".join(conds)
    sql = f"""
        SELECT vessel_id, vessel_type, flag, ship_name, lat, lon, date
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where}
        USING SAMPLE {max_rows} ROWS
        ORDER BY vessel_id, date
    """
    try:
        con = duckdb.connect()
        df = con.execute(sql).df()
        con.close()
        return df
    except Exception as e:
        print(f"  FAIL  fallback failed : {e}")
        return pd.DataFrame()
