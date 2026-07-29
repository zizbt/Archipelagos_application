"""
preprocess.py
=============
Run ONCE to precompute everything the app needs.
The app then only reads files — no heavy queries at runtime.

Usage : python preprocess.py

Generates:
  data/precomputed/heatmaps/heatmap_{year}_{season}_{type}.parquet
  data/precomputed/trajectories/traj_{year}_{month:02d}_{type}.parquet
  data/precomputed/filters/flags.json
  data/precomputed/statistics/stats.json
  data/precomputed/statistics/type_year_season.parquet   (counts per type/year/season)
  data/precomputed/statistics/protected_areas.json        (vessels crossing WDPA zones)
  data/gis/{zone}.geojson
"""

import json
import time
import duckdb
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from config import (
    RAW_DATA, HEATMAP_DIR, TRAJECTORY_DIR, FILTER_DIR, STAT_DIR, GIS_DIR,
    YEARS, SEASONS, SEASON_ORDER, VESSEL_TYPES, ALL_TYPES, ZONES,
    MAX_HEATMAP_POINTS,
)

PARQUET_GLOB = str(RAW_DATA / "**" / "*.parquet")


def con():
    return duckdb.connect()


def log(msg):
    print(msg, flush=True)


def already_done(path):
    return Path(path).exists()


def duration(t0):
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ── 1. Maritime zones → GeoJSON ────────────────────────────────────────────────

def convert_zones():
    log("\n=== 1/6 Maritime zones -> GeoJSON ===")
    for key, zone in ZONES.items():
        out = GIS_DIR / f"{key}.geojson"
        if already_done(out):
            log(f"  skip  {key} already converted")
            continue
        t0 = time.time()
        log(f"  ...   {key}")
        try:
            gdf = gpd.read_file(zone["shp"])
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            gdf.to_file(str(out), driver="GeoJSON")
            log(f"  done  {key} ({duration(t0)}, {out.stat().st_size // 1024} KB)")
        except Exception as e:
            log(f"  FAIL  {key} : {e}")


# ── 2. Filters — flags ──────────────────────────────────────────────────────────

def compute_filters():
    log("\n=== 2/6 Filters (flags) ===")
    out = FILTER_DIR / "flags.json"
    if already_done(out):
        log("  skip  flags.json already computed")
        return
    t0 = time.time()
    sql = f"""
        SELECT flag, COUNT(*) as n
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE flag NOT IN ('UNKNOWN', '') AND flag IS NOT NULL
        GROUP BY flag ORDER BY n DESC LIMIT 150
    """
    try:
        df = con().execute(sql).df()
        flags = df["flag"].tolist()
        with open(out, "w") as f:
            json.dump(flags, f)
        log(f"  done  {len(flags)} flags ({duration(t0)})")
    except Exception as e:
        log(f"  FAIL  flags : {e}")


# ── 3. Heatmaps — year x season x type ─────────────────────────────────────────

def season_where(year, season_name):
    """WHERE clause for a given season.
    Winter = Dec of `year` + Jan/Feb of `year+1` (season spans two calendar years)."""
    if season_name == "Winter":
        return f"((year = {year} AND month = 12) OR (year = {year + 1} AND month IN (1,2)))"
    months = SEASONS[season_name]
    months_str = ",".join(str(m) for m in months)
    return f"(year = {year} AND month IN ({months_str}))"


def compute_heatmaps():
    log("\n=== 3/6 Heatmaps (year x season x type) ===")
    types_to_compute = [ALL_TYPES] + VESSEL_TYPES
    total = len(YEARS) * len(SEASON_ORDER) * len(types_to_compute)
    done = 0

    for year in YEARS:
        for season_name in SEASON_ORDER:
            for vtype in types_to_compute:
                fname = f"heatmap_{year}_{season_name}_{vtype}.parquet"
                out = HEATMAP_DIR / fname

                if already_done(out):
                    done += 1
                    continue

                conds = [season_where(year, season_name)]
                if vtype != ALL_TYPES:
                    conds.append(f"vessel_type = '{vtype}'")
                where = " AND ".join(conds)

                sql = f"""
                    SELECT lat, lon
                    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
                    WHERE {where}
                    USING SAMPLE {MAX_HEATMAP_POINTS} ROWS
                """
                t0 = time.time()
                try:
                    df = con().execute(sql).df()
                    df.to_parquet(str(out), index=False)
                    done += 1
                    log(f"  [{done}/{total}] {fname} -- {len(df):,} pts ({duration(t0)})")
                except Exception as e:
                    done += 1
                    log(f"  [{done}/{total}] FAIL {fname} : {e}")


# ── 4. Trajectories — year x month x type ──────────────────────────────────────

def compute_trajectories():
    log("\n=== 4/6 Trajectories (year x month x type) ===")
    types_to_compute = [ALL_TYPES] + VESSEL_TYPES
    total = len(YEARS) * 12 * len(types_to_compute)
    done = 0

    for year in YEARS:
        for month in range(1, 13):
            for vtype in types_to_compute:
                fname = f"traj_{year}_{month:02d}_{vtype}.parquet"
                out = TRAJECTORY_DIR / fname

                if already_done(out):
                    done += 1
                    continue

                conds = [f"year = {year}", f"month = {month}"]
                if vtype != ALL_TYPES:
                    conds.append(f"vessel_type = '{vtype}'")
                where = " AND ".join(conds)

                sql = f"""
                    SELECT vessel_id, vessel_type, flag, ship_name, lat, lon, date
                    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
                    WHERE {where}
                    ORDER BY vessel_id, date
                """
                t0 = time.time()
                try:
                    df = con().execute(sql).df()
                    df.to_parquet(str(out), index=False)
                    done += 1
                    log(f"  [{done}/{total}] {fname} -- {len(df):,} rows ({duration(t0)})")
                except Exception as e:
                    done += 1
                    log(f"  [{done}/{total}] FAIL {fname} : {e}")


# ── 5. Statistics — counts per type / year / season ────────────────────────────

def compute_type_year_season_stats():
    """
    Builds a table: year, season, vessel_type, n_vessels, n_points
    Used by the Statistics tab to compare distributions across years/seasons.
    """
    log("\n=== 5/6 Statistics (type x year x season) ===")
    out = STAT_DIR / "type_year_season.parquet"
    if already_done(out):
        log("  skip  type_year_season.parquet already computed")
        return

    rows = []
    t0 = time.time()
    for year in YEARS:
        for season_name in SEASON_ORDER:
            where = season_where(year, season_name)
            sql = f"""
                SELECT
                    vessel_type,
                    COUNT(DISTINCT vessel_id) as n_vessels,
                    COUNT(*) as n_points
                FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
                WHERE {where}
                GROUP BY vessel_type
            """
            try:
                df = con().execute(sql).df()
                df["year"] = year
                df["season"] = season_name
                rows.append(df)
                log(f"  done  {year} {season_name} -- {len(df)} types")
            except Exception as e:
                log(f"  FAIL  {year} {season_name} : {e}")

    if rows:
        result = pd.concat(rows, ignore_index=True)
        result.to_parquet(str(out), index=False)
        log(f"  done  type_year_season.parquet ({duration(t0)}, {len(result)} rows)")
    else:
        log("  FAIL  no data collected")


# ── 6. Protected areas — vessels crossing WDPA zones ───────────────────────────

def compute_protected_area_crossings():
    """
    Computes how many vessels (and which types) have at least one AIS position
    inside a WDPA protected area, per year. Uses a spatial join on a sampled
    subset of points per year for tractability (full dataset is too large for
    a pointwise spatial join in one pass).
    """
    log("\n=== 6/6 Protected areas (WDPA crossings) ===")
    out = STAT_DIR / "protected_areas.json"
    if already_done(out):
        log("  skip  protected_areas.json already computed")
        return

    wdpa_path = GIS_DIR / "wdpa.geojson"
    if not wdpa_path.exists():
        log("  FAIL  wdpa.geojson missing -- run zone conversion first")
        return

    t0 = time.time()
    try:
        wdpa = gpd.read_file(str(wdpa_path))
        # Dissolve into a single geometry for fast point-in-polygon tests
        wdpa_union = wdpa.union_all() if hasattr(wdpa, "union_all") else wdpa.unary_union
    except Exception as e:
        log(f"  FAIL  loading WDPA geometry : {e}")
        return

    results = {}
    SAMPLE_PER_YEAR = 500_000  # cap for tractable spatial join

    for year in YEARS:
        sql = f"""
            SELECT vessel_id, vessel_type, lat, lon
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
            WHERE year = {year}
            USING SAMPLE {SAMPLE_PER_YEAR} ROWS
        """
        try:
            df = con().execute(sql).df()
        except Exception as e:
            log(f"  FAIL  {year} query : {e}")
            continue

        if df.empty:
            results[str(year)] = {"total_vessels_in_zone": 0, "by_type": {}}
            continue

        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        )
        # Fast vectorized containment test against the dissolved WDPA geometry
        inside_mask = gdf.geometry.within(wdpa_union)
        inside = gdf[inside_mask]

        n_vessels = inside["vessel_id"].nunique()
        by_type = (
            inside.groupby("vessel_type")["vessel_id"]
            .nunique()
            .sort_values(ascending=False)
            .to_dict()
        )
        results[str(year)] = {
            "total_vessels_in_zone": int(n_vessels),
            "by_type": {k: int(v) for k, v in by_type.items()},
            "sample_size": len(df),
        }
        log(f"  done  {year} -- {n_vessels:,} vessels in protected areas "
            f"(sample {len(df):,} pts)")

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  done  protected_areas.json ({duration(t0)})")


# ── Global stats ────────────────────────────────────────────────────────────────

def compute_stats():
    log("\n=== Global stats ===")
    out = STAT_DIR / "stats.json"
    if already_done(out):
        log("  skip  stats.json already computed")
        return
    t0 = time.time()
    sql = f"""
        SELECT
            COUNT(*) as total_points,
            COUNT(DISTINCT vessel_id) as total_vessels,
            COUNT(DISTINCT flag) as total_flags,
            MIN(year) as year_min,
            MAX(year) as year_max
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    """
    try:
        row = con().execute(sql).df().iloc[0].to_dict()
        with open(out, "w") as f:
            json.dump({k: int(v) for k, v in row.items()}, f, indent=2)
        log(f"  done  {int(row['total_points']):,} points, "
            f"{int(row['total_vessels']):,} vessels ({duration(t0)})")
    except Exception as e:
        log(f"  FAIL  stats : {e}")


# ── Marine protected areas — zones + metadata ──────────────────────────────────

def compute_marine_protected_areas():
    """
    Filters WDPA to Marine + Coastal zones within the Aegean Sea bounding box,
    computes centroids, and saves a lightweight metadata file used by the
    Protected Areas page.

    Output: data/precomputed/statistics/marine_protected_areas.json
    {
      "zones": [
        {
          "id": <WDPAID>,
          "name": <NAME>,
          "iucn_cat": <IUCN_CAT>,
          "marine": <MARINE>,
          "area_km2": <GIS_AREA>,
          "centroid_lat": ...,
          "centroid_lon": ...,
          "bbox": [minx, miny, maxx, maxy]
        }, ...
      ]
    }
    """
    log("\n=== Marine protected areas metadata ===")
    out = STAT_DIR / "marine_protected_areas.json"
    if already_done(out):
        log("  skip  marine_protected_areas.json already computed")
        return

    wdpa_path = GIS_DIR / "wdpa.geojson"
    if not wdpa_path.exists():
        log("  FAIL  wdpa.geojson missing -- run convert_zones() first")
        return

    t0 = time.time()
    try:
        gdf = gpd.read_file(str(wdpa_path))
    except Exception as e:
        log(f"  FAIL  loading WDPA : {e}")
        return

    # Filter to marine + coastal zones within Aegean bounding box
    # Column names in this GeoJSON are lowercase
    # Marine zones: gis_m_area > 0 means the zone has a marine component
    marine_mask = gdf["gis_m_area"] > 0
    gdf_marine = gdf[marine_mask].copy()

    # Aegean bounding box filter using centroid
    gdf_marine["centroid"] = gdf_marine.geometry.centroid
    bbox_mask = (
        (gdf_marine["centroid"].x >= 19.5) & (gdf_marine["centroid"].x <= 30.5) &
        (gdf_marine["centroid"].y >= 34.0) & (gdf_marine["centroid"].y <= 41.5)
    )
    gdf_marine = gdf_marine[bbox_mask].copy()

    if gdf_marine.empty:
        log("  WARN  no marine/coastal zones found in Aegean bbox")
        with open(out, "w") as f:
            json.dump({"zones": []}, f)
        return

    zones = []
    for _, row in gdf_marine.iterrows():
        centroid = row.geometry.centroid
        bbox = list(row.geometry.bounds)
        zones.append({
            "id":           int(row.get("site_id", 0)),
            "name":         str(row.get("name_eng") or row.get("name") or "Unknown"),
            "iucn_cat":     str(row.get("iucn_cat", "?")),
            "marine":       "Marine" if float(row.get("gis_m_area", 0)) > 0 else "Coastal",
            "area_km2":     round(float(row.get("gis_area", 0)), 2),
            "marine_km2":   round(float(row.get("gis_m_area", 0)), 2),
            "centroid_lat": round(centroid.y, 5),
            "centroid_lon": round(centroid.x, 5),
            "bbox":         [round(v, 5) for v in bbox],
        })

    # Sort by area descending (largest zones first in the list)
    zones.sort(key=lambda z: z["area_km2"], reverse=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"zones": zones}, f, indent=2, ensure_ascii=False)

    log(f"  done  {len(zones)} marine/coastal zones saved ({duration(t0)})")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()
    log("==========================================")
    log("  Aegean Vessel Tracker -- Preprocessing")
    log("==========================================")
    log("Files already computed are skipped (resume supported).\n")

    convert_zones()
    compute_filters()
    compute_heatmaps()
    compute_trajectories()
    compute_type_year_season_stats()
    compute_protected_area_crossings()
    compute_stats()
    compute_marine_protected_areas()

    log(f"\nPreprocessing complete in {duration(t_total)}")
    log("You can now run: python app.py")
