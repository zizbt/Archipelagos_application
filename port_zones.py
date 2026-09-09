"""
port_zones.py
=============
Reference port / harbour points (ferry terminals, marinas, harbours),
loaded once from data/gis/ITA_vessels.geojson (an OSM extract). Used by
the Encounters and Loitering pages to flag whether a detected event
happened "in port" or "at sea".
"""

import json
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

from config import GIS_DIR

PORTS_GEOJSON = GIS_DIR / "ITA_vessels.geojson"

# Distance (metres) below which a point is considered "in port".
PORT_RADIUS_M_DEFAULT = 1000


def _load_port_points():
    if not PORTS_GEOJSON.exists():
        return None
    with open(PORTS_GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)

    lons, lats = [], []
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if not geom or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        lons.append(lon)
        lats.append(lat)

    if not lons:
        return None

    gdf = gpd.GeoDataFrame(
        {"lon": lons, "lat": lats},
        geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326",
    ).to_crs("EPSG:32634")
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


print("Loading port points...")
_PORT_XY = _load_port_points()
_PORT_TREE = cKDTree(_PORT_XY) if _PORT_XY is not None else None
print(f"  OK {0 if _PORT_XY is None else len(_PORT_XY)} port points loaded")


def has_ports() -> bool:
    return _PORT_TREE is not None


def port_mask_from_xy(x, y, radius_m=PORT_RADIUS_M_DEFAULT):
    """
    x, y: array-likes of projected coordinates (EPSG:32634, metres) --
    e.g. the same projection already used in encounter.py / loitering.py.
    Returns a boolean numpy array: True if the point lies within
    radius_m of the nearest known port point.
    """
    x = np.asarray(x)
    if _PORT_TREE is None or len(x) == 0:
        return np.zeros(len(x), dtype=bool)
    xy = np.column_stack([x, np.asarray(y)])
    dist, _ = _PORT_TREE.query(xy, k=1)
    return dist <= radius_m


def port_mask_from_lonlat(lon, lat, radius_m=PORT_RADIUS_M_DEFAULT):
    """Same as port_mask_from_xy but takes WGS84 lon/lat arrays directly."""
    lon = np.asarray(lon)
    if _PORT_TREE is None or len(lon) == 0:
        return np.zeros(len(lon), dtype=bool)
    gdf = gpd.GeoDataFrame(
        {"lon": lon, "lat": np.asarray(lat)},
        geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326",
    ).to_crs("EPSG:32634")
    return port_mask_from_xy(gdf.geometry.x.values, gdf.geometry.y.values, radius_m=radius_m)
