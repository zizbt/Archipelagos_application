"""
zone_filters.py
================
Classification spatiale des positions de navires par zone maritime :
  - EEZ (zone economique exclusive) par pays
  - Eaux territoriales (bande des 6 milles nautiques a l'interieur de l'EEZ,
    dossiers "<Pays>_6_nautic_miles_inside_EEZ")
  - Aires protegees WDPA / Natura 2000 (fichier WDPA.shp)
  - Eaux internationales (tout ce qui n'est dans aucune EEZ connue)

Conçu pour degrader proprement : si un dossier/fichier attendu est absent,
la zone correspondante est simplement ignoree (liste vide), sans planter
le reste de l'appli -- meme logique defensive que port_zones.py.

⚠️ A VERIFIER / AJUSTER CHEZ TOI :
Les chemins ci-dessous (GIS_ROOT, EEZ_DIR, WDPA_SHP) sont des hypotheses
basees sur les captures d'ecran de ton explorateur de fichiers. Adapte-les
si l'emplacement reel differe (ex: si "ZEE_and_territorial_water" et
"WDPA" ne sont pas directement a la racine du projet).
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# CHEMINS -- A AJUSTER SI BESOIN
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent  # racine du projet (a cote de app.py)

EEZ_ROOT = PROJECT_ROOT / "ZEE_and_territorial_water"
WDPA_SHP = PROJECT_ROOT / "WDPA" / "WDPA.shp"

# Mapping "nom de zone affiche" -> dossier ou fichier source.
# Complete cette liste si tu as d'autres pays/dossiers (ex: Malte, Chypre...).
EEZ_SOURCES = {
    "Greece EEZ":                    EEZ_ROOT / "Greece_EEZ",
    "Greece territorial waters (6nm)": EEZ_ROOT / "Greece_6_nautic_miles_inside_EEZ",
    "Turkey EEZ":                     EEZ_ROOT / "Turkey_EEZ",
    "Turkey territorial waters (6nm)": EEZ_ROOT / "Turkey_6_nautic_miles_inside_EEZ",
    "Italy EEZ":                      EEZ_ROOT / "italy_eez.geojson",
    "Italy national waters":          EEZ_ROOT / "ita_national_waters.geojson",
    "Malta FMZ waters":               EEZ_ROOT / "mlt_FMZ_waters.geojson",
    "Malta national waters":          EEZ_ROOT / "mlt_national_waters.geojson",
}

WGS84 = "EPSG:4326"


# ---------------------------------------------------------------------------
# CHARGEMENT (mis en cache au niveau module -- charge une seule fois)
# ---------------------------------------------------------------------------
def _load_any_geometry(path: Path):
    """
    Charge un dossier (union de tous les .geojson dedans) ou un fichier
    .geojson unique, reprojette en WGS84, et renvoie une geometrie unique
    (union de toutes les features). Renvoie None si rien n'a pu etre charge.
    """
    try:
        if path.is_dir():
            files = (list(path.glob("*.geojson")) + list(path.glob("*.json"))
                     + list(path.glob("*.shp")))
            if not files:
                return None
            gdfs = []
            for f in files:
                try:
                    gdfs.append(gpd.read_file(f))
                except Exception:
                    continue
            if not gdfs:
                return None
            gdf = pd.concat(gdfs, ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
        elif path.exists():
            gdf = gpd.read_file(path)
        else:
            return None

        if gdf.empty:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84)
        elif gdf.crs.to_string() != WGS84:
            gdf = gdf.to_crs(WGS84)

        return unary_union(gdf.geometry.values)
    except Exception as e:
        print(f"  [zone_filters] Failed to load {path}: {e}")
        return None


print("Loading maritime zone geometries...")
_ZONE_GEOMETRIES = {}
for _name, _path in EEZ_SOURCES.items():
    _geom = _load_any_geometry(_path)
    if _geom is not None:
        _ZONE_GEOMETRIES[_name] = _geom
        print(f"  OK  {_name}  (from {_path.name})")
    else:
        print(f"  --  {_name}: not found ({_path}) -- filter option will have no effect")

# Union de toutes les EEZ connues, pour deduire les "eaux internationales"
# (tout point hors de n'importe quelle EEZ repertoriee ci-dessus).
_ALL_EEZ_NAMES = [n for n in _ZONE_GEOMETRIES if "EEZ" in n or "waters" in n.lower()]
_ALL_EEZ_UNION = (unary_union([_ZONE_GEOMETRIES[n] for n in _ALL_EEZ_NAMES])
                   if _ALL_EEZ_NAMES else None)

# WDPA / Natura 2000 (aires protegees) -- charge separement, reste par pays
_WDPA_GDF = None
try:
    if WDPA_SHP.exists():
        _WDPA_GDF = gpd.read_file(WDPA_SHP)
        if _WDPA_GDF.crs is None:
            _WDPA_GDF = _WDPA_GDF.set_crs("EPSG:3857")
        _WDPA_GDF = _WDPA_GDF.to_crs(WGS84)
        print(f"  OK  Protected areas (WDPA): {len(_WDPA_GDF)} features")
    else:
        print(f"  --  Protected areas (WDPA): not found ({WDPA_SHP})")
except Exception as e:
    print(f"  [zone_filters] Failed to load WDPA: {e}")

_WDPA_UNION_BY_COUNTRY = {}
if _WDPA_GDF is not None and "iso3" in _WDPA_GDF.columns:
    for _iso3, _group in _WDPA_GDF.groupby("iso3"):
        try:
            _WDPA_UNION_BY_COUNTRY[_iso3] = unary_union(_group.geometry.values)
        except Exception:
            continue
_WDPA_UNION_ALL = unary_union(_WDPA_GDF.geometry.values) if _WDPA_GDF is not None else None


# ---------------------------------------------------------------------------
# OPTIONS POUR L'UI (dcc.Dropdown)
# ---------------------------------------------------------------------------
def get_zone_options():
    """Renvoie la liste d'options disponibles pour le filtre de zone,
    seulement celles dont la geometrie a pu etre chargee."""
    options = [{"label": name, "value": name} for name in _ZONE_GEOMETRIES]
    if _WDPA_UNION_ALL is not None:
        options.append({"label": "Protected area (WDPA / Natura 2000, any country)",
                         "value": "__WDPA_ALL__"})
        for iso3 in sorted(_WDPA_UNION_BY_COUNTRY):
            options.append({"label": f"Protected area ({iso3})", "value": f"__WDPA_{iso3}__"})
    if _ALL_EEZ_UNION is not None:
        options.append({"label": "International waters (outside any known EEZ)",
                         "value": "__INTERNATIONAL__"})
    return options


def has_any_zone_data() -> bool:
    return bool(_ZONE_GEOMETRIES) or _WDPA_UNION_ALL is not None


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------
def zone_mask(lon, lat, zone_value: str):
    """
    Renvoie un np.ndarray booleen : True si le point (lon, lat) tombe dans
    la zone demandee. `zone_value` est une des valeurs renvoyees par
    get_zone_options() (nom de source EEZ, "__WDPA_ALL__", "__WDPA_<ISO3>__",
    ou "__INTERNATIONAL__"). Renvoie tout-False si la zone est inconnue ou
    pas chargee (degradation propre, comme port_zones.py).
    """
    gdf_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat), crs=WGS84)

    if zone_value == "__INTERNATIONAL__":
        if _ALL_EEZ_UNION is None:
            return pd.Series(False, index=gdf_points.index).values
        # vectorise (GEOS via shapely/pygeos sous le capot dans geopandas,
        # bien plus rapide qu'une boucle Python point par point)
        inside = gdf_points.geometry.within(_ALL_EEZ_UNION)
        return (~inside).values

    if zone_value == "__WDPA_ALL__":
        geom = _WDPA_UNION_ALL
    elif zone_value.startswith("__WDPA_"):
        iso3 = zone_value.replace("__WDPA_", "").replace("__", "")
        geom = _WDPA_UNION_BY_COUNTRY.get(iso3)
    else:
        geom = _ZONE_GEOMETRIES.get(zone_value)

    if geom is None:
        return pd.Series(False, index=gdf_points.index).values

    return gdf_points.geometry.within(geom).values