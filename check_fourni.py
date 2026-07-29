import json
from pathlib import Path

p = Path("data/gis/wdpa.geojson")
with open(p, encoding="utf-8") as f:
    gj = json.load(f)
feats = gj.get("features", [])

# 1) Recherche par nom partiel (plus large que "fourni")
print("=== Recherche par nom (fourn / korseon / ikaria) ===")
keywords = ["fourn", "korseon", "ikaria"]
name_matches = []
for feat in feats:
    props = feat.get("properties", {})
    name = str(props.get("name", "")) + " " + str(props.get("name_eng", ""))
    if any(kw in name.lower() for kw in keywords):
        name_matches.append(feat)
        print(" ->", props.get("name"), "|", props.get("name_eng"), "|", props.get("desig_eng"))
print(f"Total: {len(name_matches)}\n")

# 2) Recherche par position (Fourni ~ lat 37.573, lon 26.281)
print("=== Recherche par position (bounding box autour de Fourni) ===")
try:
    from shapely.geometry import shape, Point
    target = Point(26.281, 37.573)  # (lon, lat)
    buffer_deg = 0.4  # ~40km de marge
    loc_matches = []
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            g = shape(geom)
            minx, miny, maxx, maxy = g.bounds
            if (minx - buffer_deg <= target.x <= maxx + buffer_deg and
                miny - buffer_deg <= target.y <= maxy + buffer_deg):
                loc_matches.append(feat)
        except Exception:
            continue
    print(f"Total zones proches de Fourni: {len(loc_matches)}")
    for feat in loc_matches[:20]:
        props = feat["properties"]
        print(" ->", props.get("name"), "|", props.get("name_eng"), "|", props.get("desig_eng"), "|", props.get("realm"))
except ImportError:
    print("shapely non installé -- lance 'pip install shapely' si besoin")