import json
from pathlib import Path
from shapely.geometry import shape, mapping

p = Path("data/gis/wdpa.geojson")
with open(p, encoding="utf-8") as f:
    gj = json.load(f)
feats = gj.get("features", [])

TARGET_NAME = "NISOS FOURNOI KAI NISIDES  THYMAINA ALATONISI, THYMAINAKI, STRONGYLO, PLAKA, MAKRONISI, MIKROS KAI MEGALOS ANTHROPOFAGOS, AGIOS MINAS KAI THALASSIA PERIOCHI"

match = None
for feat in feats:
    if feat.get("properties", {}).get("name") == TARGET_NAME:
        match = feat
        break

if not match:
    print("Feature non retrouvee par nom exact -- verifie l'orthographe.")
else:
    props = match["properties"]
    geom = shape(match["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    centroid = geom.centroid
    print("Nom:", props.get("name"))
    print("Designation:", props.get("desig_eng"))
    print("Surface (rep_area, km2):", props.get("rep_area"))
    print("Surface marine (rep_m_area, km2):", props.get("rep_m_area"))
    print("Centroid (lon, lat):", centroid.x, centroid.y)
    print("Bounding box (lon_min, lat_min, lon_max, lat_max):", minx, miny, maxx, maxy)

    out = {"type": "FeatureCollection", "features": [match]}
    out_path = Path("data/gis/fourni_protected.geojson")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\nSauvegarde dans: {out_path}")