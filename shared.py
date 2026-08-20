"""
shared.py
=========
Constantes visuelles + objets chargés une seule fois au démarrage
et réutilisés par toutes les pages (app.py, pages/*.py).
"""

import pydeck as pdk
from dash import html

from config import ZONES, ROOT, FLAG_NAMES
from loader import load_geojson, load_flags, load_stats, load_marine_protected_areas

# ── Couleurs / thème ─────────────────────────────────────────────────────────────

BG    = "#060f1a"
PANEL = "#0a1e33"
BDR   = "#132840"
DIM   = "#4a7fa5"
MAIN  = "#e0f0ff"
SOFT  = "#8ab4cc"
ACC   = "#1a6faf"

MAPBOX_KEY    = ""
AEGEAN_CENTER = {"lat": 37.5, "lon": 24.5}

GFW_DOWNLOAD_DIR = ROOT / "data" / "gfw_downloads"
GFW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

TAB_S = {"color": DIM,  "backgroundColor": BG,    "border": "none", "padding": "0.5rem 1rem"}
TAB_A = {"color": MAIN, "backgroundColor": PANEL, "borderTop": f"2px solid {ACC}", "border": "none", "padding": "0.5rem 1rem"}


def lbl(text):
    return html.P(text, style={"fontSize": "0.7rem", "fontWeight": "600", "color": DIM,
                                "textTransform": "uppercase", "letterSpacing": "0.06em",
                                "margin": "0 0 0.3rem 0"})


def card(children, mb="1rem"):
    return html.Div(children, style={"background": PANEL, "border": f"1px solid {BDR}",
                                      "borderRadius": "8px", "padding": "1rem",
                                      "marginBottom": mb})


def build_deck(layers):
    import json
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=AEGEAN_CENTER["lat"],
            longitude=AEGEAN_CENTER["lon"],
            zoom=6, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    )
    return json.loads(deck.to_json())


# ── Chargement au démarrage (une seule fois) ─────────────────────────────────────

print("Loading zones...")
ZONE_LAYERS = {}
for key, zone in ZONES.items():
    gj = load_geojson(key)
    if gj:
        if "fill_color" not in zone or "line_color" not in zone:
            raise KeyError(f"Zone '{key}' is missing 'fill_color' or 'line_color'. Zone dict: {zone}")
        ZONE_LAYERS[key] = pdk.Layer(
            "GeoJsonLayer", data=gj, stroked=True, filled=True,
            get_fill_color=zone["fill_color"], get_line_color=zone["line_color"],
            line_width_min_pixels=1, get_line_width=zone.get("line_width", 100),
            pickable=False,
        )
        print(f"  OK {zone['label']}")

RAW_FLAGS    = load_flags()
GLOBAL_STATS = load_stats()
MARINE_ZONES = load_marine_protected_areas()
FLAG_OPTIONS = [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f} for f in RAW_FLAGS]
print(f"  OK {len(RAW_FLAGS)} flags, {len(MARINE_ZONES)} marine zones")
