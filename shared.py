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

BG    = "#07111d"
PANEL = "#0d1d2d"
PANEL_2 = "#11283d"
BDR   = "#1c3550"
DIM   = "#5d87ab"
MAIN  = "#eef6ff"
SOFT  = "#9ab7cf"
ACC   = "#2c86d1"
GOOD  = "#61d39b"
WARN  = "#e0b070"
BAD   = "#e07070"
SHADOW = "0 18px 40px rgba(0,0,0,0.28)"

MAPBOX_KEY    = ""
AEGEAN_CENTER = {"lat": 37.5, "lon": 24.5}

GFW_DOWNLOAD_DIR = ROOT / "data" / "gfw_downloads"
GFW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

TAB_S = {
    "color": SOFT,
    "backgroundColor": "transparent",
    "border": f"1px solid transparent",
    "padding": "0.55rem 0.9rem",
    "borderRadius": "999px",
    "transition": "all 0.18s ease",
}
TAB_A = {
    "color": MAIN,
    "backgroundColor": PANEL_2,
    "border": f"1px solid {BDR}",
    "boxShadow": SHADOW,
    "padding": "0.55rem 0.9rem",
    "borderRadius": "999px",
    "transition": "all 0.18s ease",
}


def lbl(text):
    return html.P(text, style={
        "fontSize": "0.7rem",
        "fontWeight": "700",
        "color": SOFT,
        "textTransform": "uppercase",
        "letterSpacing": "0.08em",
        "margin": "0 0 0.35rem 0",
    })


def card(children, mb="1rem"):
    return html.Div(children, style={
        "background": f"linear-gradient(180deg,{PANEL} 0%,{PANEL_2} 100%)",
        "border": f"1px solid {BDR}",
        "borderRadius": "14px",
        "padding": "1rem",
        "marginBottom": mb,
        "boxShadow": SHADOW,
    })


def badge(text, color=SOFT, background="rgba(255,255,255,0.04)"):
    return html.Span(text, style={
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "0.35rem",
        "padding": "0.34rem 0.6rem",
        "borderRadius": "999px",
        "fontSize": "0.68rem",
        "fontWeight": "800",
        "letterSpacing": "0.02em",
        "color": color,
        "background": background,
        "border": f"1px solid {BDR}",
        "whiteSpace": "nowrap",
    })


def metric_card(value, label, color=MAIN):
    return html.Div([
        html.Div(value, style={"fontSize": "1.35rem", "fontWeight": "800", "color": color, "lineHeight": 1}),
        html.Div(label, style={"fontSize": "0.68rem", "color": SOFT, "marginTop": "0.2rem"}),
    ], style={
        "flex": "1",
        "minWidth": "120px",
        "textAlign": "center",
        "padding": "0.85rem 0.75rem",
        "border": f"1px solid {BDR}",
        "borderRadius": "12px",
        "background": "rgba(255,255,255,0.02)",
    })


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
