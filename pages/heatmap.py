"""
pages/heatmap.py
================
"Heatmaps" page. Layout is now a LEFT SIDEBAR (same pattern as
pages/map.py) instead of the old horizontal row of buttons/panels.

Sidebar section:
  - "Import a CSV" -> a single upload zone, a vessel filter, a vessel
    type filter, a gear-type filter (GFW CSVs only), and ONE checkbox
    ("Split into 4 seasons"). Unchecked -> a single live heatmap built
    from the imported CSV. Checked -> the same CSV is split into
    Winter / Spring / Summer / Fall (using its own "date" column) and
    shown as a 4-panel grid. There is NO precomputed data involved at
    any point -- everything is built live from the file the user drops.

The heatmap can be opened as a standalone, print-ready map in a new tab
("Open map" button): base-layer switcher, colored effort grid, legend,
zone overlays, north arrow and a scale bar, so it can be dropped
straight into a report.
"""

import base64
import io
import re
import uuid

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl, GFW_DOWNLOAD_DIR
from config import ROOT, ZONES
from loader import load_geojson

# Where standalone full-page maps get written. Dash automatically serves
# anything under an "assets" folder next to the app at "/assets/...", so no
# extra Flask route is needed to open these in a new tab.
FULL_MAP_DIR = ROOT / "assets" / "generated_heatmaps"
FULL_MAP_URL_PREFIX = "/assets/generated_heatmaps"
GRID_DEG = 0.05  # size (in degrees) of one effort grid cell on the full map

# Server-side cache for the currently uploaded CSV (avoids round-tripping
# a potentially large dataframe through the browser just to filter it).
_CSV_CACHE = {"df": None, "filename": None}


# Meteorological seasons, derived from the month of each row's "date"
# column -- no precomputed per-season files involved anywhere.
SEASON_MONTHS = {
    "Winter": {12, 1, 2},
    "Spring": {3, 4, 5},
    "Summer": {6, 7, 8},
    "Fall": {9, 10, 11},
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]

PLACEHOLDER_STYLE = {"color": DIM, "padding": "2rem", "fontSize": "0.85rem"}

SIDEBAR_OPEN_STYLE = {"width": "280px", "minWidth": "280px", "padding": "1rem",
                       "background": BG, "borderRight": f"1px solid {BDR}",
                       "height": "calc(100vh - 52px)", "overflowY": "auto",
                       "flexShrink": "0"}
SIDEBAR_CLOSED_STYLE = {"width": "0px", "minWidth": "0px", "padding": "0",
                         "background": BG, "borderRight": f"1px solid {BDR}",
                         "height": "calc(100vh - 52px)", "overflow": "hidden",
                         "flexShrink": "0"}

FIELD_STYLE = {"display": "flex", "flexDirection": "column", "gap": "0.25rem",
               "marginBottom": "0.6rem"}


def _sidebar_section(title, children):
    return html.Div([
        html.H6(title, style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
        *children,
    ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
              "borderBottom": f"1px solid {BDR}"})


def _checkbox(id_, label, value=False):
    return dcc.Checklist(
        id=id_,
        options=[{"label": f" {label}", "value": "on"}],
        value=(["on"] if value else []),
        style={"fontSize": "0.75rem", "color": SOFT, "marginBottom": "0.6rem"},
    )


def _upload_zone(id_):
    return dcc.Upload(
        id=id_,
        children=html.Div([
            "Drag a file here, or ",
            html.A("browse", style={"color": ACC, "textDecoration": "underline"}),
        ]),
        style={
            "width": "100%", "padding": "1rem 0.5rem",
            "textAlign": "center", "cursor": "pointer",
            "border": f"1px dashed {BDR}", "borderRadius": "6px",
            "color": SOFT, "fontSize": "0.75rem",
            "marginBottom": "0.5rem",
        },
        multiple=False,
    )


def _open_map_button():
    return html.A("Open map ↗", id="hm-open-map-link", href="", target="_blank",
                   style={"display": "none", "fontSize": "0.75rem", "fontWeight": "600",
                          "textAlign": "center", "textDecoration": "none",
                          "color": "white", "padding": "0.4rem 1rem",
                          "borderRadius": "6px",
                          "background": f"linear-gradient(135deg,{ACC},#0d4a7a)"})


def layout():
    return html.Div([

        # ── Sidebar (same pattern as pages/map.py) ──
        html.Div([

            _sidebar_section("Import a CSV", [
                _upload_zone("hm-csv-upload"),
                html.Div("No file selected", id="hm-csv-filename",
                         style={"fontSize": "0.72rem", "color": DIM,
                                "fontStyle": "italic", "marginBottom": "0.6rem"}),
                _checkbox("hm-season-check", "Split into 4 seasons (Winter/Spring/Summer/Fall)"),
                html.P("Uses the CSV's own \"date\" column. No precomputed data — "
                       "everything is rebuilt live from the file you import. No date "
                       "range to fill in either — it's already in the file you downloaded.",
                       style={"fontSize": "0.66rem", "color": DIM, "marginBottom": "0.6rem"}),

                html.Div([lbl("Vessel (from the CSV)"),
                    dcc.Dropdown(id="hm-csv-vessel-filter",
                        options=[{"label": "ALL vessels", "value": "ALL"}], value="ALL",
                        placeholder="Import a CSV first...",
                        style={"color": "#000"})],
                    style=FIELD_STYLE),

                html.Div([lbl("Vessel type"),
                    dcc.Dropdown(id="hm-csv-type-filter",
                        options=[], value=[], multi=True,
                        placeholder="All types...",
                        style={"color": "#000"})],
                    style=FIELD_STYLE),

                html.Div(id="hm-csv-gear-wrap", style={"display": "none"}, children=[
                    html.Div([lbl("Gear type"),
                        dcc.Dropdown(id="hm-csv-gear-filter",
                            options=[], value=[], multi=True,
                            placeholder="All gears...",
                            style={"color": "#000"})],
                        style=FIELD_STYLE),
                ]),

                html.Button("Show heatmap", id="hm-csv-btn-show", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem 1rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none", "borderRadius": "6px",
                           "cursor": "pointer", "fontWeight": "600", "fontSize": "0.78rem"}),
            ]),

            html.Div(id="hm-csv-status",
                     style={"fontSize": "0.72rem", "color": SOFT, "marginBottom": "0.3rem"}),

        ], id="hm-sidebar", style=SIDEBAR_OPEN_STYLE),

        dcc.Store(id="hm-sidebar-open", data=True),
        html.Button("‹", id="hm-toggle-sidebar", n_clicks=0, title="Show/hide panel",
            style={"width": "22px", "minWidth": "22px", "border": "none",
                   "background": PANEL, "color": MAIN, "cursor": "pointer",
                   "fontSize": "1.1rem", "fontWeight": "700",
                   "borderRight": f"1px solid {BDR}", "flexShrink": "0"}),

        # ── Right column: map + "Open map" button top-right ──
        html.Div([
            html.Div([
                html.Div(id="hm-info", style={"fontSize": "0.78rem", "color": SOFT}),
                _open_map_button(),
            ], style={"padding": "0.4rem 0.8rem", "background": BG,
                      "borderBottom": f"1px solid {BDR}", "flexShrink": "0",
                      "display": "flex", "justifyContent": "space-between",
                      "alignItems": "center"}),

            html.Div(
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(
                        id="hm-container",
                        children=html.P(
                            "Import a CSV in the panel, then load a heatmap.",
                            style=PLACEHOLDER_STYLE),
                        style={"height": "100%", "width": "100%"})),
                style={"flex": "1", "minHeight": 0, "position": "relative"}),

        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


# ── Deck building (inline preview map) ──────────────────────────────────────

def _heatmap_deck_from_df(df_hm):
    """Build a deck.gl HeatmapLayer from a lat/lon DataFrame."""
    if df_hm is None or df_hm.empty:
        df_hm = pd.DataFrame({"lat": [37.5], "lon": [24.5]})
    layer = pdk.Layer("HeatmapLayer", data=df_hm,
                       get_position=["lon", "lat"],
                       aggregation="SUM", radiusPixels=14,
                       intensity=2.2, threshold=0.015)
    deck = pdk.Deck(layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=37.5, longitude=24.5, zoom=6, pitch=0),
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json")
    import json
    return json.loads(deck.to_json())


def _legend_bar():
    gradient = "linear-gradient(90deg,#0000ff,#00ff00,#ffff00,#ff0000)"
    return html.Div([
        html.Div("FISHING EFFORT", style={"fontSize": "0.65rem", "fontWeight": "700",
                                           "color": MAIN, "marginBottom": "0.3rem",
                                           "letterSpacing": "0.03em"}),
        html.Div(style={"height": "8px", "width": "160px", "borderRadius": "4px",
                         "background": gradient}),
        html.Div([
            html.Span("Low", style={"fontSize": "0.6rem", "color": SOFT}),
            html.Span("High", style={"fontSize": "0.6rem", "color": SOFT}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "width": "160px", "marginTop": "0.15rem"}),
    ], style={"position": "absolute", "bottom": "14px", "left": "14px", "zIndex": 10,
              "background": "rgba(6,15,26,0.88)", "padding": "8px 10px",
              "borderRadius": "8px", "border": f"1px solid {BDR}"})


def _deck_panel(deck_json):
    return html.Div([
        dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY,
                          style={"width": "100%", "height": "100%"}),
        _legend_bar(),
    ], style={"width": "100%", "height": "100%", "position": "relative"})


def _panel(title_text, deck_json, key):
    return html.Div([
        html.Div(title_text, style={"position": "absolute", "top": "10px", "left": "10px",
                                     "zIndex": 10, "background": "rgba(6,15,26,0.88)",
                                     "color": MAIN, "padding": "4px 10px",
                                     "borderRadius": "6px", "fontSize": "0.76rem",
                                     "fontWeight": "600"}),
        dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY,
                          style={"width": "100%", "height": "100%"}),
    ], key=key, style={"position": "relative", "width": "50%", "height": "50%",
                        "border": f"1px solid {BDR}", "boxSizing": "border-box"})


def _slugify(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "map"


# ── Standalone, print-ready map (opened in a new tab) ───────────────────────
# Common building blocks (north arrow, scale bar, title box) shared with
# pages/map.py's own standalone-map builder so both "Open map" buttons
# produce maps that look and feel the same, close to something you could
# drop straight into a report.

def north_arrow_element():
    """A crisp, dependency-free compass-rose north arrow overlay (top-right)."""
    import folium
    html_str = """
    <div style="position: fixed; top: 90px; right: 16px; z-index: 9999;
                width: 52px; height: 52px; background: rgba(255,255,255,0.95);
                border-radius: 50%; box-shadow: 0 1px 6px rgba(0,0,0,0.4);
                display: flex; align-items: center; justify-content: center;
                user-select: none;">
      <svg width="40" height="40" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="47" fill="none" stroke="#888" stroke-width="1.5"/>
        <!-- main N-S-E-W star -->
        <polygon points="50,6 58,50 50,42 42,50" fill="#c0392b"/>
        <polygon points="50,94 58,50 50,58 42,50" fill="#333"/>
        <polygon points="6,50 50,42 42,50 50,58" fill="#333"/>
        <polygon points="94,50 50,42 58,50 50,58" fill="#333"/>
        <text x="50" y="20" text-anchor="middle" font-family="sans-serif"
              font-size="16" font-weight="700" fill="#111">N</text>
      </svg>
    </div>
    """
    return folium.Element(html_str)


def title_box_element(title, subtitle=""):
    import folium
    html_str = f"""
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
                background: white; padding: 8px 14px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-family: sans-serif;
                max-width: 60vw;">
      <div style="font-weight: 700; font-size: 0.9rem;">{title}</div>
      {f'<div style="font-size: 0.75rem; color: #555;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    return folium.Element(html_str)


def base_tile_layers(m):
    """Esri World Imagery (real satellite background) is the default,
    Light Gray is the lighter alternative -- toggle in the layer panel."""
    import folium
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri World Imagery", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Light Gray", show=False).add_to(m)


def layer_control_contrast_css():
    """The default Leaflet layer-control checkboxes/radios are nearly
    invisible until hovered (thin, low-contrast native browser styling).
    This forces a visible, high-contrast checkbox/radio style and dark
    label text so the panel is readable without hovering."""
    import folium
    return folium.Element("""
    <style>
      .leaflet-control-layers-list label { color: #111 !important; font-size: 12.5px; }
      .leaflet-control-layers-selector {
          width: 15px !important; height: 15px !important;
          accent-color: #2c86d1 !important;
          outline: 1px solid #667 !important;
          margin-right: 6px !important;
          vertical-align: middle !important;
      }
      .leaflet-control-layers { padding: 8px 10px !important; }
    </style>
    """)


def _rgba_to_hex(rgba):
    r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgba_to_css(rgba, alpha_scale=255):
    r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
    a = (rgba[3] / alpha_scale) if len(rgba) > 3 else 1
    return f"rgba({r},{g},{b},{round(a, 2)})"


def add_zone_layers(m):
    """Draws the same maritime zones as the inline deck.gl map (Greece/
    Turkey/Italy/Malta EEZ + territorial waters, WDPA, Fourni) on a
    standalone folium map, from the same precomputed GeoJSON files used
    by shared.ZONE_LAYERS. Returns [(label, line_color_rgba), ...] for
    the legend, for whichever zones actually had a GeoJSON on disk.
    """
    import folium
    drawn = []
    for key, zone in ZONES.items():
        gj = load_geojson(key)
        if not gj or "line_color" not in zone or "fill_color" not in zone:
            continue
        line_css = _rgba_to_css(zone["line_color"])
        fill_css = _rgba_to_css(zone["fill_color"])
        folium.GeoJson(
            gj, name=zone["label"], show=True,
            style_function=lambda _f, _lc=line_css, _fc=fill_css: {
                "color": _lc, "weight": 1.5, "fillColor": _fc, "fillOpacity": 1,
            },
        ).add_to(m)
        drawn.append((zone["label"], zone["line_color"]))
    return drawn


def zone_legend_rows_html(zones_drawn):
    return "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="width:11px;height:11px;border-radius:2px;margin-right:6px;'
        f'background:{_rgba_to_hex(line_color)};"></span>'
        f'<span style="font-size:11px;">{label}</span></div>'
        for label, line_color in zones_drawn
    )


def _build_full_map_html(df_hm, title, subtitle=""):
    """Builds a standalone, print-ready Leaflet map: real satellite
    background, a smooth heatmap layer (not a grid of squares), a
    gradient legend, zone overlays, north arrow and scale bar. Returns
    (url, error)."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        return None, ("The 'folium' package is required for the map "
                       "export. Install it with: pip install folium")

    if df_hm is None or df_hm.empty:
        return None, "No positions to export."

    df = df_hm.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return None, "No positions to export."

    df["lat_bin"] = (df["lat"] // GRID_DEG) * GRID_DEG + GRID_DEG / 2
    df["lon_bin"] = (df["lon"] // GRID_DEG) * GRID_DEG + GRID_DEG / 2
    grid = df.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")

    vmin, vmax = float(grid["count"].min()), float(grid["count"].max())
    if vmin == vmax:
        vmax = vmin + 1

    # Quantile-based color breaks: fishing-effort counts are heavily
    # skewed, so a linear scale would make almost everything look the
    # same color.
    quantile_breaks = sorted(set(
        grid["count"].quantile(q) for q in (0, 0.5, 0.75, 0.9, 0.97, 1.0)
    ))
    if len(quantile_breaks) < 2:
        quantile_breaks = [vmin, vmax]
    colors = ["#0d1b6b", "#1f77e6", "#22c55e", "#eab308", "#f97316", "#ef4444"]
    colors = colors[-len(quantile_breaks):] if len(quantile_breaks) < len(colors) else colors

    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()],
                    zoom_start=7, tiles=None, control_scale=True)

    base_tile_layers(m)
    zones_drawn = add_zone_layers(m)

    # Real heatmap layer (smooth gradient blobs), weighted by point
    # count per cell -- not a grid of flat-colored squares.
    heat_points = grid[["lat_bin", "lon_bin", "count"]].values.tolist()
    heat_gradient = {"0.0": "#0d1b6b", "0.35": "#1f77e6", "0.55": "#22c55e",
                      "0.75": "#eab308", "0.9": "#f97316", "1.0": "#ef4444"}
    fg = folium.FeatureGroup(name=f"Effort: {title}", show=True)
    HeatMap(heat_points, radius=20, blur=24, max_zoom=9,
            min_opacity=0.35, gradient=heat_gradient).add_to(fg)
    fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    effort_legend_html = f"""
    <div style="position: fixed; bottom: 24px; right: 16px; z-index: 9999;
                background: white; padding: 10px 12px; border-radius: 6px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.35); font-family: sans-serif;
                width: 180px;">
      <div style="font-weight:700; font-size:11px; text-transform:uppercase;
                  letter-spacing:0.04em; margin-bottom:6px;">
        Fishing effort<br>(points / cell)
      </div>
      <div style="height:10px; border-radius:4px;
                  background: linear-gradient(90deg,{",".join(colors)});"></div>
      <div style="display:flex; justify-content:space-between; margin-top:4px;">
        <span style="font-size:10px; color:#555;">Low</span>
        <span style="font-size:10px; color:#555;">High</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(effort_legend_html))

    if zones_drawn:
        zone_legend_html = f"""
        <div style="position: fixed; bottom: 24px; left: 16px; z-index: 9999;
                    background: white; padding: 10px 12px; border-radius: 6px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.35); font-family: sans-serif;
                    max-height: 40vh; overflow-y: auto;">
          <div style="font-weight:700; font-size:11px; text-transform:uppercase;
                      letter-spacing:0.04em; margin-bottom:6px;">Zones</div>
          {zone_legend_rows_html(zones_drawn)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(zone_legend_html))

    m.get_root().html.add_child(title_box_element(title, subtitle))
    m.get_root().html.add_child(north_arrow_element())
    m.get_root().header.add_child(layer_control_contrast_css())

    FULL_MAP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(title)}-{uuid.uuid4().hex[:8]}.html"
    m.save(str(FULL_MAP_DIR / filename))
    return f"{FULL_MAP_URL_PREFIX}/{filename}", None


def _open_map_link_props(href, error=None):
    if href:
        return href, {"display": "inline-block", "fontSize": "0.75rem", "fontWeight": "600",
                       "textAlign": "center", "textDecoration": "none", "color": "white",
                       "padding": "0.4rem 1rem", "borderRadius": "6px",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)"}
    return "", {"display": "none"}


def _parse_uploaded_csv(contents, filename):
    if contents is None:
        return None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
    return pd.read_csv(io.BytesIO(decoded))


def _build_vessel_options(df):
    """Build [{"label": ..., "value": ...}] for the "Vessel (from the CSV)"
    dropdown, grouped by vessel_id when available (the stable identity),
    with a label of the ship's name -- or, if it has none, its MMSI / IMO
    / call sign / vessel_id, in that order, so unnamed vessels still show
    up in the list instead of being silently dropped.
    Returns (options, filter_column) -- filter_column is what the "Show
    heatmap" callback should filter on to match these option values.
    """
    if "vessel_id" not in df.columns:
        if "ship_name" in df.columns:
            names = sorted(df["ship_name"].dropna().astype(str).unique())
            return [{"label": n, "value": n} for n in names], "ship_name"
        return [], None

    fallback_cols = [c for c in ("mmsi", "imo", "call_sign") if c in df.columns]
    keep_cols = ["vessel_id"] + (["ship_name"] if "ship_name" in df.columns else []) + fallback_cols
    uniq = df[keep_cols].drop_duplicates(subset="vessel_id")

    opts = []
    for _, row in uniq.iterrows():
        name = row.get("ship_name")
        if pd.notna(name) and str(name).strip():
            label = str(name).strip()
        else:
            label = None
            for c in fallback_cols:
                val = row.get(c)
                if pd.notna(val) and str(val).strip():
                    label = f"{str(val).strip()} ({c.upper()})"
                    break
            if label is None:
                label = f"Vessel {row['vessel_id']}"
        opts.append({"label": label, "value": str(row["vessel_id"])})
    opts.sort(key=lambda o: o["label"])
    return opts, "vessel_id"


def _assign_season(df, date_col="date"):
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    month = parsed.dt.month
    season = pd.Series(index=df.index, dtype="object")
    for name, months in SEASON_MONTHS.items():
        season[month.isin(months)] = name
    return season


# ── CALLBACKS ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("hm-sidebar", "style"),
        Output("hm-toggle-sidebar", "children"),
        Output("hm-sidebar-open", "data"),
        Input("hm-toggle-sidebar", "n_clicks"),
        State("hm-sidebar-open", "data"),
        prevent_initial_call=True,
    )
    def _toggle_sidebar(n, is_open):
        now_open = not is_open
        return (SIDEBAR_OPEN_STYLE if now_open else SIDEBAR_CLOSED_STYLE,
                "‹" if now_open else "›", now_open)

    # Parse the CSV as soon as it's dropped/browsed, but only to populate
    # the filter dropdowns (vessel / type) -- the heatmap itself is only
    # built once "Show heatmap" is clicked, below.
    @app.callback(
        Output("hm-csv-filename", "children"),
        Output("hm-csv-vessel-filter", "options"),
        Output("hm-csv-vessel-filter", "value"),
        Output("hm-csv-type-filter", "options"),
        Output("hm-csv-type-filter", "value"),
        Output("hm-container", "children", allow_duplicate=True),
        Output("hm-csv-status", "children"),
        Input("hm-csv-upload", "contents"),
        State("hm-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv_uploaded(contents, filename):
        if not contents:
            raise dash.exceptions.PreventUpdate
        try:
            df = _parse_uploaded_csv(contents, filename)
        except Exception as e:
            _CSV_CACHE["df"] = None
            return ((filename or ""), [{"label": "ALL vessels", "value": "ALL"}], "ALL",
                    [], [], html.P(f"Error: {e}", style={"color": "#ff6b6b", "padding": "2rem"}),
                    f"Error: {e}")

        _CSV_CACHE["df"] = df
        _CSV_CACHE["filename"] = filename

        vessel_opts = [{"label": "ALL vessels", "value": "ALL"}]
        vessel_opts_list, _vessel_filter_col = _build_vessel_options(df)
        vessel_opts += vessel_opts_list

        type_opts = []
        if "vessel_type" in df.columns:
            type_opts = [{"label": t.title(), "value": t}
                         for t in sorted(df["vessel_type"].dropna().astype(str).str.upper().unique())]

        placeholder = html.P(f"Loaded {len(df):,} rows from \"{filename}\". "
                              "Adjust filters if needed, then click \"Show heatmap\".",
                              style=PLACEHOLDER_STYLE)
        return ((filename or ""), vessel_opts, "ALL", type_opts, [], placeholder,
                f"Loaded: {len(df):,} rows")

    # Gear options depend on the vessel type(s) currently selected -- the
    # gear filter itself is only shown once a type is picked.
    @app.callback(
        Output("hm-csv-gear-wrap", "style"),
        Output("hm-csv-gear-filter", "options"),
        Output("hm-csv-gear-filter", "value"),
        Input("hm-csv-type-filter", "value"),
    )
    def _update_gear_options(selected_types):
        df = _CSV_CACHE.get("df")
        if df is None or "gear_type" not in df.columns or not selected_types:
            return {"display": "none"}, [], []
        sub = df[df["vessel_type"].astype(str).str.upper().isin(selected_types)]
        gear_opts = [{"label": g.replace("_", " ").title(), "value": g}
                     for g in sorted(sub["gear_type"].dropna().astype(str).str.upper().unique())]
        return {"display": "block"}, gear_opts, []

    # Build the heatmap (single view or 4-season grid), only on click.
    @app.callback(
        Output("hm-container", "children"),
        Output("hm-info", "children"),
        Output("hm-csv-status", "children", allow_duplicate=True),
        Output("hm-open-map-link", "href"),
        Output("hm-open-map-link", "style"),
        Input("hm-csv-btn-show", "n_clicks"),
        State("hm-season-check", "value"),
        State("hm-csv-vessel-filter", "value"),
        State("hm-csv-type-filter", "value"),
        State("hm-csv-gear-filter", "value"),
        prevent_initial_call=True,
    )
    def _show_csv_heatmap(n, season_value, vessel_filter, type_filter, gear_filter):
        if not n:
            raise dash.exceptions.PreventUpdate
        df = _CSV_CACHE.get("df")
        if df is None or df.empty:
            return (html.P("Import a CSV first.", style=PLACEHOLDER_STYLE), "", "",
                    *_open_map_link_props(None))

        df = df.copy()
        vessel_col = "vessel_id" if "vessel_id" in df.columns else (
            "ship_name" if "ship_name" in df.columns else None)
        if vessel_col and vessel_filter and vessel_filter != "ALL":
            df = df[df[vessel_col].astype(str) == str(vessel_filter)]
        if type_filter and "vessel_type" in df.columns:
            df = df[df["vessel_type"].astype(str).str.upper().isin(type_filter)]
        if gear_filter and "gear_type" in df.columns:
            df = df[df["gear_type"].astype(str).str.upper().isin(gear_filter)]

        filename = _CSV_CACHE.get("filename") or "CSV"
        split_seasons = "on" in (season_value or [])

        if df.empty:
            return (html.P("No rows match this filter.", style=PLACEHOLDER_STYLE), "",
                    "0 rows after filtering", *_open_map_link_props(None))

        if split_seasons:
            if "date" not in df.columns:
                return (html.P("This CSV has no \"date\" column, can't split it into seasons.",
                                style={"color": "#ff6b6b", "padding": "2rem"}), "",
                        "Error: missing \"date\" column", *_open_map_link_props(None))
            df = df.dropna(subset=["lat", "lon"]) if not df.empty else df
            df["_season"] = _assign_season(df)

            cards = []
            for sname in SEASON_ORDER:
                sub = df[df["_season"] == sname]
                df_hm = sub[["lat", "lon"]] if not sub.empty else pd.DataFrame()
                npts = len(df_hm)
                deck_json = _heatmap_deck_from_df(df_hm)
                title = f"{sname} — {npts:,} pts" if npts else f"{sname} — no data"
                cards.append(_panel(title, deck_json, key=sname))

            grid = html.Div(cards, style={"display": "flex", "flexWrap": "wrap",
                                           "height": "100%", "width": "100%",
                                           "alignContent": "stretch"})
            status = f"Showing: {len(df):,} rows (filtered)"
            # No single-point-set export in season mode (four separate
            # panels) -- hide the "Open map" button here.
            return (grid, f"4 Seasons — {len(df):,} pts total", status,
                    *_open_map_link_props(None))

        n_pts = len(df)
        df_hm = df[["lat", "lon"]].dropna() if not df.empty else pd.DataFrame({"lat": [37.5], "lon": [24.5]})
        deck_json = _heatmap_deck_from_df(df_hm)
        href, map_err = _build_full_map_html(df_hm, f"Imported CSV — {filename}", f"{n_pts:,} pts")
        csv_status = f"Showing: {n_pts:,} rows (filtered)" + (f" ({map_err})" if map_err else "")
        return (_deck_panel(deck_json), f"Imported CSV — {n_pts:,} pts", csv_status,
                *_open_map_link_props(href))