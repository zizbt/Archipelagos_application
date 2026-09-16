"""
pages/map.py
============
"Map & Trajectories" page:
- Import a personal CSV -> show its trajectories on the map, with an
  optional gear_type filter.

v5: the "Export CSV" button is gone. In its place, "Open map ↗" builds
a standalone, print-ready Leaflet map (base-layer switcher, positions +
trajectories colored by vessel type, a categorical legend, a north
arrow and a scale bar) and opens it in a new tab -- something you can
drop straight into a report, the same idea as pages/heatmap.py's
"Open map" button.

v6: the zone polygons (Greece/Turkey/Italy/Malta EEZ + territorial
waters, WDPA, Fourni) are now redrawn on the exported map too, using
the same precomputed GeoJSON files as the inline deck.gl map
(`loader.load_geojson`), with a matching legend entry for each.
"""

import re
import uuid

import base64
import io

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table

from shared import (
    BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY,
    lbl, card, build_deck, ZONE_LAYERS, GFW_DOWNLOAD_DIR,
)
from config import VESSEL_TYPES, TYPE_COLORS, DEFAULT_COLOR, FLAG_NAMES, ZONES, ROOT
from gfw import GEAR_TYPES
from loader import load_geojson

# Reuse the same north-arrow/title helpers as pages/heatmap.py so both
# "Open map" buttons produce visually consistent, report-ready maps.
from pages.heatmap import north_arrow_element, title_box_element, base_tile_layers, layer_control_contrast_css

MAX_POINTS = 60_000    # max number of points shown as scatter (sampled beyond that)
MAX_PATHS_FAST = 500   # max number of vessels drawn as paths in "fast" mode (checkbox unticked)
MAX_PTS_PER_PATH = 80  # max number of points per path (decimation, keeps the overall shape)

# Same standalone-map output location as pages/heatmap.py.
FULL_MAP_DIR = ROOT / "assets" / "generated_heatmaps"
FULL_MAP_URL_PREFIX = "/assets/generated_heatmaps"

# Server-side cache for the currently filtered selection (used to build
# the exported map without re-sending the whole dataframe as JSON).
_LAST_FILTERED_DF = {"df": None}

# Server-side cache for the raw uploaded CSV, before any filter is
# applied -- filters only run when "Show map" is clicked.
_CSV_CACHE = {"df": None, "filename": None}


def _sidebar_section(title, children):
    return html.Div([
        html.H6(title, style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
        *children,
    ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
              "borderBottom": f"1px solid {BDR}"})


def _legend_swatch(color_rgba, label, shape="square"):
    """One legend row: a small color swatch (square for zones, round dot
    for vessel types) next to its label."""
    r, g, b = color_rgba[0], color_rgba[1], color_rgba[2]
    a = (color_rgba[3] / 255) if len(color_rgba) > 3 else 1
    swatch_style = {
        "width": "11px", "height": "11px", "marginRight": "0.5rem",
        "flexShrink": "0",
        "backgroundColor": f"rgba({r},{g},{b},{a})",
        "border": f"1px solid rgba({r},{g},{b},1)",
        "borderRadius": "2px" if shape == "square" else "50%",
    }
    return html.Div([
        html.Span(style=swatch_style),
        html.Span(label, style={"fontSize": "0.72rem", "color": SOFT}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.3rem"})


def _legend_subtitle(text):
    return html.P(text, style={"fontSize": "0.66rem", "color": DIM, "fontWeight": "600",
                                "textTransform": "uppercase", "letterSpacing": "0.04em",
                                "margin": "0.6rem 0 0.35rem 0"})


def _legend_section():
    """Legend for zones and vessel types, in the sidebar."""
    zone_rows = [_legend_swatch(z["line_color"], z["label"], shape="square")
                 for z in ZONES.values()]
    return _sidebar_section("Legend", [
        _legend_subtitle("Zones"),
        *zone_rows,
    ])


def layout():
    return html.Div([
        dcc.Store(id="map-store-filtered-df", data=None),
        dcc.Store(id="map-sidebar-open", data=True),

        # ── Sidebar: spans the FULL height of the side (map + table included) ──
        html.Div([

            _sidebar_section("Import a CSV", [
                dcc.Upload(
                    id="map-csv-upload",
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
                ),
                html.Div(id="map-csv-filename",
                         style={"fontSize": "0.72rem", "color": DIM,
                                "marginBottom": "0.6rem", "fontStyle": "italic"}),
                lbl("Vessel type filter (optional)"),
                html.P("Keep only certain vessel types (e.g. Fishing).",
                       style={"fontSize": "0.68rem", "color": DIM, "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="map-vessel-type",
                    options=[{"label": t.replace("_", " ").capitalize(), "value": t}
                             for t in VESSEL_TYPES],
                    value=[], multi=True, placeholder="All vessel types...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                lbl("Gear type filter (optional)"),
                html.P("Only applies to CSVs downloaded from GFW, which "
                       "include a gear_type column (e.g. keep only Trawlers).",
                       style={"fontSize": "0.68rem", "color": DIM, "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="map-gear",
                    options=[{"label": g.replace("_", " ").capitalize(), "value": g}
                             for g in GEAR_TYPES],
                    value=[], multi=True, placeholder="All gear types...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                dcc.Checklist(
                    id="map-show-all-paths",
                    options=[{"label": " Show all trajectories (can be slow on a large selection)",
                              "value": "all"}],
                    value=[],
                    style={"fontSize": "0.72rem", "color": SOFT, "marginBottom": "0.8rem"},
                ),
                html.Button("Show map", id="map-btn-show", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem 1rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none", "borderRadius": "6px",
                           "cursor": "pointer", "fontWeight": "600", "fontSize": "0.78rem",
                           "marginBottom": "0.8rem"}),
                html.Div(id="map-csv-status",
                         style={"fontSize": "0.72rem", "color": SOFT}),
            ]),

            _legend_section(),

            html.Div(id="map-stats",
                     children="Select a CSV file to display trajectories.",
                     style={"fontSize": "0.75rem", "color": SOFT}),

        ], id="map-sidebar", style={"width": "260px", "minWidth": "260px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto",
                   "flexShrink": "0"}),

        html.Button("‹", id="map-toggle-sidebar", n_clicks=0, title="Show/hide panel",
            style={"width": "22px", "minWidth": "22px", "border": "none",
                   "background": PANEL, "color": MAIN, "cursor": "pointer",
                   "fontSize": "1.1rem", "fontWeight": "700",
                   "borderRight": f"1px solid {BDR}", "flexShrink": "0"}),

        # ── Right column: map fills the space; "Open map" top-right ──
        html.Div([
            html.Div(
                html.A("Open map ↗", id="map-open-map-link", href="", target="_blank",
                    style={"display": "none", "border": "none",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                           "fontWeight": "600", "textDecoration": "none",
                           "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": f"1px solid {BDR}", "flexShrink": "0",
                       "display": "flex", "justifyContent": "flex-end",
                       "alignItems": "center", "gap": "0.6rem"},
            ),

            html.Div([
                dcc.Loading(
                    type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0, "position": "relative"}),

            html.Div(id="map-vessel-list", style={"display": "none"}),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _vessel_list_table(df):
    if df.empty or "vessel_id" not in df.columns:
        return html.P("No vessels in the selection.",
                       style={"color": DIM, "fontStyle": "italic"})
    cols = [c for c in ["vessel_id", "ship_name", "flag", "vessel_type"] if c in df.columns]
    summary = df[cols].drop_duplicates(subset="vessel_id").copy()
    if "flag" in summary.columns:
        summary["flag"] = summary["flag"].map(lambda f: f"{FLAG_NAMES.get(f, f)} ({f})")
    return html.Div([
        html.P(f"{len(summary):,} vessels in the selection",
               style={"color": SOFT, "fontSize": "0.78rem", "marginBottom": "0.4rem"}),
        dash_table.DataTable(
            data=summary.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
            page_size=8,
            export_format="csv",
            export_headers="display",
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": PANEL, "color": MAIN, "border": f"1px solid {BDR}",
                        "fontSize": "0.75rem", "padding": "4px 8px"},
            style_header={"backgroundColor": BG, "color": DIM, "fontWeight": "600"},
        ),
    ])


def _apply_gear_filter(df, gear):
    """Filter by gear_type if the column exists (precomputed trajectories
    don't have it; CSVs downloaded from GFW do)."""
    if df is None or df.empty or not gear:
        return df
    if "gear_type" not in df.columns:
        return df
    gear_upper = [g.upper() for g in gear]
    return df[df["gear_type"].astype(str).str.upper().isin(gear_upper)]


def _apply_vessel_type_filter(df, vessel_types):
    """Filter by vessel_type (e.g. keep only Fishing vessels)."""
    if df is None or df.empty or not vessel_types:
        return df
    if "vessel_type" not in df.columns:
        return df
    types_upper = [t.upper() for t in vessel_types]
    return df[df["vessel_type"].astype(str).str.upper().isin(types_upper)]


def _parse_uploaded_csv(contents, filename):
    """Decode a file selected via dcc.Upload (browser file picker /
    drag-and-drop) into a DataFrame. `contents` is a base64 data URI
    of the form 'data:text/csv;base64,....'."""
    if contents is None:
        return None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
    return pd.read_csv(io.BytesIO(decoded))


def _build_map(df, show_all_paths=False):
    layers = list(ZONE_LAYERS.values())
    note = None

    if df is not None and not df.empty and "lat" in df.columns and "lon" in df.columns:
        df_plot = df.copy()

        if "vessel_type" in df_plot.columns:
            vtype_upper = df_plot["vessel_type"].astype(str).str.upper()
            df_plot["color"] = vtype_upper.map(lambda t: TYPE_COLORS.get(t, DEFAULT_COLOR))
        else:
            df_plot["color"] = [DEFAULT_COLOR] * len(df_plot)

        # Vectorized tooltip (fast, no row-by-row .apply)
        ship = df_plot["ship_name"].astype(str) if "ship_name" in df_plot.columns else "?"
        flag_lbl = (df_plot["flag"].map(lambda f: FLAG_NAMES.get(f, f))
                    if "flag" in df_plot.columns else "?")
        vtype = df_plot["vessel_type"].astype(str) if "vessel_type" in df_plot.columns else "?"
        df_plot["tooltip"] = ship + " (" + flag_lbl.astype(str) + ") - " + vtype.astype(str)

        # Scatter: sample down if there are too many points (keeps the map smooth)
        scatter_df = df_plot
        if len(scatter_df) > MAX_POINTS:
            scatter_df = scatter_df.sample(MAX_POINTS, random_state=0)
            note = f"Sampled display: {MAX_POINTS:,} / {len(df_plot):,} points"

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=scatter_df,
            get_position=["lon", "lat"], get_fill_color="color",
            get_radius=400, radius_min_pixels=2, radius_max_pixels=8,
            pickable=True, auto_highlight=True,
        ))

        # Trajectories: one line per vessel. By default (checkbox unticked)
        # we cap at MAX_PATHS_FAST vessels to keep the map smooth -- tick
        # "Show all trajectories" to draw all of them regardless of count
        # (can be slow on a large selection).
        if "vessel_id" in df.columns and "date" in df.columns:
            vessel_ids = df_plot["vessel_id"].unique()
            if not show_all_paths and len(vessel_ids) > MAX_PATHS_FAST:
                vessel_ids_kept = vessel_ids[:MAX_PATHS_FAST]
                note = (note + " · " if note else "") + \
                       (f"Trajectories limited to {MAX_PATHS_FAST:,} vessels out of "
                        f"{len(vessel_ids):,} (tick \"Show all trajectories\" to see them all)")
                sub = df_plot[df_plot["vessel_id"].isin(vessel_ids_kept)]
            else:
                sub = df_plot

            sub = sub.sort_values(["vessel_id", "date"])
            paths = []
            for vid, g in sub.groupby("vessel_id", sort=False):
                coords = g[["lon", "lat"]].values.tolist()
                # Decimation: keep at most MAX_PTS_PER_PATH points per vessel
                # (overall shape preserved, payload sent to the browser much
                # lighter -- this is what was slowing down / freezing the map).
                if len(coords) > MAX_PTS_PER_PATH:
                    step = len(coords) / MAX_PTS_PER_PATH
                    coords = [coords[int(i * step)] for i in range(MAX_PTS_PER_PATH)]
                if len(coords) > 1:
                    paths.append({
                        "path": coords,
                        "color": g["color"].iloc[0],
                        "tooltip": g["tooltip"].iloc[0],
                    })
            if paths:
                layers.append(pdk.Layer(
                    "PathLayer", data=paths,
                    get_path="path", get_color="color",
                    get_width=2, width_min_pixels=1.5, pickable=True,
                ))

    deck = build_deck(layers)
    map_widget = dash_deck.DeckGL(data=deck, mapboxKey=MAPBOX_KEY, style={"width": "100%", "height": "100%"})
    return map_widget, note


def _slugify(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "map"


def _rgba_to_hex(rgba):
    r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgba_to_css(rgba, alpha_scale=255):
    r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
    a = (rgba[3] / alpha_scale) if len(rgba) > 3 else 1
    return f"rgba({r},{g},{b},{round(a, 2)})"


def _add_zone_layers(m):
    """Draws the same maritime zones as the inline deck.gl map (Greece/
    Turkey/Italy/Malta EEZ + territorial waters, WDPA, Fourni), using
    the precomputed GeoJSON files loaded via loader.load_geojson --
    same source of truth as shared.ZONE_LAYERS, so both maps agree.
    Returns the list of (label, line_color_rgba) actually drawn, for
    the legend.
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
            gj,
            name=zone["label"],
            style_function=lambda _f, _lc=line_css, _fc=fill_css: {
                "color": _lc, "weight": 1.5, "fillColor": _fc, "fillOpacity": 1,
            },
        ).add_to(m)
        drawn.append((zone["label"], zone["line_color"]))
    return drawn


def _build_full_trajectory_map_html(df, gear, show_all_paths, title):
    """Standalone, print-ready Leaflet map of the currently filtered
    positions + trajectories: base-layer switcher, points/lines colored
    by vessel type, a categorical legend, north arrow and scale bar.
    Returns (url, error).

    Zone polygons are NOT drawn here (their geometry isn't available in
    this module, only their legend color via `ZONES`) -- see module
    docstring.
    """
    try:
        import folium
    except ImportError:
        return None, ("The 'folium' package is required for the map export. "
                       "Install it with: pip install folium")

    if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return None, "No positions to export."

    df_plot = df.dropna(subset=["lat", "lon"]).copy()
    if df_plot.empty:
        return None, "No positions to export."

    if "vessel_type" in df_plot.columns:
        vtype_upper = df_plot["vessel_type"].astype(str).str.upper()
        df_plot["_vtype"] = vtype_upper
        df_plot["_hex"] = vtype_upper.map(lambda t: _rgba_to_hex(TYPE_COLORS.get(t, DEFAULT_COLOR)))
    else:
        df_plot["_vtype"] = "UNKNOWN"
        df_plot["_hex"] = _rgba_to_hex(DEFAULT_COLOR)

    scatter_df = df_plot
    if len(scatter_df) > MAX_POINTS:
        scatter_df = scatter_df.sample(MAX_POINTS, random_state=0)

    m = folium.Map(location=[df_plot["lat"].mean(), df_plot["lon"].mean()],
                    zoom_start=7, tiles=None, control_scale=True)
    base_tile_layers(m)

    zones_drawn = _add_zone_layers(m)

    points_fg = folium.FeatureGroup(name="Positions", show=True)
    for _, row in scatter_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=2.5,
            color=row["_hex"], fill=True, fill_color=row["_hex"],
            fill_opacity=0.85, weight=0,
        ).add_to(points_fg)
    points_fg.add_to(m)

    if "vessel_id" in df_plot.columns and "date" in df_plot.columns:
        vessel_ids = df_plot["vessel_id"].unique()
        if not show_all_paths and len(vessel_ids) > MAX_PATHS_FAST:
            vessel_ids = vessel_ids[:MAX_PATHS_FAST]
        sub = df_plot[df_plot["vessel_id"].isin(vessel_ids)].sort_values(["vessel_id", "date"])

        paths_fg = folium.FeatureGroup(name="Trajectories", show=True)
        for vid, g in sub.groupby("vessel_id", sort=False):
            coords = g[["lat", "lon"]].values.tolist()
            if len(coords) > MAX_PTS_PER_PATH:
                step = len(coords) / MAX_PTS_PER_PATH
                coords = [coords[int(i * step)] for i in range(MAX_PTS_PER_PATH)]
            if len(coords) > 1:
                folium.PolyLine(coords, color=g["_hex"].iloc[0], weight=2,
                                 opacity=0.85).add_to(paths_fg)
        paths_fg.add_to(m)

    # Categorical legend (vessel types actually present in the selection)
    present_types = sorted(df_plot["_vtype"].unique().tolist())
    rows_html = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="width:11px;height:11px;border-radius:50%;margin-right:6px;'
        f'background:{_rgba_to_hex(TYPE_COLORS.get(t, DEFAULT_COLOR))};"></span>'
        f'<span style="font-size:11px;">{t.title()}</span></div>'
        for t in present_types
    )
    zone_rows_html = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<span style="width:11px;height:11px;border-radius:2px;margin-right:6px;'
        f'background:{_rgba_to_hex(line_color)};"></span>'
        f'<span style="font-size:11px;">{label}</span></div>'
        for label, line_color in zones_drawn
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 16px; z-index: 9999;
                background: white; padding: 10px 12px; border-radius: 6px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.35); font-family: sans-serif;
                max-height: 40vh; overflow-y: auto;">
      <div style="font-weight:700; font-size:11px; text-transform:uppercase;
                  letter-spacing:0.04em; margin-bottom:6px;">Vessel type</div>
      {rows_html}
      {f'<div style="font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:0.04em; margin:10px 0 6px;">Zones</div>{zone_rows_html}' if zones_drawn else ''}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    n_v = df_plot["vessel_id"].nunique() if "vessel_id" in df_plot.columns else None
    subtitle = f"{len(df_plot):,} positions" + (f" · {n_v:,} vessels" if n_v else "")
    m.get_root().html.add_child(title_box_element(title, subtitle))
    m.get_root().html.add_child(north_arrow_element())

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().header.add_child(layer_control_contrast_css())

    FULL_MAP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(title)}-{uuid.uuid4().hex[:8]}.html"
    m.save(str(FULL_MAP_DIR / filename))
    return f"{FULL_MAP_URL_PREFIX}/{filename}", None


def _open_map_link_props(href):
    if href:
        return href, {"display": "inline-block", "border": "none",
                       "background": "linear-gradient(135deg,#0d6efd,#0d4a7a)",
                       "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                       "fontWeight": "600", "textDecoration": "none",
                       "padding": "0.3rem 1rem", "borderRadius": "5px"}
    return "", {"display": "none"}


def register_callbacks(app):

    @app.callback(
        Output("map-sidebar", "style"),
        Output("map-toggle-sidebar", "children"),
        Output("map-sidebar-open", "data"),
        Input("map-toggle-sidebar", "n_clicks"),
        State("map-sidebar-open", "data"),
        prevent_initial_call=True,
    )
    def _toggle_sidebar(n, is_open):
        now_open = not is_open
        if now_open:
            style = {"width": "260px", "minWidth": "260px", "padding": "1rem",
                     "background": BG, "borderRight": f"1px solid {BDR}",
                     "height": "calc(100vh - 52px)", "overflowY": "auto",
                     "flexShrink": "0"}
            arrow = "‹"
        else:
            style = {"width": "0px", "minWidth": "0px", "padding": "0",
                     "background": BG, "borderRight": f"1px solid {BDR}",
                     "height": "calc(100vh - 52px)", "overflow": "hidden",
                     "flexShrink": "0"}
            arrow = "›"
        return style, arrow, now_open

    # Parse the CSV as soon as it's dropped/browsed -- just caches it and
    # shows the filename/row count. The map itself is only (re)built once
    # "Show map" is clicked, below (so switching filters doesn't force a
    # rebuild until you're ready).
    @app.callback(
        Output("map-csv-filename", "children"),
        Output("map-csv-status", "children"),
        Output("map-container", "children"),
        Output("map-vessel-list", "children"),
        Output("map-stats", "children"),
        Output("map-store-filtered-df", "data"),
        Output("map-open-map-link", "href"),
        Output("map-open-map-link", "style"),
        Input("map-csv-upload", "contents"),
        State("map-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv_uploaded(contents, filename):
        if not contents:
            raise dash.exceptions.PreventUpdate
        try:
            df = _parse_uploaded_csv(contents, filename)
        except Exception as e:
            _CSV_CACHE["df"] = None
            map_c, _ = _build_map(pd.DataFrame())
            return ((filename or ""), f"Error: {e}", map_c, _vessel_list_table(pd.DataFrame()),
                    "", None, *_open_map_link_props(None))

        _CSV_CACHE["df"] = df
        _CSV_CACHE["filename"] = filename
        n_v = df["vessel_id"].nunique() if "vessel_id" in df.columns else len(df)
        status = f"Loaded {len(df):,} rows · {n_v:,} vessel(s). Set your filters, then click \"Show map\"."
        return ((filename or ""), status, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update)

    # Build (or rebuild) the map -- only runs on "Show map".
    @app.callback(
        Output("map-container", "children", allow_duplicate=True),
        Output("map-vessel-list", "children", allow_duplicate=True),
        Output("map-stats", "children", allow_duplicate=True),
        Output("map-store-filtered-df", "data", allow_duplicate=True),
        Output("map-csv-status", "children", allow_duplicate=True),
        Output("map-open-map-link", "href", allow_duplicate=True),
        Output("map-open-map-link", "style", allow_duplicate=True),
        Input("map-btn-show", "n_clicks"),
        State("map-vessel-type", "value"),
        State("map-gear", "value"),
        State("map-show-all-paths", "value"),
        State("map-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def update_map(n, vessel_types, gear, show_all, filename):
        if not n:
            raise dash.exceptions.PreventUpdate
        show_all_paths = "all" in (show_all or [])

        df = _CSV_CACHE.get("df")
        if df is None:
            map_c, _ = _build_map(pd.DataFrame())
            return (map_c, _vessel_list_table(pd.DataFrame()), "", None,
                    "Import a CSV first.", *_open_map_link_props(None))

        df = _apply_vessel_type_filter(df, vessel_types)
        df = _apply_gear_filter(df, gear)
        map_c, note = _build_map(df, show_all_paths)
        n_v = df["vessel_id"].nunique() if "vessel_id" in df.columns else len(df)
        stats_children = [html.P(f"{len(df):,} positions · {n_v:,} vessels (imported CSV)",
                                  style={"color": SOFT, "fontSize": "0.78rem"})]
        if note:
            stats_children.append(html.P(note, style={"color": DIM, "fontSize": "0.7rem", "fontStyle": "italic"}))
        status = f"Showing: {len(df):,} rows (filtered)"
        _LAST_FILTERED_DF["df"] = df

        title = f"Trajectories — {filename or 'Imported CSV'}"
        href, map_err = (_build_full_trajectory_map_html(df, gear, show_all_paths, title)
                         if not df.empty else (None, None))
        if map_err:
            status += f" ({map_err})"

        return (map_c, _vessel_list_table(df), html.Div(stats_children),
                (True if not df.empty else None), status,
                *_open_map_link_props(href))