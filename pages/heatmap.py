"""
pages/heatmap.py
================
"Heatmaps" page — four modes, selected via the "Mode" segmented control:

  - season     : precomputed 4-season grid (year + vessel type), using the
                 precomputed files (no country filter, same as before).
  - csv        : heatmap built from an already-downloaded CSV (Data page),
                 imported from data/gfw_downloads.
  - afe_single : Apparent Fishing Effort (AFE) heatmap for a SINGLE vessel,
                 looked up by name / MMSI / IMO (same search as
                 pages/ais_gap.py), over a chosen date range.
  - afe_bulk   : AFE heatmap for MULTIPLE vessels, filtered by flag(s)
                 and/or vessel type(s), over a chosen date range.

Both AFE modes call the GFW API directly (like pages/data.py and
pages/ais_gap.py) -- no intermediate CSV, the heatmap is built straight
from the in-memory DataFrame.

Key fixes vs the previous version:
  - Nothing heavy loads automatically when the page opens. Loading the
    4-season grid (or any other mode) now only happens when the user
    clicks its "Show" button. This means switching modes is instant and
    never has to wait for a previous (possibly slow) heatmap to finish
    loading first.
  - The list of already-downloaded CSVs is no longer read from disk while
    building `layout()` (that used to block rendering of the whole page
    until the disk read finished). It's now loaded by a callback that
    fires after the page has rendered, so the controls are usable right
    away.
  - The mode selector is a real horizontal button group (segmented
    control) instead of a list of radios, so it can never end up stacked
    vertically.
  - The main callback checks that the active mode actually matches the
    button that was clicked before acting, so a click in one panel (e.g.
    "AFE — single vessel") can never end up producing the result/state of
    another panel (e.g. "Import CSV").
  - The inline deck.gl map now gets more vertical space, has a small
    gradient legend, and (for the CSV / AFE modes, where there is a single
    underlying point set) an "Open full map" button that opens a
    standalone, full-page Leaflet map in a new tab: colored effort grid
    cells, a proper legend, and a base-layer switcher (Esri Light Gray /
    Esri World Imagery), similar to the reference map the user shared.
    This needs the `folium` and `branca` packages (see _build_full_map_html
    below); if they aren't installed the button still appears but shows a
    clear error instead of failing silently.
"""

import asyncio
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl, GFW_DOWNLOAD_DIR
from config import YEARS, VESSEL_TYPES, SEASON_ORDER, ROOT, FLAG_NAMES
from loader import load_heatmap
from gfw import (list_downloaded_csvs, load_csv, get_gfw_client,
                 bulk_load_afe_dataframe, GFW_VESSEL_TYPES, COUNTRY_FLAGS)
from api_key import get_api_key
from pages import ais_gap as page_ais_gap

HEATMAP_COLUMNS = ["lat", "lon"]

# Where standalone full-page maps get written. Dash automatically serves
# anything under an "assets" folder next to the app at "/assets/...", so no
# extra Flask route is needed to open these in a new tab.
FULL_MAP_DIR = ROOT / "assets" / "generated_heatmaps"
FULL_MAP_URL_PREFIX = "/assets/generated_heatmaps"
GRID_DEG = 0.05  # size (in degrees) of one effort grid cell on the full map

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

MODES = [
    {"value": "season", "label": "4 seasons"},
    {"value": "csv", "label": "Import CSV"},
    {"value": "afe_single", "label": "AFE — single vessel"},
    {"value": "afe_bulk", "label": "AFE — multiple vessels"},
]

PANEL_ROW_STYLE = {"display": "flex", "alignItems": "flex-end", "gap": "0.9rem",
                    "flexWrap": "wrap", "padding": "0.9rem 1rem",
                    "background": PANEL, "border": f"1px solid {BDR}",
                    "borderRadius": "10px", "marginTop": "0.8rem"}
PANEL_ROW_HIDDEN = {"display": "none"}

FIELD_STYLE = {"display": "flex", "flexDirection": "column", "gap": "0.25rem"}

PLACEHOLDER_STYLE = {"color": DIM, "padding": "2rem", "fontSize": "0.85rem"}


def _mode_button_style(active):
    base = {
        "padding": "0.5rem 1rem",
        "fontSize": "0.78rem",
        "fontWeight": "600",
        "border": f"1px solid {BDR}",
        "cursor": "pointer",
        "background": PANEL,
        "color": SOFT,
        "flex": "0 0 auto",
        "whiteSpace": "nowrap",
    }
    if active:
        base["background"] = f"linear-gradient(135deg,{ACC},#0d4a7a)"
        base["color"] = "white"
        base["borderColor"] = ACC
    return base


def _mode_selector(active_value):
    """Horizontal segmented control, exactly one active button at a time."""
    buttons = []
    for i, m in enumerate(MODES):
        style = dict(_mode_button_style(m["value"] == active_value))
        if i == 0:
            style["borderRadius"] = "8px 0 0 8px"
        elif i == len(MODES) - 1:
            style["borderRadius"] = "0 8px 8px 0"
        else:
            style["borderRadius"] = "0"
        buttons.append(
            html.Button(m["label"], id={"type": "hm-mode-btn", "mode": m["value"]},
                        n_clicks=0, style=style)
        )
    return html.Div(buttons, style={"display": "flex", "flexDirection": "row",
                                     "flexWrap": "wrap"})


def layout():
    return html.Div([
        dcc.Store(id="hm-mode", data="season"),
        dcc.Store(id="hm-store-csv-df", data=None),
        dcc.Store(id="hmafe-search-store", data=None),

        html.Div([
            # ── Mode selector (horizontal buttons) ──
            html.Div([
                lbl("Mode"),
                html.Div(id="hm-mode-selector-wrap", children=_mode_selector("season")),
            ], style={"marginBottom": "0.4rem"}),

            # ── Panel: 4-season grid ──
            html.Div(id="hm-panel-season", style=PANEL_ROW_STYLE, children=[
                html.Div([lbl("Year"),
                    dcc.Dropdown(id="hm-year", value=YEARS[-1], clearable=False,
                        options=[{"label": str(y), "value": y} for y in YEARS],
                        style={"width": "100px", "color": "#000"})],
                    style=FIELD_STYLE),

                html.Div([lbl("Vessel type"),
                    dcc.Dropdown(id="hm-vtype", value=[], multi=True,
                        placeholder="All...",
                        options=[{"label": t.capitalize(), "value": t} for t in VESSEL_TYPES],
                        style={"width": "220px", "color": "#000"})],
                    style=FIELD_STYLE),

                html.Div([lbl(" "),
                    html.Button("Show", id="hm-btn-show", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem",
                               "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                               "color": "white", "border": "none",
                               "borderRadius": "6px", "cursor": "pointer",
                               "fontWeight": "600"})]),
            ]),

            # ── Panel: CSV import ──
            html.Div(id="hm-panel-csv", style=PANEL_ROW_HIDDEN, children=[
                html.Div([lbl("Import an already-downloaded CSV"),
                    dcc.Loading(type="dot", color=ACC, children=
                        dcc.Dropdown(id="hm-csv-selector",
                            options=[], value=None,
                            placeholder="Loading the list...",
                            style={"width": "320px", "color": "#000"}))],
                    style=FIELD_STYLE),

                html.Div([lbl(" "),
                    html.Button("Heatmap of this CSV", id="hm-btn-show-csv", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem", "background": PANEL,
                               "color": SOFT, "border": f"1px solid {BDR}",
                               "borderRadius": "6px", "cursor": "pointer"})]),
            ]),

            # ── Panel: AFE single vessel ──
            html.Div(id="hm-panel-afe-single", style=PANEL_ROW_HIDDEN, children=[
                html.Div([lbl("Vessel name / MMSI / IMO"),
                    dcc.Input(id="hmafe-query", type="text",
                        placeholder="Vessel name / MMSI / IMO", debounce=True,
                        style={"width": "220px", "padding": "0.4rem",
                               "borderRadius": "5px", "border": f"1px solid {BDR}",
                               "background": PANEL, "color": MAIN})],
                    style=FIELD_STYLE),

                html.Div([lbl(" "),
                    html.Button("Search", id="hmafe-btn-search", n_clicks=0,
                        style={"padding": "0.44rem 1rem",
                               "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "600"})]),

                html.Div([lbl("Vessel found"),
                    dcc.Loading(type="dot", color=ACC, children=
                        dcc.Dropdown(id="hmafe-vessel-selector", options=[], value=None,
                            placeholder="Search first...",
                            style={"width": "300px", "color": "#000"}))],
                    style=FIELD_STYLE),

                html.Div([lbl("Start date"),
                    dcc.DatePickerSingle(id="hmafe-start", date=date(YEARS[-1], 1, 1),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style=FIELD_STYLE),
                html.Div([lbl("End date"),
                    dcc.DatePickerSingle(id="hmafe-end", date=date(YEARS[-1], 12, 31),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style=FIELD_STYLE),

                html.Div([lbl(" "),
                    html.Button("Show AFE heatmap", id="hmafe-btn-show-single", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem",
                               "background": "linear-gradient(135deg,#d15400,#a03e00)",
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "600"})]),
            ]),

            # ── Panel: AFE multiple vessels ──
            html.Div(id="hm-panel-afe-bulk", style=PANEL_ROW_HIDDEN, children=[
                html.Div([lbl("Country / flag"),
                    dcc.Dropdown(id="hmafe-bulk-flags",
                        options=[{"label": "ALL countries", "value": "ALL"}] +
                                [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f}
                                 for f in COUNTRY_FLAGS],
                        value=["GRC"], multi=True, placeholder="Select countries...",
                        style={"width": "260px", "color": "#000"})],
                    style=FIELD_STYLE),

                html.Div([lbl("Vessel type (leave empty for ALL)"),
                    dcc.Dropdown(id="hmafe-bulk-vtypes",
                        options=[{"label": t.capitalize(), "value": t} for t in GFW_VESSEL_TYPES],
                        value=[], multi=True, placeholder="All types...",
                        style={"width": "220px", "color": "#000"})],
                    style=FIELD_STYLE),

                html.Div([lbl("Start date"),
                    dcc.DatePickerSingle(id="hmafe-bulk-start", date=date(YEARS[-1], 1, 1),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style=FIELD_STYLE),
                html.Div([lbl("End date"),
                    dcc.DatePickerSingle(id="hmafe-bulk-end", date=date(YEARS[-1], 12, 31),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style=FIELD_STYLE),

                html.Div([lbl(" "),
                    html.Button("Show AFE heatmap", id="hmafe-btn-show-bulk", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem",
                               "background": "linear-gradient(135deg,#d15400,#a03e00)",
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "600"})]),
            ]),

            html.Div([
                html.Div(id="hm-info",
                         style={"fontSize": "0.7rem", "color": DIM}),
                html.A("Open full map ↗", id="hm-open-map-link",
                       href="", target="_blank",
                       style={"display": "none", "fontSize": "0.72rem",
                              "color": ACC, "fontWeight": "600",
                              "textDecoration": "none",
                              "border": f"1px solid {ACC}",
                              "borderRadius": "5px", "padding": "0.15rem 0.6rem"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "0.8rem",
                      "marginTop": "0.5rem"}),
            html.Div(id="hm-csv-status",
                     style={"fontSize": "0.72rem", "color": SOFT,
                            "marginTop": "0.15rem"}),
            html.Div(id="hmafe-status",
                     style={"fontSize": "0.72rem", "color": SOFT,
                            "marginTop": "0.15rem"}),

        ], style={"padding": "0.6rem 1.2rem", "background": BG,
                   "borderBottom": f"1px solid {BDR}"}),

        # Heatmap grid (4 seasons OR AFE OR imported CSV) — starts empty,
        # nothing is fetched/computed until the user clicks a "Show" button.
        # Given more vertical room now that the header panel is tighter.
        html.Div(
            dcc.Loading(type="circle", color=ACC,
                children=html.Div(
                    id="hm-container",
                    children=html.P(
                        "Choose a mode above, set your filters, then click "
                        "the corresponding \"Show\" button to load a heatmap.",
                        style=PLACEHOLDER_STYLE),
                    style={"height": "calc(100vh - 52px - 96px)", "width": "100%"})),
            style={"flex": "1", "minHeight": 0}),

    ], style={"display": "flex", "flexDirection": "column",
              "height": "calc(100vh - 52px)", "background": BG})


# ── Deck building ─────────────────────────────────────────────────────────

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
    return json.loads(deck.to_json())


def _legend_bar():
    """Small fixed gradient legend overlay, echoing the reference full map."""
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


def _slugify(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "map"


def _build_full_map_html(df_hm, title, subtitle=""):
    """Builds a standalone, full-page Leaflet map (Esri base layers + a
    colored effort grid + a legend + a layer control) and saves it under
    the app's assets folder. Returns the URL to open it, or None plus an
    error message if it couldn't be built (e.g. folium isn't installed).
    """
    try:
        import folium
        import branca.colormap as bcm
    except ImportError:
        return None, ("The 'folium' and 'branca' packages are required for "
                       "the full map export. Install them with: "
                       "pip install folium branca")

    if df_hm is None or df_hm.empty:
        return None, "No positions to export."

    df = df_hm.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return None, "No positions to export."

    df["lat_bin"] = (df["lat"] // GRID_DEG) * GRID_DEG
    df["lon_bin"] = (df["lon"] // GRID_DEG) * GRID_DEG
    grid = df.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")

    vmin, vmax = float(grid["count"].min()), float(grid["count"].max())
    if vmin == vmax:
        vmax = vmin + 1

    # Fishing-effort counts are heavily skewed (a handful of hot cells vs.
    # a huge number of lightly-used ones). A plain linear color scale makes
    # almost every cell look "blue" and the map reads as a solid wall of
    # color. Instead, color breakpoints are placed at the data's own
    # quantiles (so color is spread evenly across how the values are
    # actually distributed) and cell opacity increases with effort, so low
    # -effort cells fade into the background and only real hot spots stand
    # out — closer to how GFW's own fishing-effort maps look.
    quantile_breaks = sorted(set(
        grid["count"].quantile(q) for q in (0, 0.5, 0.75, 0.9, 0.97, 1.0)
    ))
    if len(quantile_breaks) < 2:
        quantile_breaks = [vmin, vmax]
    colors = ["#0d1b6b", "#1f77e6", "#22c55e", "#eab308", "#f97316", "#ef4444"]
    colors = colors[-len(quantile_breaks):] if len(quantile_breaks) < len(colors) else colors
    colormap = bcm.LinearColormap(colors, index=quantile_breaks,
                                   vmin=quantile_breaks[0], vmax=quantile_breaks[-1])
    colormap.caption = "Fishing effort (points per cell)"

    def _opacity_for(count):
        rank = (count - vmin) / (vmax - vmin) if vmax > vmin else 1.0
        return round(0.25 + 0.6 * (rank ** 0.4), 2)  # sqrt-ish curve: low counts stay faint

    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()],
                    zoom_start=7, tiles=None)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri Light Gray", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri World Imagery", show=False).add_to(m)

    fg = folium.FeatureGroup(name=f"Effort: {title}", show=True)
    for _, row in grid.iterrows():
        count = row["count"]
        folium.Rectangle(
            bounds=[[row.lat_bin, row.lon_bin],
                    [row.lat_bin + GRID_DEG, row.lon_bin + GRID_DEG]],
            color=None, weight=0, fill=True,
            fill_color=colormap(count), fill_opacity=_opacity_for(count),
            popup=f"{int(count)} pts",
        ).add_to(fg)
    fg.add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    info_html = f"""
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
                background: white; padding: 8px 14px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-family: sans-serif;">
      <div style="font-weight: 700; font-size: 0.85rem;">{title}</div>
      {f'<div style="font-size: 0.75rem; color: #555;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    FULL_MAP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(title)}-{uuid.uuid4().hex[:8]}.html"
    m.save(str(FULL_MAP_DIR / filename))
    return f"{FULL_MAP_URL_PREFIX}/{filename}", None


def _season_deck(year, season_name, vtypes):
    if vtypes:
        frames = [load_heatmap(year, season_name, t, columns=HEATMAP_COLUMNS) for t in vtypes]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    else:
        df = load_heatmap(year, season_name, None, columns=HEATMAP_COLUMNS)
    n_pts = len(df)
    return _heatmap_deck_from_df(df), n_pts


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


# ── Synchronous GFW calls (thread + dedicated asyncio loop) ────────────────
# Same approach as pages/data.py::_do_download: the Dash callback is
# synchronous, so the coroutine runs in a thread with its own asyncio loop
# and we wait for the result (join). This keeps GFW calls off the main
# Flask/Dash request thread that other, lightweight callbacks (like the
# mode switch) also need, so switching modes stays responsive even while a
# heatmap is loading.

def _run_async_in_thread(make_coro):
    result = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(make_coro())
        except Exception as e:
            result["error"] = str(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()
    return result


def _open_map_link_props(href, error=None):
    """Style/href for the 'Open full map' link. Hidden when there's nothing
    to show; visible (and pointing at the generated file) on success."""
    if href:
        return href, {"display": "inline-block", "fontSize": "0.72rem",
                       "color": ACC, "fontWeight": "600", "textDecoration": "none",
                       "border": f"1px solid {ACC}", "borderRadius": "5px",
                       "padding": "0.15rem 0.6rem"}
    return "", {"display": "none"}


def _afe_single_heatmap(idx, entries, start, end):
    """Returns (children, hm-info, hm-csv-status, hmafe-status, map-href, map-style)."""
    if not start or not end:
        return dash.no_update, dash.no_update, dash.no_update, "Please choose a start and end date.", *_open_map_link_props(None)
    if idx is None or not entries:
        return dash.no_update, dash.no_update, dash.no_update, "Search and select a vessel first.", *_open_map_link_props(None)

    api_key = get_api_key()
    if not api_key:
        return dash.no_update, dash.no_update, dash.no_update, "No API key saved.", *_open_map_link_props(None)

    info = entries[int(idx)]
    vessel_ids = info.get("ids") or []
    if not vessel_ids:
        return dash.no_update, dash.no_update, dash.no_update, "This vessel has no usable vessel_id.", *_open_map_link_props(None)

    client = get_gfw_client(api_key)
    res = _run_async_in_thread(lambda: bulk_load_afe_dataframe(
        None, start[:10], end[:10], client, vessel_ids=vessel_ids))

    if res.get("error"):
        return dash.no_update, dash.no_update, dash.no_update, "GFW error: " + res["error"][:120], *_open_map_link_props(None)

    df = res.get("value")
    vessel_name = info.get("name") or "this vessel"
    if df is None or df.empty:
        return dash.no_update, dash.no_update, dash.no_update, f"No AFE data for {vessel_name} in this period.", *_open_map_link_props(None)
    if "lat" not in df.columns or "lon" not in df.columns:
        return dash.no_update, dash.no_update, dash.no_update, "AFE data has no lat/lon columns.", *_open_map_link_props(None)

    df_hm = df[["lat", "lon"]].dropna()
    n_pts = len(df_hm)
    if df_hm.empty:
        return dash.no_update, dash.no_update, dash.no_update, f"No positions found for {vessel_name}/period.", *_open_map_link_props(None)

    deck_json = _heatmap_deck_from_df(df_hm)
    title = f"AFE — {vessel_name} — {n_pts:,} pts"
    href, map_err = _build_full_map_html(df_hm, f"AFE — {vessel_name}", f"{n_pts:,} pts · {start[:10]} → {end[:10]}")
    status = f"{n_pts:,} positions loaded for {vessel_name}." + (f" ({map_err})" if map_err else "")
    return _deck_panel(deck_json), title, "", status, *_open_map_link_props(href)


def _afe_bulk_heatmap(flags, vtypes, start, end):
    """Returns (children, hm-info, hm-csv-status, hmafe-status, map-href, map-style)."""
    if not start or not end:
        return dash.no_update, dash.no_update, dash.no_update, "Please choose a start and end date.", *_open_map_link_props(None)

    api_key = get_api_key()
    if not api_key:
        return dash.no_update, dash.no_update, dash.no_update, "No API key saved.", *_open_map_link_props(None)

    resolved_flags = COUNTRY_FLAGS if (flags and "ALL" in flags) else (flags or None)

    client = get_gfw_client(api_key)
    res = _run_async_in_thread(lambda: bulk_load_afe_dataframe(
        resolved_flags, start[:10], end[:10], client, vessel_types=vtypes or None))

    if res.get("error"):
        return dash.no_update, dash.no_update, dash.no_update, "GFW error: " + res["error"][:120], *_open_map_link_props(None)

    df = res.get("value")
    if df is None or df.empty:
        return dash.no_update, dash.no_update, dash.no_update, "No AFE data for this filter/period.", *_open_map_link_props(None)
    if "lat" not in df.columns or "lon" not in df.columns:
        return dash.no_update, dash.no_update, dash.no_update, "AFE data has no lat/lon columns.", *_open_map_link_props(None)

    df_hm = df[["lat", "lon"]].dropna()
    n_pts = len(df_hm)
    if df_hm.empty:
        return dash.no_update, dash.no_update, dash.no_update, "No positions found for this filter/period.", *_open_map_link_props(None)

    n_vessels = df["vessel_id"].nunique() if "vessel_id" in df.columns else None
    vlabel = f", {n_vessels} vessel(s)" if n_vessels else ""

    deck_json = _heatmap_deck_from_df(df_hm)
    title = f"AFE — {n_pts:,} pts{vlabel}"
    href, map_err = _build_full_map_html(df_hm, "AFE — multiple vessels", f"{n_pts:,} pts{vlabel} · {start[:10]} → {end[:10]}")
    status = f"{n_pts:,} positions loaded{vlabel}." + (f" ({map_err})" if map_err else "")
    return _deck_panel(deck_json), title, "", status, *_open_map_link_props(href)


# ── CALLBACKS ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # Mode selector: buttons -> Store. Independent of everything else, no
    # disk/network access here, so it always responds instantly regardless
    # of what any other callback is doing.
    @app.callback(
        Output("hm-mode", "data"),
        Output("hm-mode-selector-wrap", "children"),
        Input({"type": "hm-mode-btn", "mode": dash.ALL}, "n_clicks"),
        State({"type": "hm-mode-btn", "mode": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def _set_mode(n_clicks_list, ids):
        trig = dash.callback_context.triggered_id
        if not trig or not isinstance(trig, dict):
            raise dash.exceptions.PreventUpdate
        new_mode = trig["mode"]
        return new_mode, _mode_selector(new_mode)

    @app.callback(
        Output("hm-panel-season", "style"),
        Output("hm-panel-csv", "style"),
        Output("hm-panel-afe-single", "style"),
        Output("hm-panel-afe-bulk", "style"),
        Input("hm-mode", "data"),
    )
    def _toggle_hm_mode(mode):
        return (
            PANEL_ROW_STYLE if mode == "season" else PANEL_ROW_HIDDEN,
            PANEL_ROW_STYLE if mode == "csv" else PANEL_ROW_HIDDEN,
            PANEL_ROW_STYLE if mode == "afe_single" else PANEL_ROW_HIDDEN,
            PANEL_ROW_STYLE if mode == "afe_bulk" else PANEL_ROW_HIDDEN,
        )

    # List of already-downloaded CSVs: fires once after the page has
    # rendered (not while building layout()), so it never blocks the
    # initial display of the page while it reads the disk.
    @app.callback(
        Output("hm-csv-selector", "options"),
        Output("hm-csv-selector", "placeholder"),
        Input("hm-mode", "data"),
        prevent_initial_call=False,
    )
    def _load_csv_list(mode):
        try:
            files = list_downloaded_csvs(ROOT / "data")
        except Exception as e:
            return [], f"Read error: {e}"
        opts = [{"label": f["filename"], "value": f["path"]} for f in files]
        placeholder = "Choose a CSV..." if opts else "No downloaded CSV found"
        return opts, placeholder

    @app.callback(
        Output("hmafe-vessel-selector", "options"),
        Output("hmafe-vessel-selector", "value"),
        Output("hmafe-search-store", "data"),
        Output("hmafe-status", "children"),
        Input("hmafe-btn-search", "n_clicks"),
        State("hmafe-query", "value"),
        prevent_initial_call=True,
    )
    def _search_afe_vessel(n, query):
        if not n:
            raise dash.exceptions.PreventUpdate
        api_key = get_api_key()
        if not api_key:
            return [], None, None, "No API key saved."
        if not query or not str(query).strip():
            return [], None, None, "Enter a name, MMSI or IMO first."
        try:
            df = page_ais_gap.do_search_vessel(str(query).strip(), api_key)
        except Exception as e:
            return [], None, None, "Search failed: " + str(e)[:70]

        entries = page_ais_gap._group_results_by_identity(df)
        if not entries:
            return [], None, None, "No vessel found."
        opts = [{"label": e["label"], "value": str(i)} for i, e in enumerate(entries)]
        return opts, None, entries, f"{len(entries)} vessel(s) found."

    @app.callback(
        Output("hm-container", "children"),
        Output("hm-info", "children"),
        Output("hm-csv-status", "children"),
        Output("hmafe-status", "children", allow_duplicate=True),
        Output("hm-open-map-link", "href"),
        Output("hm-open-map-link", "style"),
        Input("hm-btn-show", "n_clicks"),
        Input("hm-btn-show-csv", "n_clicks"),
        Input("hmafe-btn-show-single", "n_clicks"),
        Input("hmafe-btn-show-bulk", "n_clicks"),
        State("hm-mode", "data"),
        State("hm-year", "value"),
        State("hm-vtype", "value"),
        State("hm-csv-selector", "value"),
        State("hmafe-vessel-selector", "value"),
        State("hmafe-search-store", "data"),
        State("hmafe-start", "date"),
        State("hmafe-end", "date"),
        State("hmafe-bulk-flags", "value"),
        State("hmafe-bulk-vtypes", "value"),
        State("hmafe-bulk-start", "date"),
        State("hmafe-bulk-end", "date"),
        prevent_initial_call=True,
    )
    def update_heatmap(n1, n2, n3, n4, mode, year, vtypes, csv_path,
                        afe_idx, afe_entries, afe_start, afe_end,
                        bulk_flags, bulk_vtypes, bulk_start, bulk_end):
        trigger = dash.callback_context.triggered_id if dash.callback_context.triggered else None
        if trigger is None:
            raise dash.exceptions.PreventUpdate

        # Safety net: a click only counts if the currently displayed mode
        # actually matches the panel the button belongs to (avoids a
        # "ghost" button from another panel, or a late Dash render,
        # triggering the wrong result).
        expected_mode = {
            "hm-btn-show": "season",
            "hm-btn-show-csv": "csv",
            "hmafe-btn-show-single": "afe_single",
            "hmafe-btn-show-bulk": "afe_bulk",
        }.get(trigger)
        if expected_mode is not None and mode != expected_mode:
            raise dash.exceptions.PreventUpdate

        # Case: AFE heatmap — single vessel
        if trigger == "hmafe-btn-show-single":
            return _afe_single_heatmap(afe_idx, afe_entries, afe_start, afe_end)

        # Case: AFE heatmap — multiple vessels
        if trigger == "hmafe-btn-show-bulk":
            return _afe_bulk_heatmap(bulk_flags, bulk_vtypes, bulk_start, bulk_end)

        # Case: heatmap from an imported CSV
        if trigger == "hm-btn-show-csv":
            if not csv_path:
                return html.P("Choose a CSV first.", style=PLACEHOLDER_STYLE), "", "", "", *_open_map_link_props(None)
            try:
                df = load_csv(csv_path)
            except Exception as e:
                return html.P(f"Error: {e}", style={"color": "#ff6b6b", "padding": "2rem"}), "", f"Error: {e}", "", *_open_map_link_props(None)

            n_pts = len(df)
            df_hm = df[["lat", "lon"]].dropna() if not df.empty else pd.DataFrame({"lat": [37.5], "lon": [24.5]})
            deck_json = _heatmap_deck_from_df(df_hm)
            csv_name = csv_path.split("/")[-1].split("\\")[-1]
            href, map_err = _build_full_map_html(df_hm, f"Imported CSV — {csv_name}", f"{n_pts:,} pts")
            csv_status = f"Loaded: {n_pts:,} rows" + (f" ({map_err})" if map_err else "")
            return _deck_panel(deck_json), f"Imported CSV — {n_pts:,} pts", csv_status, "", *_open_map_link_props(href)

        # Case: 4-season precomputed grid (no single full map here: four
        # separate grids, one per season, are shown side by side instead)
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda s: (s,) + _season_deck(year, s, vtypes), SEASON_ORDER))

        cards = []
        for sname, deck_json, npts in results:
            title = f"{sname} {year} — {npts:,} pts" if npts else f"{sname} {year} — no data"
            cards.append(_panel(title, deck_json, key=sname))

        grid = html.Div(cards, style={"display": "flex", "flexWrap": "wrap",
                                       "height": "100%", "width": "100%",
                                       "alignContent": "stretch"})
        return grid, f"4 seasons · {year}", "", "", *_open_map_link_props(None)