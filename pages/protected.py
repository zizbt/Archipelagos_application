"""
pages/protected.py
===================
"Protected Area" page -- zoom on the Fourni island (SPA protected zone
under the EU Birds Directive, including Thymaina and Agios Minas + the
marine area, extracted from data/gis/wdpa.geojson).

For a given date range (defaults to the most recent year), detects
which vessels passed inside the zone's polygon and shows, per vessel:
name, flag, type, number of hours detected inside, first/last
detection. CSV export included.

Two data sources, like the other pages:
- Precomputed (3.5 years of trajectories): gives vessel_type but NOT
  gear_type (so no way to specifically isolate "Trawlers" here).
- Import a downloaded CSV: has the gear_type column -> lets you
  actually filter/spot trawlers.

The date range is not locked to a single calendar year (a selection
can freely span across two years, e.g. Dec 2024 -> Jan 2025), same as
on the Map page.
"""

import json
from datetime import date, timedelta

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table
from shapely.geometry import shape, Point
from shapely.prepared import prep

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl
from config import YEARS, VESSEL_TYPES, TYPE_COLORS, DEFAULT_COLOR, FLAG_NAMES, ROOT, FOURNI_CENTER
from loader import load_geojson, load_trajectories_range
from gfw import list_downloaded_csvs, load_csv, GEAR_TYPES

ZONE_KEY = "fourni_protected"
ZONE_FILL = [255, 215, 0, 60]
ZONE_LINE = [255, 215, 0, 230]

# Full range of dates the app has data for -- the start/end date pickers
# are bounded by this instead of a single calendar year, so a selection
# can freely cross a year boundary (e.g. Dec 2024 -> Jan 2025).
GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)


def _load_zone():
    """Load the Fourni protected area polygon (once)."""
    gj = load_geojson(ZONE_KEY)
    if not gj or not gj.get("features"):
        return None, None
    feat = gj["features"][0]
    poly = shape(feat["geometry"])
    return poly, gj


_ZONE_POLYGON, _ZONE_GEOJSON = _load_zone()


def _filter_points_in_zone(df):
    """Filter positions falling inside the zone's polygon. Fast bounding-box
    pre-filter first, then an exact test only on the already-close subset --
    avoids testing every single point in the full dataset one by one (slow)."""
    if df is None or df.empty or _ZONE_POLYGON is None:
        return pd.DataFrame()
    if "lat" not in df.columns or "lon" not in df.columns:
        return pd.DataFrame()

    minx, miny, maxx, maxy = _ZONE_POLYGON.bounds
    bbox_mask = (df["lon"] >= minx) & (df["lon"] <= maxx) & \
                (df["lat"] >= miny) & (df["lat"] <= maxy)
    sub = df[bbox_mask]
    if sub.empty:
        return sub

    prepared = prep(_ZONE_POLYGON)
    inside = [prepared.contains(Point(lon, lat)) for lon, lat in zip(sub["lon"], sub["lat"])]
    return sub[inside]


def _aggregate_by_vessel(df):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["vessel_id", "date"])

    agg_dict = {
        "n_positions": ("date", "count"),
        "first_seen": ("date", "min"),
        "last_seen": ("date", "max"),
    }
    for col in ["ship_name", "flag", "vessel_type", "gear_type"]:
        if col in df.columns:
            agg_dict[col] = (col, "first")

    result = df.groupby("vessel_id").agg(**agg_dict).reset_index()
    # HOURLY resolution -> 1 position = ~1h detected inside the zone
    result["hours_detected"] = result["n_positions"]
    result["first_seen"] = result["first_seen"].dt.strftime("%Y-%m-%d %H:%M")
    result["last_seen"] = result["last_seen"].dt.strftime("%Y-%m-%d %H:%M")
    if "flag" in result.columns:
        result["flag_label"] = result["flag"].map(lambda f: f"{FLAG_NAMES.get(f, f)} ({f})")
    return result.sort_values("hours_detected", ascending=False)


def layout():
    return html.Div([
        dcc.Store(id="prot-store-agg", data=None),
        dcc.Download(id="prot-download-csv"),

        # ── Sidebar ──────────────────────────────────────────────────────────
        html.Div([
            html.H6("Protected zone", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
            dcc.Dropdown(id="prot-zone", value="fourni",
                options=[{"label": "Fourni (Fournoi Korseon)", "value": "fourni"}],
                clearable=False, style={"color": "#000", "marginBottom": "1rem"}),

            html.Div([
                lbl("Jump to a year (optional)"),
                dcc.Dropdown(id="prot-year", value=None, clearable=True,
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    placeholder="Jump to a year...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                lbl("Start date"),
                dcc.DatePickerSingle(id="prot-start", date=date(YEARS[-1], 1, 1),
                    display_format="YYYY-MM-DD",
                    min_date_allowed=GLOBAL_MIN_DATE,
                    max_date_allowed=GLOBAL_MAX_DATE,
                    style={"marginBottom": "0.6rem"}),
                lbl("End date"),
                dcc.DatePickerSingle(id="prot-end", date=date(YEARS[-1], 12, 31),
                    display_format="YYYY-MM-DD",
                    min_date_allowed=GLOBAL_MIN_DATE,
                    max_date_allowed=GLOBAL_MAX_DATE,
                    style={"marginBottom": "0.6rem"}),
                html.P("The range can span across two years (e.g. Dec 2024 -> Jan 2025).",
                       style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                              "marginBottom": "0.6rem"}),
                html.Button("Analyze", id="prot-btn-run", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none",
                           "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600"}),
            ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
                       "borderBottom": f"1px solid {BDR}"}),

            html.Div([
                html.H6("Import a downloaded CSV", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
                html.P("Needed to filter by gear type (Trawlers): "
                       "precomputed trajectories don't have this column.",
                       style={"fontSize": "0.68rem", "color": DIM, "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="prot-csv-selector",
                    options=[{"label": f["filename"], "value": f["path"]}
                             for f in list_downloaded_csvs(ROOT / "data")],
                    value=None, placeholder="Choose a CSV...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                lbl("Gear type filter (optional)"),
                dcc.Dropdown(id="prot-gear-filter",
                    options=[{"label": g.replace("_", " ").capitalize(), "value": g} for g in GEAR_TYPES],
                    value=[], multi=True, placeholder="All gear types (e.g. Trawlers)...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                html.Button("Analyze this CSV", id="prot-btn-run-csv", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem",
                           "background": PANEL, "color": SOFT,
                           "border": f"1px solid {BDR}", "borderRadius": "6px",
                           "cursor": "pointer", "marginBottom": "0.4rem"}),
                html.Div(id="prot-csv-status", style={"fontSize": "0.72rem", "color": SOFT}),
            ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
                       "borderBottom": f"1px solid {BDR}"}),

            html.Div(id="prot-summary", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        # ── Map (Fourni zoom) + vessel table ────────────────────────────────
        html.Div([
            html.Div([
                dcc.Loading(
                    type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="prot-map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0}),

            html.Div(
                dcc.Loading(children=html.Div(id="prot-vessel-table")),
                style={"height": "260px", "flexShrink": "0", "overflowY": "auto",
                       "borderTop": f"1px solid {BDR}", "padding": "0.5rem 1rem", "background": BG},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _build_zone_map(df_inside):
    layers = []
    if _ZONE_GEOJSON:
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=_ZONE_GEOJSON, stroked=True, filled=True,
            get_fill_color=ZONE_FILL, get_line_color=ZONE_LINE,
            line_width_min_pixels=2, pickable=False,
        ))

    if df_inside is not None and not df_inside.empty:
        plot = df_inside.copy()
        if "vessel_type" in plot.columns:
            plot["color"] = plot["vessel_type"].astype(str).str.upper().map(
                lambda t: TYPE_COLORS.get(t, DEFAULT_COLOR))
        else:
            plot["color"] = [DEFAULT_COLOR] * len(plot)
        ship = plot["ship_name"].astype(str) if "ship_name" in plot.columns else "?"
        flag_lbl = plot["flag"].map(lambda f: FLAG_NAMES.get(f, f)) if "flag" in plot.columns else "?"
        vtype = plot["vessel_type"].astype(str) if "vessel_type" in plot.columns else "?"
        plot["tooltip"] = ship + " (" + flag_lbl.astype(str) + ") - " + vtype.astype(str)

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["lon", "lat"], get_fill_color="color",
            get_radius=150, radius_min_pixels=3, radius_max_pixels=9,
            pickable=True, auto_highlight=True,
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=FOURNI_CENTER["lat"], longitude=FOURNI_CENTER["lon"],
            zoom=10.5, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    )
    deck_json = json.loads(deck.to_json())
    return dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY, style={"width": "100%", "height": "100%"})


def _vessel_table(agg):
    if agg.empty:
        return html.P("No vessel detected in the zone for this period.",
                       style={"color": DIM, "fontStyle": "italic"})
    cols = ["vessel_id"]
    for c in ["ship_name", "flag_label", "vessel_type", "gear_type", "hours_detected", "first_seen", "last_seen"]:
        if c in agg.columns:
            cols.append(c)
    display_names = {
        "vessel_id": "Vessel Id", "ship_name": "Ship Name", "flag_label": "Flag",
        "vessel_type": "Vessel Type", "gear_type": "Gear Type",
        "hours_detected": "Hours Detected", "first_seen": "First Seen", "last_seen": "Last Seen",
    }
    return dash_table.DataTable(
        data=agg[cols].to_dict("records"),
        columns=[{"name": display_names.get(c, c), "id": c} for c in cols],
        page_size=10, export_format="csv", export_headers="display",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": PANEL, "color": MAIN, "border": f"1px solid {BDR}",
                    "fontSize": "0.75rem", "padding": "4px 8px"},
        style_header={"backgroundColor": BG, "color": DIM, "fontWeight": "600"},
    )


def _summary(agg, note=None):
    if agg.empty:
        return html.P("No data.", style={"color": DIM})
    n_vessels = len(agg)
    total_hours = int(agg["hours_detected"].sum())
    children = [
        html.P(f"{n_vessels:,} vessels detected in the zone", style={"color": SOFT, "margin": "0.2rem 0"}),
        html.P(f"{total_hours:,} cumulative hours detected", style={"color": SOFT, "margin": "0.2rem 0"}),
    ]
    if "vessel_type" in agg.columns:
        by_type = agg["vessel_type"].value_counts()
        children.append(html.Div([
            html.P("By type:", style={"color": DIM, "margin": "0.6rem 0 0.2rem 0", "fontWeight": "600"}),
            *[html.P(f"  {t}: {n}", style={"color": SOFT, "margin": "0.1rem 0", "fontSize": "0.72rem"})
              for t, n in by_type.items()],
        ]))
    if "gear_type" in agg.columns:
        n_trawlers = (agg["gear_type"].astype(str).str.upper() == "TRAWLERS").sum()
        children.append(html.P(f"Of which Trawlers (gear type): {n_trawlers}",
                                style={"color": "#ffd700", "marginTop": "0.5rem", "fontWeight": "600"}))
    if note:
        children.append(html.P(note, style={"color": DIM, "fontSize": "0.68rem", "fontStyle": "italic", "marginTop": "0.5rem"}))
    return html.Div(children)


def register_callbacks(app):

    # "Jump to a year" is just a convenience: it fills in Jan 1 -> Dec 31
    # of the chosen year, but doesn't restrict the pickers -- the user can
    # still edit either date afterwards, including across a year boundary.
    @app.callback(
        Output("prot-start", "date"),
        Output("prot-end", "date"),
        Input("prot-year", "value"),
        prevent_initial_call=True,
    )
    def jump_to_year(year):
        if not year:
            raise dash.exceptions.PreventUpdate
        return date(year, 1, 1), date(year, 12, 31)

    @app.callback(
        Output("prot-map-container", "children"),
        Output("prot-vessel-table", "children"),
        Output("prot-summary", "children"),
        Output("prot-csv-status", "children"),
        Input("prot-btn-run", "n_clicks"),
        Input("prot-btn-run-csv", "n_clicks"),
        State("prot-start", "date"),
        State("prot-end", "date"),
        State("prot-csv-selector", "value"),
        State("prot-gear-filter", "value"),
        prevent_initial_call=False,
    )
    def run_analysis(n1, n2, start, end, csv_path, gear_filter):
        trigger = dash.callback_context.triggered_id if dash.callback_context.triggered else None

        if _ZONE_POLYGON is None:
            msg = html.P("Zone not found: data/gis/fourni_protected.geojson is missing.",
                          style={"color": "#ff6b6b"})
            return _build_zone_map(pd.DataFrame()), "", msg, ""

        if trigger == "prot-btn-run-csv":
            if not csv_path:
                return _build_zone_map(pd.DataFrame()), "", "", "Choose a CSV first."
            try:
                df = load_csv(csv_path)
            except Exception as e:
                return _build_zone_map(pd.DataFrame()), "", "", f"Error: {e}"

            df_inside = _filter_points_in_zone(df)
            if gear_filter and "gear_type" in df_inside.columns:
                df_inside = df_inside[df_inside["gear_type"].astype(str).str.upper().isin(
                    [g.upper() for g in gear_filter])]
            agg = _aggregate_by_vessel(df_inside)
            note = "Analysis based on the imported CSV (gear_type available)."
            status = f"CSV loaded: {len(df):,} rows -> {len(df_inside):,} positions in the zone"
            return _build_zone_map(df_inside), _vessel_table(agg), _summary(agg, note), status

        if not start or not end:
            return _build_zone_map(pd.DataFrame()), "", "Select a date range.", ""

        # Be forgiving if the user picks the dates in the "wrong" order.
        if pd.to_datetime(start) > pd.to_datetime(end):
            start, end = end, start

        df = load_trajectories_range(start, end, None, None)
        df_inside = _filter_points_in_zone(df)
        agg = _aggregate_by_vessel(df_inside)
        note = ("Based on precomputed trajectories (vessel_type available, "
                "no gear_type -> import a CSV to isolate Trawlers).")
        return _build_zone_map(df_inside), _vessel_table(agg), _summary(agg, note), ""
