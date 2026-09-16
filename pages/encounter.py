"""
pages/encounters.py
===================
Page "Encounters" -- dection of vessel encounters: two vessels within
DIST_THRESHOLD_M for TIME_THRESHOLD_H or more.

v2 -- SINGLE data source now: an imported CSV, exactly like
pages/heatmap.py (drag & drop + server-side cache). No more precomputed
trajectories (load_trajectories_range) and no more fixed global date
range -- the date-pickers bounds and the "jump to a year" dropdown are
derived live from the imported CSV's own "date" column.
"""

import json
from datetime import date
import base64
import io

import dash
import numpy as np
import pandas as pd
import pydeck as pdk
import dash_deck
import geopandas as gpd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl
from shared import AEGEAN_CENTER
from port_zones import port_mask_from_xy, PORT_RADIUS_M_DEFAULT, has_ports

PLACEHOLDER_STYLE = {"color": DIM, "padding": "2rem", "fontSize": "0.85rem", "fontStyle": "italic"}

DIST_THRESHOLD_M = 500
TIME_THRESHOLD_H = 2

ENC_COLOR = [128, 0, 128, 200]

# Cache serveur du CSV importé, comme _CSV_CACHE dans pages/heatmap.py --
# évite l'aller-retour du dataframe par le navigateur.
_CSV_CACHE = {"df": None, "filename": None}


def _parse_uploaded_csv(contents, filename):
    """Identique à heatmap._parse_uploaded_csv (dcc.Upload -> DataFrame)."""
    if contents is None:
        return None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
    return pd.read_csv(io.BytesIO(decoded))


def get_encounters_dataframe(df, dist_threshold_meters=DIST_THRESHOLD_M,
                             time_threshold_hours=TIME_THRESHOLD_H,
                             port_radius_m=PORT_RADIUS_M_DEFAULT):
    """
    Dection of vessel encounters: two vessels within dist_threshold_meters for
    time_threshold_hours or more.

    Each encounter is also flagged with 'in_port' (True/False): whether its
    midpoint lies within port_radius_m of a known port/harbour point
    (see ports.py). This lets the UI filter "at sea only" / "in port only".
    """
    empty_cols = ['vessel_1', 'vessel_2', 'vessel_1_id', 'vessel_2_id',
                  'start', 'end', 'duration_hours', 'n_points',
                  'median_distance_m', 'reliability', 'lat', 'lon', 'in_port']

    if df is None or df.empty or 'vessel_id' not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    from scipy.spatial import cKDTree

    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat),
                           crs="EPSG:4326").to_crs("EPSG:32634")
    g = pd.DataFrame({
        "vid": gdf["vessel_id"].astype(str).values,
        "ship": gdf["ship_name"].astype(str).values if "ship_name" in gdf.columns else "",
        "date": pd.to_datetime(gdf["date"].values),
        "x": gdf.geometry.x.values,
        "y": gdf.geometry.y.values,
    })
    g["tb"] = g["date"].dt.floor("30min")
    g = g.reset_index(drop=True)

    # Build candidate pairs per time bucket using a KD-tree instead of a
    # 9x-shifted grid cross-join, which can blow up memory when many points
    # share the same/adjacent grid cells (e.g. vessels idling in port).
    pair_rows = []
    for tb, grp in g.groupby("tb", sort=False):
        n = len(grp)
        if n < 2:
            continue
        coords = grp[["x", "y"]].to_numpy()
        tree = cKDTree(coords)
        # query_pairs returns each unordered pair of points within radius once
        pairs = tree.query_pairs(r=dist_threshold_meters, output_type="ndarray")
        if pairs.size == 0:
            continue
        idx_l = grp.index.to_numpy()[pairs[:, 0]]
        idx_r = grp.index.to_numpy()[pairs[:, 1]]
        pair_rows.append(np.column_stack([idx_l, idx_r]))

    if not pair_rows:
        return pd.DataFrame(columns=empty_cols)

    idx_pairs = np.concatenate(pair_rows, axis=0)
    left = g.loc[idx_pairs[:, 0]].reset_index(drop=True)
    right = g.loc[idx_pairs[:, 1]].reset_index(drop=True)
    m = left.join(right, lsuffix="_l", rsuffix="_r")

    # keep a consistent vid ordering per pair, dedupe identical pairs across
    # buckets, and drop self-pairs of the same vessel
    swap = m["vid_l"].to_numpy() > m["vid_r"].to_numpy()
    if swap.any():
        for c in ["vid", "ship", "date", "x", "y"]:
            l = m[f"{c}_l"].to_numpy().copy()
            r = m[f"{c}_r"].to_numpy().copy()
            l[swap], r[swap] = r[swap], l[swap]
            m[f"{c}_l"] = l
            m[f"{c}_r"] = r
    m = m[m["vid_l"] != m["vid_r"]]
    if m.empty:
        return pd.DataFrame(columns=empty_cols)

    m["dist_m"] = np.hypot(m["x_l"].values - m["x_r"].values,
                           m["y_l"].values - m["y_r"].values)
    m["time_diff"] = (m["date_l"] - m["date_r"]).abs()
    m = m[m["time_diff"] <= pd.Timedelta(minutes=30)]
    if m.empty:
        return pd.DataFrame(columns=empty_cols)
    m = m.drop_duplicates(subset=["vid_l", "vid_r", "date_l", "date_r"])

    cp = m[["vid_l", "vid_r", "ship_l", "ship_r", "date_l",
            "x_l", "y_l", "x_r", "y_r", "dist_m"]].copy()
    cp = cp.rename(columns={"date_l": "date"})
    cp["pair"] = cp["vid_l"] + "_" + cp["vid_r"]
    cp = cp.sort_values(["pair", "date"])
    cp["gap"] = cp.groupby("pair")["date"].diff()
    cp["grp"] = (cp["gap"] > pd.Timedelta(hours=1)).cumsum()

    agg = cp.groupby(["pair", "grp"]).agg(
        start=("date", "min"),
        end=("date", "max"),
        n_points=("date", "size"),
        median_distance_m=("dist_m", "median"),
        vessel_1=("ship_l", "first"),
        vessel_2=("ship_r", "first"),
        vessel_1_id=("vid_l", "first"),
        vessel_2_id=("vid_r", "first"),
        x_mid=("x_l", "median"),
        y_mid=("y_l", "median"),
        x1min=("x_l", "min"), x1max=("x_l", "max"),
        y1min=("y_l", "min"), y1max=("y_l", "max"),
        x2min=("x_r", "min"), x2max=("x_r", "max"),
        y2min=("y_r", "min"), y2max=("y_r", "max"),
    ).reset_index(drop=True)

    agg["duration_hours"] = (agg["end"] - agg["start"]).dt.total_seconds() / 3600
    agg = agg[agg["duration_hours"] >= time_threshold_hours]
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    MOVE_MIN_M = 500.0
    move1 = np.hypot(agg["x1max"] - agg["x1min"], agg["y1max"] - agg["y1min"])
    move2 = np.hypot(agg["x2max"] - agg["x2min"], agg["y2max"] - agg["y2min"])
    agg = agg[(move1 >= MOVE_MIN_M) | (move2 >= MOVE_MIN_M)]
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    agg["reliability"] = np.select(
        [(agg["n_points"] >= 8) & (agg["duration_hours"] >= 4),
         (agg["n_points"] >= 4)],
        ["high", "medium"], default="low")

    pts = gpd.GeoSeries(gpd.points_from_xy(agg["x_mid"], agg["y_mid"]),
                        crs="EPSG:32634").to_crs("EPSG:4326")
    agg["lat"] = pts.y.round(5).values
    agg["lon"] = pts.x.round(5).values
    agg["duration_hours"] = agg["duration_hours"].round(2)
    agg["median_distance_m"] = agg["median_distance_m"].round(1)
    agg["in_port"] = port_mask_from_xy(agg["x_mid"].values, agg["y_mid"].values,
                                       radius_m=port_radius_m)

    return agg[empty_cols].sort_values("start").reset_index(drop=True)


def _upload_zone():
    return dcc.Upload(
        id="enc-csv-upload",
        children=html.Div([
            "Drag a CSV here, or ",
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


# LAYOUT
def layout():
    return html.Div([
        dcc.Store(id="enc-store", data=None),
        dcc.Store(id="enc-store-csv-loaded", data=None),
        dcc.Download(id="enc-download-csv"),

        # ── Sidebar ──────────────────────────────────────────────
        html.Div([
            html.H6("Vessel encounters", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
            html.P(f"Two vessels within {DIST_THRESHOLD_M} m for {TIME_THRESHOLD_H} h or more.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            html.Div([
                html.H6("Import a CSV", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
                _upload_zone(),
                html.Div("No file selected", id="enc-csv-filename",
                          style={"fontSize": "0.72rem", "color": DIM,
                                 "fontStyle": "italic", "marginBottom": "0.6rem"}),
            ], style={"marginBottom": "1rem", "paddingBottom": "1rem",
                       "borderBottom": f"1px solid {BDR}"}),

            lbl("Jump to a year (optional)"),
            dcc.Dropdown(id="enc-year", value=None, clearable=True,
                options=[], placeholder="Import a CSV first...",
                style={"color": "#000", "marginBottom": "0.6rem"}),
            lbl("Start date"),
            dcc.DatePickerSingle(id="enc-start", date=None,
                display_format="YYYY-MM-DD",
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="enc-end", date=None,
                display_format="YYYY-MM-DD",
                style={"marginBottom": "0.6rem"}),
            html.P("Tip: keep the range short (days/weeks). Encounter detection is heavy.",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "0.6rem"}),

            lbl("Location"),
            html.P("No port reference file found (data/gis/ITA_vessels.geojson) "
                   "-- filter disabled.",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "0.4rem", "display": "block" if not has_ports() else "none"}),
            dcc.RadioItems(
                id="enc-port-filter",
                options=[
                    {"label": " Sea + Port", "value": "both"},
                    {"label": " At sea only", "value": "sea"},
                    {"label": " In port only", "value": "port"},
                ],
                value="both",
                labelStyle={"display": "block", "fontSize": "0.75rem",
                            "color": SOFT, "cursor": "pointer", "marginBottom": "0.15rem"},
                style={"marginBottom": "0.6rem",
                       "display": "block" if has_ports() else "none"},
            ),

            html.Div([
                lbl("Port radius (m)"),
                dcc.Slider(id="enc-port-radius", min=200, max=3000, step=100,
                    value=PORT_RADIUS_M_DEFAULT,
                    marks={200: "200", 1500: "1500", 3000: "3000"},
                    tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "1rem",
                      "display": "block" if has_ports() else "none"}),

            html.Button("Analyze", id="enc-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            html.Div(id="enc-summary", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        # ── Map full height + Export button top-right ────────────
        html.Div([
            html.Div(
                html.Button("Export CSV", id="enc-btn-export", n_clicks=0,
                    style={"border": "none",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                           "fontWeight": "600", "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": f"1px solid {BDR}", "flexShrink": "0",
                       "display": "flex", "justifyContent": "flex-end"},
            ),
            html.Div([
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="enc-map-container",
                        children=html.P("Import a CSV, then click \"Analyze\".", style=PLACEHOLDER_STYLE),
                        style={"height": "100%", "width": "100%"}),
                ),
                html.Div(id="enc-click-info",
                    style={"position": "absolute", "top": "0.6rem", "left": "0.6rem",
                           "maxWidth": "280px", "background": "rgba(26,13,42,0.95)",
                           "border": "1px solid #6b2d8f", "borderRadius": "8px",
                           "padding": "0.7rem 0.9rem", "color": "#e6d9f2",
                           "fontSize": "0.75rem", "display": "none", "zIndex": "10"}),
            ], style={"flex": "1", "minHeight": 0, "position": "relative"}),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


# HELPERS carte
def _build_map(enc_df):
    layers = []
    if enc_df is not None and not enc_df.empty:
        plot = enc_df.copy()
        plot["tooltip"] = (plot["vessel_1"].astype(str) + " <-> " + plot["vessel_2"].astype(str)
                           + " (" + plot["duration_hours"].astype(str) + "h)")
        plot["v1"] = plot["vessel_1"].astype(str)
        plot["v2"] = plot["vessel_2"].astype(str)
        plot["s"] = plot["start"].astype(str)
        plot["e"] = plot["end"].astype(str)
        plot["dur"] = plot["duration_hours"].astype(str)
        plot["dist"] = plot["median_distance_m"].astype(str)
        plot["rel"] = plot["reliability"].astype(str)
        plot["loc"] = np.where(plot["in_port"], "In port", "At sea")
        plot["radius"] = plot["duration_hours"].clip(lower=1) * 200

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["lon", "lat"],
            get_fill_color=ENC_COLOR,
            get_radius="radius", radius_min_pixels=4, radius_max_pixels=40,
            pickable=True, auto_highlight=True, opacity=0.7,
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=AEGEAN_CENTER["lat"], longitude=AEGEAN_CENTER["lon"],
            zoom=6, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    )
    deck_json = json.loads(deck.to_json())
    return dash_deck.DeckGL(id="enc-deck", data=deck_json, mapboxKey=MAPBOX_KEY,
                            tooltip={"text": "{tooltip}"},
                            style={"width": "100%", "height": "100%"})


def _click_panel(obj):
    """Contenu du panneau affiche au clic sur un cercle (clickInfo.object)."""
    if not obj:
        return "", {"display": "none"}
    v1 = obj.get("v1", "?")
    v2 = obj.get("v2", "?")
    body = html.Div([
        html.Div([html.B(v1), " ↔ ", html.B(v2)], style={"marginBottom": "4px"}),
        html.Div(f"Start: {obj.get('s', '-')}"),
        html.Div(f"End: {obj.get('e', '-')}"),
        html.Div(f"Duration: {obj.get('dur', '-')} h"),
        html.Div(f"Median distance: {obj.get('dist', '-')} m"),
        html.Div(f"Reliability: {obj.get('rel', '-')}"),
        html.Div(f"Location: {obj.get('loc', '-')}"),
    ])
    return body, {"position": "absolute", "top": "0.6rem", "left": "0.6rem",
                  "maxWidth": "280px", "background": "rgba(26,13,42,0.95)",
                  "border": "1px solid #6b2d8f", "borderRadius": "8px",
                  "padding": "0.7rem 0.9rem", "color": "#e6d9f2",
                  "fontSize": "0.75rem", "display": "block", "zIndex": "10"}


# CALLBACKS
def register_callbacks(app):

    # Parse le CSV dès qu'il est déposé -- peuple le sélecteur d'années et
    # les bornes des date-pickers (prises dans le fichier lui-même). Le
    # calcul des rencontres n'a lieu qu'au clic sur "Analyze", plus bas.
    @app.callback(
        Output("enc-csv-filename", "children"),
        Output("enc-year", "options"),
        Output("enc-year", "value"),
        Output("enc-start", "date"),
        Output("enc-start", "min_date_allowed"),
        Output("enc-start", "max_date_allowed"),
        Output("enc-end", "date"),
        Output("enc-end", "min_date_allowed"),
        Output("enc-end", "max_date_allowed"),
        Output("enc-store-csv-loaded", "data"),
        Input("enc-csv-upload", "contents"),
        State("enc-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv_uploaded(contents, filename):
        if not contents:
            raise dash.exceptions.PreventUpdate
        try:
            df = _parse_uploaded_csv(contents, filename)
        except Exception as e:
            _CSV_CACHE["df"] = None
            return (f"Error: {e}", [], None, None, None, None, None, None, None, None)

        required = {"lat", "lon", "vessel_id", "date"}
        if not required.issubset(df.columns):
            _CSV_CACHE["df"] = None
            missing = ", ".join(sorted(required - set(df.columns)))
            return (f'"{filename}" is missing required column(s): {missing}.',
                    [], None, None, None, None, None, None, None, None)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        _CSV_CACHE["df"] = df
        _CSV_CACHE["filename"] = filename

        min_date = df["date"].min()
        max_date = df["date"].max()
        min_d = min_date.date() if pd.notna(min_date) else None
        max_d = max_date.date() if pd.notna(max_date) else None

        years = sorted(df["date"].dt.year.dropna().astype(int).unique(), reverse=True)
        year_opts = [{"label": str(y), "value": y} for y in years]

        return (f"Loaded: {len(df):,} rows from \"{filename}\"", year_opts, None,
                min_d, min_d, max_d, max_d, min_d, max_d, "loaded")

    @app.callback(
        Output("enc-start", "date", allow_duplicate=True),
        Output("enc-end", "date", allow_duplicate=True),
        Input("enc-year", "value"),
        prevent_initial_call=True,
    )
    def _jump_year(year):
        if not year:
            raise dash.exceptions.PreventUpdate
        return date(year, 1, 1), date(year, 1, 31)

    @app.callback(
        Output("enc-map-container", "children"),
        Output("enc-summary", "children"),
        Output("enc-store", "data"),
        Input("enc-btn-run", "n_clicks"),
        State("enc-start", "date"),
        State("enc-end", "date"),
        State("enc-port-filter", "value"),
        State("enc-port-radius", "value"),
        prevent_initial_call=True,
    )
    def _run(n, start, end, port_filter, port_radius):
        if not n:
            raise dash.exceptions.PreventUpdate

        df = _CSV_CACHE.get("df")
        if df is None or df.empty:
            return html.P("Import a CSV first.", style=PLACEHOLDER_STYLE), "Import a CSV first.", None

        sub = df
        if start and end:
            s, e = pd.to_datetime(start), pd.to_datetime(end)
            if s > e:
                s, e = e, s
            sub = sub[(sub["date"] >= s) & (sub["date"] <= e + pd.Timedelta(days=1))]

        if sub.empty:
            return _build_map(None), "No rows for this range.", None

        enc = get_encounters_dataframe(sub, port_radius_m=port_radius or PORT_RADIUS_M_DEFAULT)

        if not enc.empty and port_filter in ("sea", "port"):
            enc = enc[enc["in_port"] == (port_filter == "port")]

        loc_txt = {"sea": " (at sea only)", "port": " (in port only)"}.get(port_filter, "")
        summary = (f"{len(enc)} encounter(s) found "
                   f"({start} -> {end}){loc_txt}.") if not enc.empty else "No encounter found."
        store = enc.assign(start=enc["start"].astype(str),
                           end=enc["end"].astype(str)).to_dict("records") if not enc.empty else None
        return _build_map(enc), summary, store

    @app.callback(
        Output("enc-click-info", "children"),
        Output("enc-click-info", "style"),
        Input("enc-deck", "clickInfo"),
        prevent_initial_call=True,
    )
    def _on_click(click_info):
        obj = (click_info or {}).get("object")
        return _click_panel(obj)

    # CSV Export
    @app.callback(
        Output("enc-download-csv", "data"),
        Input("enc-btn-export", "n_clicks"),
        State("enc-store", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        out = pd.DataFrame(store)
        return dcc.send_data_frame(out.to_csv, "encounters.csv", index=False)
