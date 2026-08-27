"""
pages/encounters.py
===================
Page "Encounters" -- dection of vessel encounters: two vessels within
DIST_THRESHOLD_M for TIME_THRESHOLD_H or more.
"""

import json
from datetime import date

import dash
import numpy as np
import pandas as pd
import pydeck as pdk
import dash_deck
import geopandas as gpd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl
from shared import AEGEAN_CENTER
from config import YEARS, FLAG_NAMES
from loader import load_trajectories_range

TRAJECTORY_COLUMNS = ["lat", "lon", "vessel_id", "ship_name", "date"]

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

DIST_THRESHOLD_M = 500
TIME_THRESHOLD_H = 2

ENC_COLOR = [128, 0, 128, 200]

def get_encounters_dataframe(df, dist_threshold_meters=DIST_THRESHOLD_M,
                             time_threshold_hours=TIME_THRESHOLD_H):
    """
    Dection of vessel encounters: two vessels within dist_threshold_meters for
    time_threshold_hours or more.
    """
    empty_cols = ['vessel_1', 'vessel_2', 'vessel_1_id', 'vessel_2_id',
                  'start', 'end', 'duration_hours', 'n_points',
                  'median_distance_m', 'reliability', 'lat', 'lon']

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

    return agg[empty_cols].sort_values("start").reset_index(drop=True)

# LAYOUT
def layout():
    return html.Div([
        dcc.Store(id="enc-store", data=None),
        dcc.Download(id="enc-download-csv"),

        # ── Sidebar ──────────────────────────────────────────────
        html.Div([
            html.H6("Vessel encounters", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
            html.P(f"Two vessels within {DIST_THRESHOLD_M} m for {TIME_THRESHOLD_H} h or more.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Jump to a year (optional)"),
            dcc.Dropdown(id="enc-year", value=None, clearable=True,
                options=[{"label": str(y), "value": y} for y in YEARS],
                placeholder="Jump to a year...",
                style={"color": "#000", "marginBottom": "0.6rem"}),
            lbl("Start date"),
            dcc.DatePickerSingle(id="enc-start", date=date(YEARS[-1], 1, 1),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="enc-end", date=date(YEARS[-1], 1, 31),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            html.P("Tip: keep the range short (days/weeks). Encounter detection is heavy.",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "0.6rem"}),

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
                    children=html.Div(id="enc-map-container", style={"height": "100%", "width": "100%"}),
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
    ])
    return body, {"position": "absolute", "top": "0.6rem", "left": "0.6rem",
                  "maxWidth": "280px", "background": "rgba(26,13,42,0.95)",
                  "border": "1px solid #6b2d8f", "borderRadius": "8px",
                  "padding": "0.7rem 0.9rem", "color": "#e6d9f2",
                  "fontSize": "0.75rem", "display": "block", "zIndex": "10"}


# CALLBACKS
def register_callbacks(app):

    @app.callback(
        Output("enc-start", "date"),
        Output("enc-end", "date"),
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
        prevent_initial_call=True,
    )
    def _run(n, start, end):
        if not n:
            raise dash.exceptions.PreventUpdate

        df = load_trajectories_range(start, end, None, None, columns=TRAJECTORY_COLUMNS)
        if df is None or df.empty:
            return _build_map(None), "No trajectory data for this range.", None

        enc = get_encounters_dataframe(df)
        summary = (f"{len(enc)} encounter(s) found "
                   f"({start} -> {end}).") if not enc.empty else "No encounter found."
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