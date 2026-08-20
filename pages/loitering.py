"""
pages/loitering.py
==================
Page "Loitering" -- Detection of vessels holding a low speed for a sustained period.

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

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl, AEGEAN_CENTER
from config import YEARS
from loader import load_trajectories_range

TRAJECTORY_COLUMNS = ["lat", "lon", "vessel_id", "ship_name", "date"]

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

DEFAULT_SPEED = 1.5
DEFAULT_DURATION = 2.0

LOITER_COLOR = [255, 140, 0, 170]

# CALCUL DU LOITERING
def get_loitering_dataframe(df, speed_threshold_knots=DEFAULT_SPEED,
                            min_duration_hours=DEFAULT_DURATION):
    """
    Detecte les segments ou un navire reste sous un seuil de vitesse pendant
    au moins min_duration_hours. Version vectorisee (pas de double boucle).

    Filtre "en mer" : on ecarte les evenements de rayon quasi nul, qui
    correspondent a des navires immobiles a quai (vitesse ~0 en continu).
    """
    empty_cols = ['vessel_id', 'ship_name', 'start', 'end', 'duration_hours',
                  'avg_speed_knots', 'n_points', 'centroid_lat', 'centroid_lon',
                  'max_radius_m']
    MIN_RADIUS_M = 100.0

    if df is None or df.empty or 'vessel_id' not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.dropna(subset=['lat', 'lon', 'date']).sort_values(['vessel_id', 'date'])
    if d.empty:
        return pd.DataFrame(columns=empty_cols)

    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326")
    gdf_m = gdf.to_crs("EPSG:32634")
    d['x'] = gdf_m.geometry.x.values
    d['y'] = gdf_m.geometry.y.values
    d = d.reset_index(drop=True)

    grp = d.groupby('vessel_id', sort=False)
    px = grp['x'].shift()
    py = grp['y'].shift()
    pdate = grp['date'].shift()
    dist_m = np.hypot(d['x'] - px, d['y'] - py)
    dt_h = (d['date'] - pdate).dt.total_seconds() / 3600
    d['speed_knots'] = (dist_m / 1852) / dt_h.replace(0, np.nan)

    d['slow'] = (d['speed_knots'] <= speed_threshold_knots).fillna(False)

    vessel_change = d['vessel_id'].ne(d['vessel_id'].shift())
    slow_change = d['slow'].ne(d['slow'].shift())
    d['run'] = (vessel_change | slow_change).cumsum()

    slow = d[d['slow']].copy()
    if slow.empty:
        return pd.DataFrame(columns=empty_cols)

    g = slow.groupby('run')
    agg = g.agg(
        vessel_id=('vessel_id', 'first'),
        ship_name=('ship_name', 'first'),
        start=('date', 'min'),
        end=('date', 'max'),
        n_points=('date', 'size'),
        avg_speed_knots=('speed_knots', 'mean'),
        cx=('x', 'mean'),
        cy=('y', 'mean'),
        xmin=('x', 'min'), xmax=('x', 'max'),
        ymin=('y', 'min'), ymax=('y', 'max'),
    ).reset_index(drop=True)

    agg = agg[agg['n_points'] >= 2]
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    agg['duration_hours'] = (agg['end'] - agg['start']).dt.total_seconds() / 3600
    agg = agg[agg['duration_hours'] >= min_duration_hours]
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    agg['max_radius_m'] = np.hypot(agg['xmax'] - agg['xmin'],
                                   agg['ymax'] - agg['ymin']) / 2.0
    agg = agg[agg['max_radius_m'] >= MIN_RADIUS_M]
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    pts = gpd.GeoSeries(gpd.points_from_xy(agg['cx'], agg['cy']),
                        crs="EPSG:32634").to_crs("EPSG:4326")
    agg['centroid_lat'] = pts.y.round(5).values
    agg['centroid_lon'] = pts.x.round(5).values
    agg['duration_hours'] = agg['duration_hours'].round(2)
    agg['avg_speed_knots'] = agg['avg_speed_knots'].round(2)
    agg['max_radius_m'] = agg['max_radius_m'].round(1)

    return agg[empty_cols].sort_values('start').reset_index(drop=True)


# LAYOUT
def layout():
    return html.Div([
        dcc.Store(id="loi-store", data=None),
        dcc.Download(id="loi-download-csv"),

        # ── Sidebar ──────────────────────────────────────────────
        html.Div([
            html.H6("Loitering vessels", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
            html.P("Vessels holding a low speed for a sustained period.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Jump to a year (optional)"),
            dcc.Dropdown(id="loi-year", value=None, clearable=True,
                options=[{"label": str(y), "value": y} for y in YEARS],
                placeholder="Jump to a year...",
                style={"color": "#000", "marginBottom": "0.6rem"}),
            lbl("Start date"),
            dcc.DatePickerSingle(id="loi-start", date=date(YEARS[-1], 1, 1),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="loi-end", date=date(YEARS[-1], 1, 31),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "1rem"}),

            html.Div([
                lbl(f"Max speed (knots)"),
                dcc.Slider(id="loi-speed", min=0.5, max=5, step=0.5, value=DEFAULT_SPEED,
                    marks={0.5: "0.5", 2.5: "2.5", 5: "5"},
                    tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "1rem"}),

            html.Div([
                lbl("Min duration (hours)"),
                dcc.Slider(id="loi-duration", min=1, max=12, step=1, value=int(DEFAULT_DURATION),
                    marks={1: "1", 6: "6", 12: "12"},
                    tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "1rem"}),

            html.Button("Analyze", id="loi-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            html.Div(id="loi-summary", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        # ── Map full height + Export button top-right ────────────
        html.Div([
            html.Div(
                html.Button("Export CSV", id="loi-btn-export", n_clicks=0,
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
                    children=html.Div(id="loi-map-container", style={"height": "100%", "width": "100%"}),
                ),
                html.Div(id="loi-click-info",
                    style={"position": "absolute", "top": "0.6rem", "left": "0.6rem",
                           "maxWidth": "280px", "background": "rgba(40,24,0,0.95)",
                           "border": "1px solid #b3600a", "borderRadius": "8px",
                           "padding": "0.7rem 0.9rem", "color": "#ffe6c2",
                           "fontSize": "0.75rem", "display": "none", "zIndex": "10"}),
            ], style={"flex": "1", "minHeight": 0, "position": "relative"}),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


# HELPERS
def _build_map(loi_df):
    layers = []
    if loi_df is not None and not loi_df.empty:
        plot = loi_df.copy()
        plot["tooltip"] = (plot["ship_name"].astype(str)
                           + " (" + plot["duration_hours"].astype(str) + "h @ "
                           + plot["avg_speed_knots"].astype(str) + "kn)")
        plot["ship"] = plot["ship_name"].astype(str)
        plot["s"] = plot["start"].astype(str)
        plot["e"] = plot["end"].astype(str)
        plot["dur"] = plot["duration_hours"].astype(str)
        plot["spd"] = plot["avg_speed_knots"].astype(str)
        plot["npt"] = plot["n_points"].astype(str)
        plot["rad"] = plot["max_radius_m"].astype(str)
        plot["radius"] = plot["max_radius_m"].clip(lower=200)

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["centroid_lon", "centroid_lat"],
            get_fill_color=LOITER_COLOR,
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
    return dash_deck.DeckGL(id="loi-deck", data=deck_json, mapboxKey=MAPBOX_KEY,
                            tooltip={"text": "{tooltip}"},
                            style={"width": "100%", "height": "100%"})


def _click_panel(obj):
    """Sign to display a small info panel when clicking on a loitering event."""
    if not obj:
        return "", {"display": "none"}
    body = html.Div([
        html.Div(html.B(obj.get("ship", "?")), style={"marginBottom": "4px"}),
        html.Div(f"Start: {obj.get('s', '-')}"),
        html.Div(f"End: {obj.get('e', '-')}"),
        html.Div(f"Duration: {obj.get('dur', '-')} h"),
        html.Div(f"Avg speed: {obj.get('spd', '-')} kn"),
        html.Div(f"Points: {obj.get('npt', '-')}"),
        html.Div(f"Max radius: {obj.get('rad', '-')} m"),
    ])
    return body, {"position": "absolute", "top": "0.6rem", "left": "0.6rem",
                  "maxWidth": "280px", "background": "rgba(40,24,0,0.95)",
                  "border": "1px solid #b3600a", "borderRadius": "8px",
                  "padding": "0.7rem 0.9rem", "color": "#ffe6c2",
                  "fontSize": "0.75rem", "display": "block", "zIndex": "10"}


# CALLBACKS
def register_callbacks(app):

    @app.callback(
        Output("loi-start", "date"),
        Output("loi-end", "date"),
        Input("loi-year", "value"),
        prevent_initial_call=True,
    )
    def _jump_year(year):
        if not year:
            raise dash.exceptions.PreventUpdate
        return date(year, 1, 1), date(year, 1, 31)

    @app.callback(
        Output("loi-map-container", "children"),
        Output("loi-summary", "children"),
        Output("loi-store", "data"),
        Input("loi-btn-run", "n_clicks"),
        State("loi-start", "date"),
        State("loi-end", "date"),
        State("loi-speed", "value"),
        State("loi-duration", "value"),
        prevent_initial_call=True,
    )
    def _run(n, start, end, speed, duration):
        if not n:
            raise dash.exceptions.PreventUpdate

        df = load_trajectories_range(start, end, None, None, columns=TRAJECTORY_COLUMNS)
        if df is None or df.empty:
            return _build_map(None), "No trajectory data for this range.", None

        loi = get_loitering_dataframe(df, speed_threshold_knots=float(speed),
                                      min_duration_hours=float(duration))
        summary = (f"{len(loi)} loitering event(s) "
                   f"(<= {speed} kn, >= {duration} h).") if not loi.empty else "No loitering found."
        store = (loi.assign(start=loi["start"].astype(str), end=loi["end"].astype(str))
                 .to_dict("records")) if not loi.empty else None
        return _build_map(loi), summary, store

    @app.callback(
        Output("loi-click-info", "children"),
        Output("loi-click-info", "style"),
        Input("loi-deck", "clickInfo"),
        prevent_initial_call=True,
    )
    def _on_click(click_info):
        obj = (click_info or {}).get("object")
        return _click_panel(obj)

    @app.callback(
        Output("loi-download-csv", "data"),
        Input("loi-btn-export", "n_clicks"),
        State("loi-store", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        return dcc.send_data_frame(pd.DataFrame(store).to_csv, "loitering.csv", index=False)