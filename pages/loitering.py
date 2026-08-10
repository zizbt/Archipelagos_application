"""
pages/loitering.py
==================
Page "Loitering" -- detecte les navires qui trainent : vitesse faible
maintenue pendant une duree minimale. Version simple (pas de distance
a la cote). Seuils reglables dans la sidebar.

Meme structure que les autres pages :
- Donnees : trajectoires precalculees via load_trajectories_range.
- Sidebar : plage de dates + seuils (vitesse max, duree min) + Analyze.
- Carte pydeck (ScatterplotLayer) : un cercle par evenement de loitering,
  taille proportionnelle a la duree.
- Table recapitulative + export CSV.
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

DEFAULT_SPEED = 1.5     # noeuds
DEFAULT_DURATION = 2.0  # heures

LOITER_COLOR = [255, 140, 0, 170]   # orange


# ---------------------------------------------------------------------------
# CALCUL DU LOITERING
# ---------------------------------------------------------------------------
def get_loitering_dataframe(df, speed_threshold_knots=DEFAULT_SPEED,
                            min_duration_hours=DEFAULT_DURATION):
    """
    Detecte les segments ou un navire reste sous un seuil de vitesse
    pendant au moins min_duration_hours. Renvoie un DataFrame plat.
    """
    empty_cols = ['vessel_id', 'ship_name', 'start', 'end', 'duration_hours',
                  'avg_speed_knots', 'n_points', 'centroid_lat', 'centroid_lon',
                  'max_radius_m']

    if df is None or df.empty or 'vessel_id' not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.dropna(subset=['lat', 'lon', 'date']).sort_values(['vessel_id', 'date'])

    # Projection metrique pour distances/vitesses
    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326")
    gdf_m = gdf.to_crs("EPSG:32634")
    d['x'] = gdf_m.geometry.x.values
    d['y'] = gdf_m.geometry.y.values

    events = []
    for vid, grp in d.groupby('vessel_id'):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue

        # Vitesse instantanee (noeuds) entre points consecutifs
        dx = grp['x'].diff()
        dy = grp['y'].diff()
        dist_m = np.sqrt(dx**2 + dy**2)
        dt_h = grp['date'].diff().dt.total_seconds() / 3600
        speed_knots = (dist_m / 1852) / dt_h.replace(0, np.nan)  # 1852 m = 1 NM
        grp['speed_knots'] = speed_knots

        # Points "lents" (sous le seuil)
        grp['slow'] = grp['speed_knots'] <= speed_threshold_knots
        grp['slow'] = grp['slow'].fillna(False)

        # Regrouper les runs consecutifs de points lents
        grp['run'] = (grp['slow'] != grp['slow'].shift()).cumsum()
        for _, run in grp[grp['slow']].groupby('run'):
            if len(run) < 2:
                continue
            start = run['date'].min()
            end = run['date'].max()
            dur = (end - start).total_seconds() / 3600
            if dur < min_duration_hours:
                continue

            cx, cy = run['x'].mean(), run['y'].mean()
            radius = float(np.sqrt((run['x'] - cx)**2 + (run['y'] - cy)**2).max())

            # centroid en lat/lon
            cpt = gpd.GeoSeries([gpd.points_from_xy([cx], [cy])[0]],
                                crs="EPSG:32634").to_crs("EPSG:4326").iloc[0]

            speeds = run['speed_knots'].dropna()
            avg_speed = round(float(speeds.mean()), 2) if not speeds.empty else 0.0

            events.append({
                'vessel_id': vid,
                'ship_name': run['ship_name'].iloc[0] if 'ship_name' in run else str(vid),
                'start': start, 'end': end,
                'duration_hours': round(dur, 2),
                'avg_speed_knots': avg_speed,
                'n_points': len(run),
                'centroid_lat': round(cpt.y, 5),
                'centroid_lon': round(cpt.x, 5),
                'max_radius_m': round(radius, 1),
            })

    if not events:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(events).sort_values('start').reset_index(drop=True)


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
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
            html.Button("Export CSV", id="loi-btn-export", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": PANEL, "color": SOFT,
                       "border": f"1px solid {BDR}", "borderRadius": "6px",
                       "cursor": "pointer", "marginBottom": "1rem"}),

            html.Div(id="loi-summary", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        # ── Map + table ──────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="loi-map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0}),
            html.Div(
                dcc.Loading(children=html.Div(id="loi-table")),
                style={"height": "260px", "flexShrink": "0", "overflowY": "auto",
                       "borderTop": f"1px solid {BDR}", "padding": "0.5rem 1rem", "background": BG},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


# ---------------------------------------------------------------------------
# HELPERS carte + table
# ---------------------------------------------------------------------------
def _build_map(loi_df):
    layers = []
    if loi_df is not None and not loi_df.empty:
        plot = loi_df.copy()
        plot["tooltip"] = (plot["ship_name"].astype(str)
                           + " (" + plot["duration_hours"].astype(str) + "h @ "
                           + plot["avg_speed_knots"].astype(str) + "kn)")
        plot["radius"] = plot["max_radius_m"].clip(lower=200)

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["centroid_lon", "centroid_lat"],
            get_fill_color=LOITER_COLOR,
            get_radius="radius", radius_min_pixels=4, radius_max_pixels=40,
            pickable=True, auto_highlight=True, opacity=0.5,
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
    return dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY,
                            style={"width": "100%", "height": "100%"})


def _table(loi_df):
    if loi_df is None or loi_df.empty:
        return html.P("No loitering detected for this period.",
                      style={"color": SOFT, "fontSize": "0.8rem"})
    show = loi_df.copy()
    show["start"] = show["start"].astype(str)
    show["end"] = show["end"].astype(str)
    cols = ['ship_name', 'start', 'end', 'duration_hours', 'avg_speed_knots',
            'n_points', 'max_radius_m']
    return dash_table.DataTable(
        data=show[cols].to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": f"1px solid {BDR}",
                    "fontSize": "0.72rem", "padding": "4px 8px"},
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
        page_size=20,
    )


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------
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
        Output("loi-table", "children"),
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
            return _build_map(None), _table(None), "No trajectory data for this range.", None

        loi = get_loitering_dataframe(df, speed_threshold_knots=float(speed),
                                      min_duration_hours=float(duration))
        summary = (f"{len(loi)} loitering event(s) "
                   f"(<= {speed} kn, >= {duration} h).") if not loi.empty else "No loitering found."
        store = (loi.assign(start=loi["start"].astype(str), end=loi["end"].astype(str))
                 .to_dict("records")) if not loi.empty else None
        return _build_map(loi), _table(loi), summary, store

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