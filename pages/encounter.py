"""
pages/encounters.py
===================
Page "Encounters" -- detecte les rencontres entre navires : deux bateaux
restes a moins de 500 m l'un de l'autre pendant 2 h ou plus.

Meme structure que les autres pages :
- Source de donnees : trajectoires precalculees via load_trajectories_range
  (memes colonnes que la page Map : lat, lon, vessel_id, ship_name, date).
- Sidebar avec plage de dates + bouton Analyze.
- Carte pydeck (ScatterplotLayer) : un point par rencontre.
- Table recapitulative + export CSV.

La plage de dates peut chevaucher deux annees (ex. Dec 2024 -> Jan 2025).
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

# Bornes de dates de l'app
GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

# Parametres de detection
DIST_THRESHOLD_M = 500
TIME_THRESHOLD_H = 2

ENC_COLOR = [255, 0, 200, 180]   # rose/violet, comme tes points d'origine


# ---------------------------------------------------------------------------
# CALCUL DES RENCONTRES (logique reprise de VP_bulk_map.get_encounters_dataframe)
# ---------------------------------------------------------------------------
def get_encounters_dataframe(df, dist_threshold_meters=DIST_THRESHOLD_M,
                             time_threshold_hours=TIME_THRESHOLD_H):
    empty_cols = ['vessel_1', 'vessel_2', 'vessel_1_id', 'vessel_2_id',
                  'start', 'end', 'duration_hours', 'n_points',
                  'median_distance_m', 'reliability', 'lat', 'lon']

    if df is None or df.empty or 'vessel_id' not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf_metric = gdf.to_crs("EPSG:32634")
    gdf_metric['date'] = pd.to_datetime(gdf_metric['date'])
    gdf_metric = gdf_metric[['vessel_id', 'ship_name', 'date', 'geometry']]

    gdf_buffered = gdf_metric.copy()
    gdf_buffered['geometry'] = gdf_buffered.geometry.buffer(dist_threshold_meters)

    spatial_join = gpd.sjoin(gdf_metric, gdf_buffered, how='inner', predicate='within')
    pairs = spatial_join[spatial_join['vessel_id_left'] < spatial_join['vessel_id_right']].copy()
    if pairs.empty:
        return pd.DataFrame(columns=empty_cols)

    right_geom = gdf_metric.geometry
    pairs['geom_right'] = pairs['index_right'].map(right_geom)
    pairs['dist_m'] = pairs.geometry.distance(gpd.GeoSeries(pairs['geom_right'], crs=gdf_metric.crs))

    pairs['time_diff'] = (pairs['date_left'] - pairs['date_right']).abs()
    close_pairs = pairs[pairs['time_diff'] <= pd.Timedelta(minutes=30)].copy()
    if close_pairs.empty:
        return pd.DataFrame(columns=empty_cols)

    close_pairs['pair_id'] = (close_pairs['vessel_id_left'].astype(str) + "_"
                              + close_pairs['vessel_id_right'].astype(str))
    close_pairs = close_pairs.sort_values(['pair_id', 'date_left'])
    close_pairs['time_gap'] = close_pairs.groupby('pair_id')['date_left'].diff()
    close_pairs['group'] = (close_pairs['time_gap'] > pd.Timedelta(hours=1)).cumsum()

    rows = []
    for _, grp in close_pairs.groupby(['pair_id', 'group']):
        start = grp['date_left'].min()
        end = grp['date_left'].max()
        dur = (end - start).total_seconds() / 3600
        if dur < time_threshold_hours:
            continue

        n_points = len(grp)
        median_dist = round(float(grp['dist_m'].median()), 1)
        if n_points >= 8 and dur >= 4:
            reliability = "high"
        elif n_points >= 4:
            reliability = "medium"
        else:
            reliability = "low"

        mid = grp.iloc[len(grp) // 2]
        pt = gpd.GeoSeries([mid['geometry']], crs="EPSG:32634").to_crs("EPSG:4326").iloc[0]

        rows.append({
            'vessel_1': mid['ship_name_left'],
            'vessel_2': mid['ship_name_right'],
            'vessel_1_id': mid['vessel_id_left'],
            'vessel_2_id': mid['vessel_id_right'],
            'start': start, 'end': end,
            'duration_hours': round(dur, 2),
            'n_points': n_points,
            'median_distance_m': median_dist,
            'reliability': reliability,
            'lat': round(pt.y, 5), 'lon': round(pt.x, 5),
        })

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).sort_values('start').reset_index(drop=True)


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
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

            html.Button("Export CSV", id="enc-btn-export", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": PANEL, "color": SOFT,
                       "border": f"1px solid {BDR}", "borderRadius": "6px",
                       "cursor": "pointer", "marginBottom": "1rem"}),

            html.Div(id="enc-summary", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        # ── Map + table ──────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="enc-map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0}),

            html.Div(
                dcc.Loading(children=html.Div(id="enc-table")),
                style={"height": "260px", "flexShrink": "0", "overflowY": "auto",
                       "borderTop": f"1px solid {BDR}", "padding": "0.5rem 1rem", "background": BG},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


# ---------------------------------------------------------------------------
# HELPERS carte + table
# ---------------------------------------------------------------------------
def _build_map(enc_df):
    layers = []
    if enc_df is not None and not enc_df.empty:
        plot = enc_df.copy()
        plot["tooltip"] = (plot["vessel_1"].astype(str) + " <-> " + plot["vessel_2"].astype(str)
                           + " (" + plot["duration_hours"].astype(str) + "h, "
                           + plot["reliability"].astype(str) + ")")
        # rayon proportionnel a la duree (min 300 m visuel)
        plot["radius"] = plot["duration_hours"].clip(lower=1) * 200

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["lon", "lat"],
            get_fill_color=ENC_COLOR,
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


def _table(enc_df):
    if enc_df is None or enc_df.empty:
        return html.P("No encounter detected for this period.",
                      style={"color": SOFT, "fontSize": "0.8rem"})
    show = enc_df.copy()
    show["start"] = show["start"].astype(str)
    show["end"] = show["end"].astype(str)
    cols = ['vessel_1', 'vessel_2', 'start', 'end', 'duration_hours',
            'median_distance_m', 'reliability']
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

    # Raccourci "annee" -> remplit start/end
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

    # Analyze
    @app.callback(
        Output("enc-map-container", "children"),
        Output("enc-table", "children"),
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

        df = load_trajectories_range(start, end, None, None)
        if df is None or df.empty:
            return _build_map(None), _table(None), "No trajectory data for this range.", None

        enc = get_encounters_dataframe(df)
        summary = (f"{len(enc)} encounter(s) found "
                   f"({start} -> {end}).") if not enc.empty else "No encounter found."
        store = enc.assign(start=enc["start"].astype(str),
                           end=enc["end"].astype(str)).to_dict("records") if not enc.empty else None
        return _build_map(enc), _table(enc), summary, store

    # Export CSV
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