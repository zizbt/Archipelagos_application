"""
pages/heatmap.py
================
Page "Heatmaps" : grille 4 saisons (année + type de bateau, comme avant
-- pas de filtre pays, les fichiers précalculés ne contiennent que lat/lon).
Ajout : import d'un CSV personnel -> heatmap dédiée de ce jeu de données.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl, GFW_DOWNLOAD_DIR
from config import YEARS, VESSEL_TYPES, SEASON_ORDER, ROOT
from loader import load_heatmap
from gfw import list_downloaded_csvs, load_csv


def layout():
    return html.Div([
        dcc.Store(id="hm-store-csv-df", data=None),

        # Barre de contrôle
        html.Div([
            html.Div([lbl("Année"),
                dcc.Dropdown(id="hm-year", value=YEARS[-1], clearable=False,
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    style={"width": "100px", "color": "#000"})],
                style={"marginRight": "1.2rem"}),
            html.Div([lbl("Type de bateau"),
                dcc.Dropdown(id="hm-vtype", value=[], multi=True,
                    placeholder="Tous...",
                    options=[{"label": t.capitalize(), "value": t} for t in VESSEL_TYPES],
                    style={"width": "220px", "color": "#000"})],
                style={"marginRight": "1.2rem"}),
            html.Div([lbl(" "),
                html.Button("Afficher", id="hm-btn-show", n_clicks=0,
                    style={"padding": "0.45rem 1.2rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none",
                           "borderRadius": "6px", "cursor": "pointer",
                           "fontWeight": "600"})]),
            html.Div(id="hm-info",
                     style={"fontSize": "0.7rem", "color": DIM,
                            "marginLeft": "1rem", "alignSelf": "flex-end",
                            "paddingBottom": "2px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "gap": "0.5rem",
                   "padding": "0.8rem 1.2rem", "background": BG,
                   "borderBottom": f"1px solid {BDR}", "flexWrap": "wrap"}),

        # Import CSV personnel
        html.Div([
            html.Div([lbl("Importer un CSV téléchargé"),
                dcc.Dropdown(id="hm-csv-selector",
                    options=[{"label": f["filename"], "value": f["path"]}
                             for f in list_downloaded_csvs(ROOT / "data")],
                    value=None, placeholder="Choisir un CSV...",
                    style={"width": "320px", "color": "#000"})],
                style={"marginRight": "1rem"}),
            html.Div([lbl(" "),
                html.Button("Heatmap de ce CSV", id="hm-btn-show-csv", n_clicks=0,
                    style={"padding": "0.45rem 1.2rem", "background": PANEL,
                           "color": SOFT, "border": f"1px solid {BDR}",
                           "borderRadius": "6px", "cursor": "pointer"})]),
            html.Div(id="hm-csv-status",
                     style={"fontSize": "0.72rem", "color": SOFT,
                            "marginLeft": "1rem", "alignSelf": "flex-end", "paddingBottom": "6px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "gap": "0.5rem",
                   "padding": "0.6rem 1.2rem", "background": BG,
                   "borderBottom": f"1px solid {BDR}", "flexWrap": "wrap"}),

        # Grille des heatmaps (4 saisons OU CSV importé)
        html.Div(
            dcc.Loading(type="circle", color=ACC,
                children=html.Div(id="hm-container",
                    style={"height": "calc(100vh - 52px - 140px)", "width": "100%"})),
            style={"flex": "1", "minHeight": 0}),

    ], style={"display": "flex", "flexDirection": "column",
              "height": "calc(100vh - 52px)", "background": BG})


def _season_deck(year, season_name, vtypes):
    if vtypes:
        frames = [load_heatmap(year, season_name, t) for t in vtypes]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    else:
        df = load_heatmap(year, season_name, None)
    n_pts = len(df)
    if df.empty:
        df = pd.DataFrame({"lat": [37.5], "lon": [24.5]})
    layer = pdk.Layer("HeatmapLayer", data=df,
                       get_position=["lon", "lat"],
                       aggregation="SUM", radiusPixels=14,
                       intensity=2.2, threshold=0.015)
    deck = pdk.Deck(layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=37.5, longitude=24.5, zoom=6, pitch=0),
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json")
    return json.loads(deck.to_json()), n_pts


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


def register_callbacks(app):

    @app.callback(
        Output("hm-container", "children"),
        Output("hm-info", "children"),
        Output("hm-csv-status", "children"),
        Input("hm-btn-show", "n_clicks"),
        Input("hm-btn-show-csv", "n_clicks"),
        State("hm-year", "value"),
        State("hm-vtype", "value"),
        State("hm-csv-selector", "value"),
        prevent_initial_call=False,
    )
    def update_heatmap(n1, n2, year, vtypes, csv_path):
        trigger = dash.callback_context.triggered_id if dash.callback_context.triggered else None

        # Cas : heatmap à partir d'un CSV importé
        if trigger == "hm-btn-show-csv":
            if not csv_path:
                return html.P("Choisis un CSV d'abord.", style={"color": DIM, "padding": "2rem"}), "", ""
            try:
                df = load_csv(csv_path)
            except Exception as e:
                return html.P(f"Erreur : {e}", style={"color": "#ff6b6b", "padding": "2rem"}), "", f"Erreur : {e}"

            n_pts = len(df)
            df_hm = df[["lat", "lon"]].dropna() if not df.empty else pd.DataFrame({"lat": [37.5], "lon": [24.5]})
            layer = pdk.Layer("HeatmapLayer", data=df_hm,
                               get_position=["lon", "lat"],
                               aggregation="SUM", radiusPixels=14,
                               intensity=2.2, threshold=0.015)
            deck = pdk.Deck(layers=[layer],
                            initial_view_state=pdk.ViewState(
                                latitude=37.5, longitude=24.5, zoom=6, pitch=0),
                            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json")
            deck_json = json.loads(deck.to_json())

            panel = html.Div([
                dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY,
                                  style={"width": "100%", "height": "100%"}),
            ], style={"width": "100%", "height": "100%"})
            return panel, f"CSV importé — {n_pts:,} pts", f"Chargé : {n_pts:,} lignes"

        # Cas par défaut : grille 4 saisons précalculées
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda s: (s,) + _season_deck(year, s, vtypes), SEASON_ORDER))

        cards = []
        for sname, deck_json, npts in results:
            title = f"{sname} {year} — {npts:,} pts" if npts else f"{sname} {year} — no data"
            cards.append(_panel(title, deck_json, key=sname))

        grid = html.Div(cards, style={"display": "flex", "flexWrap": "wrap",
                                       "height": "100%", "width": "100%",
                                       "alignContent": "stretch"})
        return grid, f"4 saisons · {year}", ""
