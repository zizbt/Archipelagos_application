"""
pages/heatmap.py
================
Page "Heatmaps" — quatre modes, choisis via le radio "Mode" :

  - season     : grille 4 saisons (année + type de bateau) sur les fichiers
                 précalculés (pas de filtre pays, comme avant).
  - csv        : heatmap d'un CSV déjà téléchargé (page Data), importé
                 depuis data/gfw_downloads.
  - afe_single : heatmap de l'Apparent Fishing Effort (AFE) pour UN SEUL
                 navire, recherché par nom / MMSI / IMO (même recherche que
                 pages/ais_gap.py), sur une période choisie.
  - afe_bulk   : heatmap de l'AFE pour PLUSIEURS navires, filtrés par
                 pavillon(s) et/ou type(s) de navire, sur une période
                 choisie.

Les deux modes AFE appellent directement l'API GFW (comme pages/data.py et
pages/ais_gap.py) -- pas de CSV intermédiaire, la heatmap est construite
directement à partir du DataFrame en mémoire.
"""

import asyncio
import json
import threading
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

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

PANEL_ROW_STYLE = {"display": "flex", "alignItems": "flex-end", "gap": "0.5rem",
                    "flexWrap": "wrap"}
PANEL_ROW_HIDDEN = {"display": "none"}


def layout():
    return html.Div([
        dcc.Store(id="hm-store-csv-df", data=None),
        dcc.Store(id="hmafe-search-store", data=None),

        html.Div([
            # ── Sélecteur de mode ──
            html.Div([
                lbl("Mode"),
                dcc.RadioItems(
                    id="hm-mode",
                    value="season",
                    options=[
                        {"label": " 4 saisons (précalculé)", "value": "season"},
                        {"label": " Import CSV", "value": "csv"},
                        {"label": " AFE — single vessel", "value": "afe_single"},
                        {"label": " AFE — multiple vessels", "value": "afe_bulk"},
                    ],
                    labelStyle={"display": "inline-block", "marginRight": "16px",
                                "fontSize": "0.75rem", "color": SOFT, "cursor": "pointer"},
                ),
            ], style={"marginBottom": "0.7rem"}),

            # ── Panneau : grille 4 saisons ──
            html.Div(id="hm-panel-season", style=PANEL_ROW_STYLE, children=[
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
            ]),

            # ── Panneau : import CSV ──
            html.Div(id="hm-panel-csv", style=PANEL_ROW_HIDDEN, children=[
                html.Div([lbl("Import a downloaded CSV"),
                    dcc.Dropdown(id="hm-csv-selector",
                        options=[{"label": f["filename"], "value": f["path"]}
                                 for f in list_downloaded_csvs(ROOT / "data")],
                        value=None, placeholder="Choose a CSV...",
                        style={"width": "320px", "color": "#000"})],
                    style={"marginRight": "1rem"}),

                html.Div([lbl(" "),
                    html.Button("Heatmap of this CSV", id="hm-btn-show-csv", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem", "background": PANEL,
                               "color": SOFT, "border": f"1px solid {BDR}",
                               "borderRadius": "6px", "cursor": "pointer"})]),
            ]),

            # ── Panneau : AFE single vessel ──
            html.Div(id="hm-panel-afe-single", style=PANEL_ROW_HIDDEN, children=[
                html.Div([lbl("Vessel name / MMSI / IMO"),
                    dcc.Input(id="hmafe-query", type="text",
                        placeholder="Vessel name / MMSI / IMO", debounce=True,
                        style={"width": "220px", "padding": "0.4rem",
                               "borderRadius": "5px", "border": f"1px solid {BDR}",
                               "background": PANEL, "color": MAIN})],
                    style={"marginRight": "0.6rem"}),

                html.Div([lbl(" "),
                    html.Button("Search", id="hmafe-btn-search", n_clicks=0,
                        style={"padding": "0.44rem 1rem",
                               "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "600"})],
                    style={"marginRight": "1rem"}),

                html.Div([lbl("Vessel found"),
                    dcc.Loading(type="dot", color=ACC, children=
                        dcc.Dropdown(id="hmafe-vessel-selector", options=[], value=None,
                            placeholder="Search first...",
                            style={"width": "340px", "color": "#000"}))],
                    style={"marginRight": "1rem"}),

                html.Div([lbl("Start date"),
                    dcc.DatePickerSingle(id="hmafe-start", date=date(YEARS[-1], 1, 1),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style={"marginRight": "0.8rem"}),
                html.Div([lbl("End date"),
                    dcc.DatePickerSingle(id="hmafe-end", date=date(YEARS[-1], 12, 31),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style={"marginRight": "1rem"}),

                html.Div([lbl(" "),
                    html.Button("Show AFE heatmap", id="hmafe-btn-show-single", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem",
                               "background": "linear-gradient(135deg,#d15400,#a03e00)",
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "600"})]),
            ]),

            # ── Panneau : AFE multiple vessels ──
            html.Div(id="hm-panel-afe-bulk", style=PANEL_ROW_HIDDEN, children=[
                html.Div([lbl("Country / Flag"),
                    dcc.Dropdown(id="hmafe-bulk-flags",
                        options=[{"label": "ALL countries", "value": "ALL"}] +
                                [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f}
                                 for f in COUNTRY_FLAGS],
                        value=["GRC"], multi=True, placeholder="Select countries...",
                        style={"width": "260px", "color": "#000"})],
                    style={"marginRight": "1rem"}),

                html.Div([lbl("Vessel type (leave empty for ALL)"),
                    dcc.Dropdown(id="hmafe-bulk-vtypes",
                        options=[{"label": t.capitalize(), "value": t} for t in GFW_VESSEL_TYPES],
                        value=[], multi=True, placeholder="All types...",
                        style={"width": "220px", "color": "#000"})],
                    style={"marginRight": "1rem"}),

                html.Div([lbl("Start date"),
                    dcc.DatePickerSingle(id="hmafe-bulk-start", date=date(YEARS[-1], 1, 1),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style={"marginRight": "0.8rem"}),
                html.Div([lbl("End date"),
                    dcc.DatePickerSingle(id="hmafe-bulk-end", date=date(YEARS[-1], 12, 31),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE)],
                    style={"marginRight": "1rem"}),

                html.Div([lbl(" "),
                    html.Button("Show AFE heatmap", id="hmafe-btn-show-bulk", n_clicks=0,
                        style={"padding": "0.45rem 1.2rem",
                               "background": "linear-gradient(135deg,#d15400,#a03e00)",
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "600"})]),
            ]),

            html.Div(id="hm-info",
                     style={"fontSize": "0.7rem", "color": DIM,
                            "marginTop": "0.5rem"}),
            html.Div(id="hm-csv-status",
                     style={"fontSize": "0.72rem", "color": SOFT,
                            "marginTop": "0.15rem"}),
            html.Div(id="hmafe-status",
                     style={"fontSize": "0.72rem", "color": SOFT,
                            "marginTop": "0.15rem"}),

        ], style={"padding": "0.8rem 1.2rem", "background": BG,
                   "borderBottom": f"1px solid {BDR}"}),

        # Grille des heatmaps (4 saisons OU AFE OU CSV importé)
        html.Div(
            dcc.Loading(type="circle", color=ACC,
                children=html.Div(id="hm-container",
                    style={"height": "calc(100vh - 52px - 130px)", "width": "100%"})),
            style={"flex": "1", "minHeight": 0}),

    ], style={"display": "flex", "flexDirection": "column",
              "height": "calc(100vh - 52px)", "background": BG})


# ── Construction des decks ──────────────────────────────────────────────────

def _heatmap_deck_from_df(df_hm):
    """Construit un deck.gl HeatmapLayer a partir d'un DataFrame lat/lon."""
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


def _deck_panel(deck_json):
    return html.Div([
        dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY,
                          style={"width": "100%", "height": "100%"}),
    ], style={"width": "100%", "height": "100%"})


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


# ── Appels GFW synchrones (thread + boucle asyncio dédiée) ────────────────
# Même principe que pages/data.py::_do_download : le callback Dash est
# synchrone, donc on lance la coroutine dans un thread avec sa propre boucle
# asyncio et on attend le résultat (join).

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


def _afe_single_heatmap(idx, entries, start, end):
    """Renvoie (children, hm-info, hm-csv-status, hmafe-status)."""
    if not start or not end:
        return dash.no_update, dash.no_update, dash.no_update, "Please choose a start and end date."
    if idx is None or not entries:
        return dash.no_update, dash.no_update, dash.no_update, "Search and select a vessel first."

    api_key = get_api_key()
    if not api_key:
        return dash.no_update, dash.no_update, dash.no_update, "No API key saved."

    info = entries[int(idx)]
    vessel_ids = info.get("ids") or []
    if not vessel_ids:
        return dash.no_update, dash.no_update, dash.no_update, "This vessel has no usable vessel_id."

    client = get_gfw_client(api_key)
    res = _run_async_in_thread(lambda: bulk_load_afe_dataframe(
        None, start[:10], end[:10], client, vessel_ids=vessel_ids))

    if res.get("error"):
        return dash.no_update, dash.no_update, dash.no_update, "GFW error: " + res["error"][:120]

    df = res.get("value")
    vessel_name = info.get("name") or "this vessel"
    if df is None or df.empty:
        return dash.no_update, dash.no_update, dash.no_update, f"No AFE data for {vessel_name} in this period."
    if "lat" not in df.columns or "lon" not in df.columns:
        return dash.no_update, dash.no_update, dash.no_update, "AFE data has no lat/lon columns."

    df_hm = df[["lat", "lon"]].dropna()
    n_pts = len(df_hm)
    if df_hm.empty:
        return dash.no_update, dash.no_update, dash.no_update, f"No positions found for {vessel_name}/period."

    deck_json = _heatmap_deck_from_df(df_hm)
    title = f"AFE — {vessel_name} — {n_pts:,} pts"
    return _deck_panel(deck_json), title, "", f"{n_pts:,} positions loaded for {vessel_name}."


def _afe_bulk_heatmap(flags, vtypes, start, end):
    """Renvoie (children, hm-info, hm-csv-status, hmafe-status)."""
    if not start or not end:
        return dash.no_update, dash.no_update, dash.no_update, "Please choose a start and end date."

    api_key = get_api_key()
    if not api_key:
        return dash.no_update, dash.no_update, dash.no_update, "No API key saved."

    resolved_flags = COUNTRY_FLAGS if (flags and "ALL" in flags) else (flags or None)

    client = get_gfw_client(api_key)
    res = _run_async_in_thread(lambda: bulk_load_afe_dataframe(
        resolved_flags, start[:10], end[:10], client, vessel_types=vtypes or None))

    if res.get("error"):
        return dash.no_update, dash.no_update, dash.no_update, "GFW error: " + res["error"][:120]

    df = res.get("value")
    if df is None or df.empty:
        return dash.no_update, dash.no_update, dash.no_update, "No AFE data for this filter/period."
    if "lat" not in df.columns or "lon" not in df.columns:
        return dash.no_update, dash.no_update, dash.no_update, "AFE data has no lat/lon columns."

    df_hm = df[["lat", "lon"]].dropna()
    n_pts = len(df_hm)
    if df_hm.empty:
        return dash.no_update, dash.no_update, dash.no_update, "No positions found for this filter/period."

    n_vessels = df["vessel_id"].nunique() if "vessel_id" in df.columns else None
    vlabel = f", {n_vessels} vessel(s)" if n_vessels else ""

    deck_json = _heatmap_deck_from_df(df_hm)
    title = f"AFE — {n_pts:,} pts{vlabel}"
    return _deck_panel(deck_json), title, "", f"{n_pts:,} positions loaded{vlabel}."


# ── CALLBACKS ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("hm-panel-season", "style"),
        Output("hm-panel-csv", "style"),
        Output("hm-panel-afe-single", "style"),
        Output("hm-panel-afe-bulk", "style"),
        Input("hm-mode", "value"),
    )
    def _toggle_hm_mode(mode):
        return (
            PANEL_ROW_STYLE if mode == "season" else PANEL_ROW_HIDDEN,
            PANEL_ROW_STYLE if mode == "csv" else PANEL_ROW_HIDDEN,
            PANEL_ROW_STYLE if mode == "afe_single" else PANEL_ROW_HIDDEN,
            PANEL_ROW_STYLE if mode == "afe_bulk" else PANEL_ROW_HIDDEN,
        )

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
        Input("hm-btn-show", "n_clicks"),
        Input("hm-btn-show-csv", "n_clicks"),
        Input("hmafe-btn-show-single", "n_clicks"),
        Input("hmafe-btn-show-bulk", "n_clicks"),
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
        prevent_initial_call='initial_duplicate',
    )
    def update_heatmap(n1, n2, n3, n4, year, vtypes, csv_path,
                        afe_idx, afe_entries, afe_start, afe_end,
                        bulk_flags, bulk_vtypes, bulk_start, bulk_end):
        trigger = dash.callback_context.triggered_id if dash.callback_context.triggered else None

        # Cas : heatmap AFE — single vessel
        if trigger == "hmafe-btn-show-single":
            return _afe_single_heatmap(afe_idx, afe_entries, afe_start, afe_end)

        # Cas : heatmap AFE — multiple vessels
        if trigger == "hmafe-btn-show-bulk":
            return _afe_bulk_heatmap(bulk_flags, bulk_vtypes, bulk_start, bulk_end)

        # Cas : heatmap à partir d'un CSV importé
        if trigger == "hm-btn-show-csv":
            if not csv_path:
                return html.P("Choisis un CSV d'abord.", style={"color": DIM, "padding": "2rem"}), "", "", ""
            try:
                df = load_csv(csv_path)
            except Exception as e:
                return html.P(f"Erreur : {e}", style={"color": "#ff6b6b", "padding": "2rem"}), "", f"Erreur : {e}", ""

            n_pts = len(df)
            df_hm = df[["lat", "lon"]].dropna() if not df.empty else pd.DataFrame({"lat": [37.5], "lon": [24.5]})
            deck_json = _heatmap_deck_from_df(df_hm)
            return _deck_panel(deck_json), f"CSV importé — {n_pts:,} pts", f"Chargé : {n_pts:,} lignes", ""

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
        return grid, f"4 saisons · {year}", "", ""