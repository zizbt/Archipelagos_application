"""
pages/ports.py
==============
Page "Port visits" -- recupere les visites de ports via l'API GFW
(dataset public-global-port-visits-events:latest) pour les navires
presents dans un CSV telecharge, puis agrege par port.

Workflow :
- L'utilisateur choisit un CSV deja telecharge (page Data) + saisit sa cle API.
- On extrait les vessel_id uniques du CSV.
- On appelle GFW PORT_VISIT sur ces navires (logique reprise de Port_visits.py).
- On agrege par port : taille du point = nombre de visites.
- Table detaillee (avec confidence) + export CSV.

L'appel API peut etre long (reseau + pagination) : dcc.Loading affiche
un spinner pendant le traitement.
"""

import json
import asyncio
from datetime import date

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl, AEGEAN_CENTER
from config import YEARS, ROOT, FLAG_NAMES
from gfw import get_gfw_client, list_downloaded_csvs, load_csv

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

PORT_COLOR = [0, 200, 255, 190]   # cyan


# ---------------------------------------------------------------------------
# APPEL GFW PORT_VISIT (repris a l'identique de Port_visits.py)
# ---------------------------------------------------------------------------
async def _load_port_visits(vessel_ids, start, end, client):
    if isinstance(vessel_ids, str):
        vessel_ids = [vessel_ids]

    events = await client.events.get_all_events(
        datasets=["public-global-port-visits-events:latest"],
        vessels=vessel_ids,
        start_date=start,
        end_date=end,
        limit=99999,
    )
    df = events.df()
    if df.empty:
        return df

    df = df.copy()
    df["_start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["_end"] = pd.to_datetime(df["end"], utc=True, errors="coerce")
    df = df.sort_values("_start")

    # Filtre 1 : visites corrompues (sortie jamais detectee)
    corrupted = df["_end"] > df["_start"].shift(-1)
    df = df[~corrupted]
    if df.empty:
        return df

    # Filtre 2 : visites chevauchant la fenetre demandee
    win_start = pd.to_datetime(start, utc=True)
    win_end = pd.to_datetime(end, utc=True) + pd.Timedelta(days=1)
    overlaps = (df["_start"] < win_end) & (df["_end"] >= win_start)
    df = df[overlaps].drop(columns=["_start", "_end"])
    if df.empty:
        return df

    def _anchor(pv, key):
        if pv is None:
            return None
        if hasattr(pv, "model_dump"):
            pv = pv.model_dump()
        elif hasattr(pv, "dict"):
            pv = pv.dict()
        if not isinstance(pv, dict):
            return None
        a = (pv.get("intermediate_anchorage")
             or pv.get("intermediateAnchorage")
             or pv.get("start_anchorage")
             or pv.get("startAnchorage")
             or {})
        if hasattr(a, "model_dump"):
            a = a.model_dump()
        return a.get(key) if isinstance(a, dict) else None

    df["port_name"] = df["port_visit"].apply(lambda p: _anchor(p, "name"))
    df["port_flag"] = df["port_visit"].apply(lambda p: _anchor(p, "flag"))
    df["port_id"] = df["port_visit"].apply(lambda p: _anchor(p, "id"))

    def _field(pv, key):
        if hasattr(pv, "model_dump"):
            pv = pv.model_dump()
        return pv.get(key) if isinstance(pv, dict) else None

    df["duration_hrs"] = df["port_visit"].apply(lambda p: _field(p, "duration_hrs"))
    df["confidence"] = df["port_visit"].apply(lambda p: _field(p, "confidence"))

    df["start"] = pd.to_datetime(df["start"]).dt.strftime("%Y-%m-%d %H:%M")
    df["end"] = pd.to_datetime(df["end"]).dt.strftime("%Y-%m-%d %H:%M")
    df["duration_hrs"] = pd.to_numeric(df["duration_hrs"], errors="coerce").round(1)

    keep = ["start", "end", "port_name", "port_flag", "port_id",
            "duration_hrs", "confidence", "lat", "lon"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values("start")


def fetch_port_visits(vessel_ids, start, end, api_key):
    """Wrapper synchrone : cree le client GFW et lance l'appel async."""
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(_load_port_visits(vessel_ids, start, end, client))
    finally:
        loop.close()
    return df


def aggregate_by_port(visits):
    """Agrege les visites par port : nb de visites + duree totale + confiance moyenne."""
    if visits is None or visits.empty:
        return pd.DataFrame(columns=["port_name", "port_flag", "n_visits",
                                     "total_hours", "avg_confidence", "lat", "lon"])
    g = (visits.groupby(["port_name", "port_flag"], dropna=False)
         .agg(n_visits=("start", "size"),
              total_hours=("duration_hrs", "sum"),
              avg_confidence=("confidence", "mean"),
              lat=("lat", "mean"),
              lon=("lon", "mean"))
         .reset_index())
    g["total_hours"] = g["total_hours"].round(1)
    g["avg_confidence"] = g["avg_confidence"].round(2)
    return g.sort_values("n_visits", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
def layout():
    return html.Div([
        dcc.Store(id="port-store", data=None),
        dcc.Download(id="port-download-csv"),

        html.Div([
            html.H6("Port visits", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
            html.P("Fetches GFW port-visit events for the vessels in a downloaded CSV.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("GFW API key"),
            dcc.Input(id="port-api-key", type="password", placeholder="Paste your API key...",
                style={"width": "100%", "marginBottom": "0.8rem", "color": "#000",
                       "padding": "0.4rem", "borderRadius": "4px"}),

            lbl("Downloaded CSV (vessel source)"),
            dcc.Dropdown(id="port-csv-selector",
                options=[{"label": f["filename"], "value": f["path"]}
                         for f in list_downloaded_csvs(ROOT / "data")],
                value=None, placeholder="Choose a CSV...",
                style={"color": "#000", "marginBottom": "0.8rem"}),

            lbl("Start date"),
            dcc.DatePickerSingle(id="port-start", date=date(YEARS[-1], 1, 1),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="port-end", date=date(YEARS[-1], 12, 31),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "1rem"}),

            html.Button("Fetch port visits", id="port-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),
            html.Button("Export CSV", id="port-btn-export", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": PANEL, "color": SOFT,
                       "border": f"1px solid {BDR}", "borderRadius": "6px",
                       "cursor": "pointer", "marginBottom": "1rem"}),

            dcc.Loading(type="dot", color=ACC,
                children=html.Div(id="port-status", style={"fontSize": "0.72rem", "color": SOFT})),
            html.Div(id="port-summary", style={"fontSize": "0.75rem", "color": SOFT, "marginTop": "0.6rem"}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        html.Div([
            html.Div([
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="port-map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0}),
            html.Div(
                dcc.Loading(children=html.Div(id="port-table")),
                style={"height": "260px", "flexShrink": "0", "overflowY": "auto",
                       "borderTop": f"1px solid {BDR}", "padding": "0.5rem 1rem", "background": BG},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


# ---------------------------------------------------------------------------
# HELPERS carte + table
# ---------------------------------------------------------------------------
def _build_map(agg):
    layers = []
    if agg is not None and not agg.empty:
        plot = agg.dropna(subset=["lat", "lon"]).copy()
        plot["tooltip"] = (plot["port_name"].astype(str)
                           + " - " + plot["n_visits"].astype(str) + " visits ("
                           + plot["total_hours"].astype(str) + "h)")
        # rayon proportionnel au nombre de visites
        plot["radius"] = (plot["n_visits"] ** 0.5) * 800

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["lon", "lat"],
            get_fill_color=PORT_COLOR,
            get_radius="radius", radius_min_pixels=5, radius_max_pixels=50,
            pickable=True, auto_highlight=True, opacity=0.6,
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


def _table(agg):
    if agg is None or agg.empty:
        return html.P("No port visit found.", style={"color": SOFT, "fontSize": "0.8rem"})
    show = agg.copy()
    if "port_flag" in show.columns:
        show["port_flag"] = show["port_flag"].map(lambda f: FLAG_NAMES.get(f, f) if pd.notna(f) else "?")
    cols = ["port_name", "port_flag", "n_visits", "total_hours", "avg_confidence"]
    cols = [c for c in cols if c in show.columns]
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
        Output("port-map-container", "children"),
        Output("port-table", "children"),
        Output("port-summary", "children"),
        Output("port-status", "children"),
        Output("port-store", "data"),
        Input("port-btn-run", "n_clicks"),
        State("port-api-key", "value"),
        State("port-csv-selector", "value"),
        State("port-start", "date"),
        State("port-end", "date"),
        prevent_initial_call=True,
    )
    def _run(n, api_key, csv_path, start, end):
        if not n:
            raise dash.exceptions.PreventUpdate
        if not api_key:
            return _build_map(None), _table(None), "", "Please enter your GFW API key.", None
        if not csv_path:
            return _build_map(None), _table(None), "", "Please choose a CSV.", None

        try:
            df = load_csv(csv_path)
        except Exception as e:
            return _build_map(None), _table(None), "", f"CSV error: {str(e)[:80]}", None

        if "vessel_id" not in df.columns:
            return _build_map(None), _table(None), "", "This CSV has no vessel_id column.", None

        vessel_ids = df["vessel_id"].dropna().unique().tolist()
        if not vessel_ids:
            return _build_map(None), _table(None), "", "No vessel in this CSV.", None

        try:
            visits = fetch_port_visits(vessel_ids, start, end, api_key)
        except Exception as e:
            return _build_map(None), _table(None), "", f"GFW error: {str(e)[:80]}", None

        agg = aggregate_by_port(visits)
        summary = (f"{len(agg)} port(s), {int(agg['n_visits'].sum())} visit(s) "
                   f"from {len(vessel_ids)} vessel(s).") if not agg.empty else "No port visit found."
        status = "Done."
        # on stocke les visites detaillees pour l'export
        store = visits.to_dict("records") if visits is not None and not visits.empty else None
        return _build_map(agg), _table(agg), summary, status, store

    @app.callback(
        Output("port-download-csv", "data"),
        Input("port-btn-export", "n_clicks"),
        State("port-store", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        return dcc.send_data_frame(pd.DataFrame(store).to_csv, "port_visits.csv", index=False)