"""
pages/ports.py
==============
Page "Port" -- reproduces the old desktop app page: search a vessel
on GFW (name / MMSI / IMO), list found vessels,

Sidebar: search + results (radio) + dates + "Get port visits".
On the right (instead of the map): table of visits + export CSV.

The GFW functions (search_vessel, load_port_visits, generate_port_report)
included at the bottom of the file (copied from the old app).
"""

import asyncio
import tempfile
import os
from datetime import date

import dash
import pandas as pd
import gfwapiclient as gfw
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl
from config import YEARS, FLAG_NAMES
from gfw import get_gfw_client
from api_key import get_api_key

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)


# CALL SYNC GFW FUNCTIONS FROM THREAD (avoid blocking the Dash event loop) 
def do_search_vessel(query, api_key):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(search_vessel(query, client))
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def do_port_visits(vessel_ids, start, end, api_key):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(load_port_visits(vessel_ids, start, end, client))
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def _group_results_by_mmsi(df):
    """Renvoie une liste d'entrees {label, ids, name, owner, mmsi, imo, ...}."""
    entries = []
    if df is None or df.empty:
        return entries
    if "to" in df.columns:
        df = df.sort_values("to", ascending=False)

    seen = set()
    for _, row in df.iterrows():
        vid = row.get("vessel_id")
        mmsi = row.get("mmsi")
        if pd.isnull(vid) or pd.isnull(mmsi):
            continue
        if mmsi in seen:
            continue
        seen.add(mmsi)

        grp = df[df["mmsi"] == mmsi]
        ids = grp["vessel_id"].dropna().tolist()

        def first_valid(col):
            s = grp[col].dropna() if col in grp.columns else pd.Series(dtype=object)
            return s.iloc[0] if not s.empty else None

        name = row.get("ship_name") if pd.notnull(row.get("ship_name")) else "Unknown"
        flag = row.get("flag") if pd.notnull(row.get("flag")) else "?"
        owner = row.get("owner") if pd.notnull(row.get("owner")) else "Owner unknown"
        label = f"{name} | MMSI {mmsi} | {flag} | Owner: {owner}"
        entries.append({
            "label": label, "ids": ids, "name": name, "owner": owner,
            "mmsi": mmsi, "imo": first_valid("imo"),
            "vessel_type": first_valid("vessel_type"),
            "gear_type": first_valid("gear_type"),
            "length_m": first_valid("length_m"),
        })
    return entries


# LAYOUT
def layout():
    return html.Div([
        dcc.Store(id="port-search-store", data=None),
        dcc.Store(id="port-store", data=None),
        dcc.Download(id="port-download-csv"),

        html.Div([
            html.H6("Port visits lookup", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P("Search a vessel by name, MMSI or IMO, then retrieve every port "
                   "it visited during the selected period.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Vessel name / MMSI / IMO"),
            dcc.Input(id="port-query", type="text", placeholder="Vessel name / MMSI / IMO",
                debounce=True,
                style={"width": "100%", "padding": "0.4rem", "marginBottom": "0.5rem",
                       "borderRadius": "5px", "border": "1px solid " + BDR,
                       "background": PANEL, "color": MAIN}),
            html.Button("Search", id="port-btn-search", n_clicks=0,
                style={"width": "100%", "padding": "0.45rem",
                       "background": "linear-gradient(135deg," + ACC + ",#0d4a7a)",
                       "color": "white", "border": "none", "borderRadius": "6px",
                       "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.8rem"}),

            lbl("Vessels found"),
            dcc.Loading(type="dot", color=ACC,
                children=html.Div(
                    dcc.RadioItems(id="port-vessel-selector", options=[], value=None,
                        labelStyle={"display": "block", "marginBottom": "5px",
                                    "fontSize": "0.7rem", "color": SOFT, "cursor": "pointer"}),
                    style={"maxHeight": "200px", "overflowY": "auto",
                           "border": "1px solid " + BDR, "borderRadius": "6px",
                           "padding": "0.5rem", "marginBottom": "0.4rem", "background": BG},
                )),
            html.Div(id="port-selected", style={"fontSize": "0.72rem", "color": ACC,
                                                "fontWeight": "600", "marginBottom": "0.8rem"}),

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

            html.Button("Get port visits", id="port-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": "linear-gradient(135deg,#d15400,#a03e00)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            html.Div(id="port-status", style={"fontSize": "0.72rem", "color": SOFT}),
            html.Div(id="port-summary", style={"fontSize": "0.75rem", "color": SOFT, "marginTop": "0.4rem"}),

        ], style={"width": "310px", "minWidth": "310px", "padding": "1rem",
                   "background": BG, "borderRight": "1px solid " + BDR,
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        html.Div([
            html.Div(
                html.Button("Export CSV", id="port-btn-export", n_clicks=0,
                    style={"border": "none",
                           "background": "linear-gradient(135deg," + ACC + ",#0d4a7a)",
                           "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                           "fontWeight": "600", "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": "1px solid " + BDR, "flexShrink": "0",
                       "display": "flex", "justifyContent": "flex-end"},
            ),
            html.Div(
                dcc.Loading(type="circle", color=ACC,
                    children=html.Div(id="port-table",
                        children=html.P("Search a vessel, select it, choose dates, then click Get port visits.",
                                        style={"color": DIM, "fontSize": "0.8rem"}))),
                style={"flex": "1", "minHeight": 0, "overflowY": "auto", "padding": "1rem"},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _table(visits):
    if visits is None or visits.empty:
        return html.P("No port visit found for this vessel/period.",
                      style={"color": SOFT, "fontSize": "0.8rem"})
    show = visits.copy()
    if "port_flag" in show.columns:
        show["port_flag"] = show["port_flag"].map(
            lambda f: FLAG_NAMES.get(f, f) if pd.notna(f) else "?")
    cols = ["start", "end", "port_name", "port_flag", "duration_hrs", "confidence"]
    cols = [c for c in cols if c in show.columns]
    return dash_table.DataTable(
        data=show[cols].to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        sort_action="native", filter_action="native", page_size=30,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": "1px solid " + BDR,
                    "fontSize": "0.75rem", "padding": "4px 8px"},
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
    )

# CALLBACKS
def register_callbacks(app):

    @app.callback(
        Output("port-vessel-selector", "options"),
        Output("port-vessel-selector", "value"),
        Output("port-search-store", "data"),
        Output("port-status", "children"),
        Input("port-btn-search", "n_clicks"),
        State("port-query", "value"),
        prevent_initial_call=True,
    )
    def _search(n, query):
        if not n:
            raise dash.exceptions.PreventUpdate
        api_key = get_api_key()
        if not api_key:
            return [], None, None, "No API key saved."
        if not query or not str(query).strip():
            return [], None, None, "Enter a name, MMSI or IMO first."
        try:
            df = do_search_vessel(str(query).strip(), api_key)
        except Exception as e:
            return [], None, None, "Search failed: " + str(e)[:70]

        entries = _group_results_by_mmsi(df)
        if not entries:
            return [], None, None, "No vessel found."
        opts = [{"label": e["label"], "value": str(i)} for i, e in enumerate(entries)]
        return opts, None, entries, f"{len(entries)} vessel(s) found."

    @app.callback(
        Output("port-selected", "children"),
        Input("port-vessel-selector", "value"),
        State("port-search-store", "data"),
        prevent_initial_call=True,
    )
    def _selected(idx, entries):
        if idx is None or not entries:
            return ""
        info = entries[int(idx)]
        return f"Selected: {info['label']}"

    @app.callback(
        Output("port-table", "children"),
        Output("port-summary", "children"),
        Output("port-store", "data"),
        Input("port-btn-run", "n_clicks"),
        State("port-vessel-selector", "value"),
        State("port-search-store", "data"),
        State("port-start", "date"),
        State("port-end", "date"),
        prevent_initial_call=True,
    )
    def _run(n, idx, entries, start, end):
        if not n:
            raise dash.exceptions.PreventUpdate
        api_key = get_api_key()
        if not api_key:
            return _table(None), "No API key saved.", None
        if idx is None or not entries:
            return _table(None), "Select a vessel first.", None

        info = entries[int(idx)]
        try:
            visits = do_port_visits(info["ids"], start, end, api_key)
        except Exception as e:
            return _table(None), "GFW error: " + str(e)[:80], None
        if visits is None or visits.empty:
            return _table(None), "No port visit for this vessel/period.", None

        vis = visits.copy()
        vis.insert(0, "length_m", info.get("length_m") or "Unknown")
        vis.insert(0, "gear_type", info.get("gear_type") or "Unknown")
        vis.insert(0, "vessel_type", info.get("vessel_type") or "Unknown")
        vis.insert(0, "imo", info.get("imo") or "Unknown")
        vis.insert(0, "mmsi", info.get("mmsi") or "Unknown")
        vis.insert(0, "owner", info.get("owner") or "Unknown")
        vis.insert(0, "ship_name", info.get("name") or "Unknown")

        n_ports = visits["port_name"].nunique() if "port_name" in visits.columns else 0
        summary = f"{len(visits)} visit(s) across {n_ports} distinct port(s)."
        return _table(visits), summary, vis.to_dict("records")

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


# GFW FONCTIONS (async)
async def search_vessel(query, client):
    result = await client.vessels.search_vessels(
        query=query,
        datasets=["public-global-vessel-identity:latest"],
        includes=["OWNERSHIP", "MATCH_CRITERIA"],
    )
    df = result.df()
    if df.empty:
        return df

    def _d(x):
        """Pydantic -> dict si besoin."""
        if hasattr(x, "model_dump"):
            return x.model_dump()
        return x if isinstance(x, dict) else {}

    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()

        owners = d.get("registry_owners") or []
        owner_name = _d(owners[-1]).get("name") if owners else None

        registry = d.get("registry_info") or []
        reg = _d(registry[-1]) if registry else {}
        reg_imo = reg.get("imo")
        reg_length = reg.get("length_m")
        reg_tonnage = reg.get("tonnage_gt")

        gear_by_vid, ship_by_vid = {}, {}
        any_gear = any_ship = None

        for c in (d.get("combined_sources_info") or []):
            c = _d(c)
            vid = c.get("vessel_id")
            gears = c.get("gear_types") or []
            ships = c.get("ship_types") or []
            if gears:
                g = _d(gears[-1]).get("name")
                gear_by_vid[vid] = g
                any_gear = any_gear or g
            if ships:
                s = _d(ships[-1]).get("name")
                ship_by_vid[vid] = s
                any_ship = any_ship or s

        for i in (d.get("self_reported_info") or []):
            i = _d(i)
            vid = i.get("id")
            rows.append({
                "vessel_id": vid,
                "ship_name": i.get("ship_name"),
                "mmsi": i.get("ssvid"),
                "imo": i.get("imo") or reg_imo,
                "call_sign": i.get("call_sign"),
                "flag": i.get("flag"),
                "vessel_type": ship_by_vid.get(vid),
                "gear_type": gear_by_vid.get(vid),
                "length_m": reg_length,
                "tonnage_gt": reg_tonnage,
                "owner": owner_name,
                "from": i.get("transmission_date_from"),
                "to": i.get("transmission_date_to"),
            })

    out = pd.DataFrame(rows)
    out = out[out["vessel_id"].notna()].drop_duplicates(subset=["vessel_id"])
    return out


async def load_port_visits(vessel_ids, start, end, client):
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

    # --- FILTER 1 : drop corrupted visits (exit never detected) ---
    # If a visit ends after the next one starts, its end date is wrong.
    corrupted = df["_end"] > df["_start"].shift(-1)
    df = df[~corrupted]
    if df.empty:
        return df

    # --- FILTER 2 : keep only visits overlapping the requested window ---
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


def generate_port_report(vessel_ids, vessel_name, start, end, client,
                         owner=None, mmsi=None, imo=None,
                         vessel_type=None, gear_type=None, length_m=None,
                         progress_callback=None):
    if progress_callback:
        progress_callback("Fetching port visits...", 0.3)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    df = loop.run_until_complete(
        load_port_visits(vessel_ids, start, end, client)
    )

    if progress_callback:
        progress_callback("Writing report...", 0.8)

    if df.empty:
        raise ValueError("No port visits found for this vessel/period.")

    df.insert(0, "length_m", length_m if length_m else "Unknown")
    df.insert(0, "gear_type", gear_type if gear_type else "Unknown")
    df.insert(0, "vessel_type", vessel_type if vessel_type else "Unknown")
    df.insert(0, "imo", imo if imo else "Unknown")
    df.insert(0, "mmsi", mmsi if mmsi else "Unknown")
    df.insert(0, "owner", owner if owner else "Unknown")
    df.insert(0, "ship_name", vessel_name)

    safe = "".join(c for c in str(vessel_name) if c.isalnum() or c in " _-").strip()

    tmp_dir = tempfile.gettempdir()
    out = os.path.join(tmp_dir, f"PORT_visits_{safe}_{start}-{end}.csv")

    df.to_csv(out, index=False)

    n_ports = df["port_name"].nunique()
    return out, len(df), n_ports