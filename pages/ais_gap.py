"""
pages/ais_gap.py
=================
Page "AIS Gaps" -- deux modes :

  - SINGLE VESSEL : recherche un navire (nom / MMSI / IMO), on le
    selectionne dans la liste (radio), comme sur pages/ports.py.

  - ALL VESSELS   : pas de recherche nominative -- on filtre directement
    par pavillon(s) et/ou type(s) de navire / engin de peche (ex:
    trawlers uniquement) sur la zone Egee, comme le fait la page Data
    pour le Vessel Presence.

Dans les deux cas, une periode (start/end) est obligatoire.

A droite : tableau des gaps AIS + export CSV.
"""

import asyncio
from datetime import date

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl
from config import YEARS, FLAG_NAMES
from gfw import get_gfw_client, AEGEAN_GEOJSON, GFW_VESSEL_TYPES, GEAR_TYPES, COUNTRY_FLAGS
from api_key import get_api_key

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)


# ── Appels sync GFW depuis le callback (evite de bloquer la boucle Dash) ───────

def do_search_vessel(query, api_key):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(search_vessel(query, client))
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def do_ais_gaps_single(vessel_ids, start, end, api_key):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(load_ais_gaps(vessel_ids, start, end, client))
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def do_ais_gaps_bulk(flags, vessel_types, gear_types, start, end, api_key):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(
            load_ais_gaps_bulk(flags, vessel_types, gear_types, start, end, client))
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def _clean_identity_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _identity_key(row):
    for field in ("imo", "mmsi", "vessel_id"):
        value = _clean_identity_value(row.get(field))
        if value:
            return f"{field}:{value}"
    return None


def _group_results_by_identity(df):
    """Renvoie une liste d'entrees {label, ids, name, owner, mmsi, imo, ...}."""
    entries = []
    if df is None or df.empty:
        return entries
    if "to" in df.columns:
        df = df.sort_values("to", ascending=False)

    work = df.copy()
    work["_identity_key"] = work.apply(_identity_key, axis=1)
    work = work[work["_identity_key"].notna()]

    for _, grp in work.groupby("_identity_key", sort=False):
        row = grp.iloc[0]
        ids = grp["vessel_id"].dropna().astype(str).unique().tolist() if "vessel_id" in grp.columns else []
        mmsi_values = grp["mmsi"].dropna().astype(str).unique().tolist() if "mmsi" in grp.columns else []

        def first_valid(col):
            s = grp[col].dropna() if col in grp.columns else pd.Series(dtype=object)
            return s.iloc[0] if not s.empty else None

        name = row.get("ship_name") if pd.notnull(row.get("ship_name")) else "Unknown"
        flag = row.get("flag") if pd.notnull(row.get("flag")) else "?"
        owner = row.get("owner") if pd.notnull(row.get("owner")) else "Owner unknown"
        imo = first_valid("imo")
        mmsi = ", ".join(mmsi_values) if mmsi_values else "?"
        label = f"{name} | IMO {imo or '?'} | MMSI {mmsi} | {flag} | Owner: {owner}"
        entries.append({
            "label": label, "ids": ids, "name": name, "owner": owner,
            "mmsi": mmsi_values[0] if mmsi_values else None, "mmsi_values": mmsi_values,
            "imo": imo,
            "vessel_type": first_valid("vessel_type"),
            "gear_type": first_valid("gear_type"),
            "length_m": first_valid("length_m"),
            "call_sign": first_valid("call_sign"),
        })
    return entries


# ── LAYOUT ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="gap-search-store", data=None),
        dcc.Store(id="gap-store", data=None),
        dcc.Download(id="gap-download-csv"),

        html.Div([
            html.H6("AIS gaps lookup", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P("Look up AIS transmission gaps for one vessel, or for every "
                   "vessel matching a flag / type filter, over a chosen period.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Mode"),
            dcc.RadioItems(
                id="gap-mode",
                options=[
                    {"label": " Single vessel", "value": "single"},
                    {"label": " All vessels (filter by flag / type)", "value": "bulk"},
                ],
                value="single",
                labelStyle={"display": "block", "marginBottom": "4px",
                            "fontSize": "0.75rem", "color": SOFT, "cursor": "pointer"},
                style={"marginBottom": "1rem"},
            ),

            # ── Panneau mode SINGLE ──
            html.Div(id="gap-panel-single", children=[
                lbl("Vessel name / MMSI / IMO"),
                dcc.Input(id="gap-query", type="text", placeholder="Vessel name / MMSI / IMO",
                    debounce=True,
                    style={"width": "100%", "padding": "0.4rem", "marginBottom": "0.5rem",
                           "borderRadius": "5px", "border": "1px solid " + BDR,
                           "background": PANEL, "color": MAIN}),
                html.Button("Search", id="gap-btn-search", n_clicks=0,
                    style={"width": "100%", "padding": "0.45rem",
                           "background": "linear-gradient(135deg," + ACC + ",#0d4a7a)",
                           "color": "white", "border": "none", "borderRadius": "6px",
                           "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.8rem"}),

                lbl("Vessels found"),
                dcc.Loading(type="dot", color=ACC,
                    children=html.Div(
                        dcc.RadioItems(id="gap-vessel-selector", options=[], value=None,
                            labelStyle={"display": "block", "marginBottom": "5px",
                                        "fontSize": "0.7rem", "color": SOFT, "cursor": "pointer"}),
                        style={"maxHeight": "200px", "overflowY": "auto",
                               "border": "1px solid " + BDR, "borderRadius": "6px",
                               "padding": "0.5rem", "marginBottom": "0.4rem", "background": BG},
                    )),
                html.Div(id="gap-selected", style={"fontSize": "0.72rem", "color": ACC,
                                                    "fontWeight": "600", "marginBottom": "0.8rem"}),
            ]),

            # ── Panneau mode BULK ──
            html.Div(id="gap-panel-bulk", children=[
                lbl("Country / Flag"),
                dcc.Dropdown(id="gap-bulk-flags",
                    options=[{"label": "ALL countries", "value": "ALL"}] +
                            [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f}
                             for f in COUNTRY_FLAGS],
                    value=["GRC"], multi=True, placeholder="Select countries...",
                    style={"color": "#000", "marginBottom": "0.8rem"}),

                lbl("Vessel type (leave empty for ALL)"),
                dcc.Dropdown(id="gap-bulk-vtypes",
                    options=[{"label": t.capitalize(), "value": t} for t in GFW_VESSEL_TYPES],
                    value=[], multi=True, placeholder="All types...",
                    style={"color": "#000", "marginBottom": "0.8rem"}),

                lbl("Gear type (e.g. trawlers only)"),
                dcc.Dropdown(id="gap-bulk-gear",
                    options=[{"label": g.replace("_", " ").title(), "value": g} for g in GEAR_TYPES],
                    value=[], multi=True, placeholder="All gear types...",
                    style={"color": "#000", "marginBottom": "0.4rem"}),
                html.P("Gear type is applied after download (not all vessels report it).",
                       style={"fontSize": "0.65rem", "color": DIM, "marginBottom": "0.8rem"}),
            ], style={"display": "none"}),

            lbl("Start date"),
            dcc.DatePickerSingle(id="gap-start", date=date(YEARS[-1], 1, 1),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="gap-end", date=date(YEARS[-1], 12, 31),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "1rem"}),

            html.Button("Get AIS gaps", id="gap-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": "linear-gradient(135deg,#d15400,#a03e00)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            html.Div(id="gap-status", style={"fontSize": "0.72rem", "color": SOFT}),
            html.Div(id="gap-summary", style={"fontSize": "0.75rem", "color": SOFT, "marginTop": "0.4rem"}),

        ], style={"width": "330px", "minWidth": "330px", "padding": "1rem",
                   "background": BG, "borderRight": "1px solid " + BDR,
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        html.Div([
            html.Div(
                html.Button("Export CSV", id="gap-btn-export", n_clicks=0,
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
                    children=html.Div(id="gap-table",
                        children=html.P("Choose a mode, select a vessel or filters, "
                                        "pick dates, then click Get AIS gaps.",
                                        style={"color": DIM, "fontSize": "0.8rem"}))),
                style={"flex": "1", "minHeight": 0, "overflowY": "auto", "padding": "1rem"},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _table(gaps):
    if gaps is None or gaps.empty:
        return html.P("No AIS gap found for this selection/period.",
                      style={"color": SOFT, "fontSize": "0.8rem"})
    show = gaps.copy()
    cols = ["ship_name", "mmsi", "flag", "vessel_type", "gear_type",
            "start", "end", "duration_hrs", "distance_km",
            "off_lat", "off_lon", "on_lat", "on_lon"]
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


# ── CALLBACKS ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("gap-panel-single", "style"),
        Output("gap-panel-bulk", "style"),
        Input("gap-mode", "value"),
    )
    def _toggle_mode(mode):
        if mode == "bulk":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("gap-vessel-selector", "options"),
        Output("gap-vessel-selector", "value"),
        Output("gap-search-store", "data"),
        Output("gap-status", "children"),
        Input("gap-btn-search", "n_clicks"),
        State("gap-query", "value"),
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

        entries = _group_results_by_identity(df)
        if not entries:
            return [], None, None, "No vessel found."
        opts = [{"label": e["label"], "value": str(i)} for i, e in enumerate(entries)]
        return opts, None, entries, f"{len(entries)} vessel(s) found."

    @app.callback(
        Output("gap-selected", "children"),
        Input("gap-vessel-selector", "value"),
        State("gap-search-store", "data"),
        prevent_initial_call=True,
    )
    def _selected(idx, entries):
        if idx is None or not entries:
            return ""
        info = entries[int(idx)]
        return f"Selected: {info['label']}"

    @app.callback(
        Output("gap-table", "children"),
        Output("gap-summary", "children"),
        Output("gap-store", "data"),
        Input("gap-btn-run", "n_clicks"),
        State("gap-mode", "value"),
        State("gap-vessel-selector", "value"),
        State("gap-search-store", "data"),
        State("gap-bulk-flags", "value"),
        State("gap-bulk-vtypes", "value"),
        State("gap-bulk-gear", "value"),
        State("gap-start", "date"),
        State("gap-end", "date"),
        prevent_initial_call=True,
    )
    def _run(n, mode, idx, entries, bulk_flags, bulk_vtypes, bulk_gear, start, end):
        if not n:
            raise dash.exceptions.PreventUpdate
        api_key = get_api_key()
        if not api_key:
            return _table(None), "No API key saved.", None
        if not start or not end:
            return _table(None), "Please choose a start and end date.", None

        # ── Mode SINGLE ──
        if mode == "single":
            if idx is None or not entries:
                return _table(None), "Select a vessel first.", None
            info = entries[int(idx)]
            try:
                gaps = do_ais_gaps_single(info["ids"], start, end, api_key)
            except Exception as e:
                return _table(None), "GFW error: " + str(e)[:80], None
            if gaps is None or gaps.empty:
                return _table(None), "No AIS gap for this vessel/period.", None

            gaps.insert(0, "gear_type", info.get("gear_type") or "Unknown")
            gaps.insert(0, "vessel_type", info.get("vessel_type") or "Unknown")
            gaps.insert(0, "flag", "Unknown")
            gaps.insert(0, "mmsi", info.get("mmsi") or "Unknown")
            gaps.insert(0, "ship_name", info.get("name") or "Unknown")

            total_hrs = gaps["duration_hrs"].sum() if "duration_hrs" in gaps.columns else 0
            summary = f"{len(gaps)} gap(s) for {info['name']}, {total_hrs:,.1f}h total AIS off."
            return _table(gaps), summary, gaps.to_dict("records")

        # ── Mode BULK (tous les navires filtres par pavillon / type) ──
        resolved_flags = COUNTRY_FLAGS if (bulk_flags and "ALL" in bulk_flags) else (bulk_flags or None)
        try:
            gaps = do_ais_gaps_bulk(resolved_flags, bulk_vtypes, bulk_gear, start, end, api_key)
        except Exception as e:
            return _table(None), "GFW error: " + str(e)[:80], None
        if gaps is None or gaps.empty:
            return _table(None), "No AIS gap found for this filter/period.", None

        total_hrs = gaps["duration_hrs"].sum() if "duration_hrs" in gaps.columns else 0
        n_vessels = gaps["mmsi"].nunique() if "mmsi" in gaps.columns else len(gaps)
        summary = f"{len(gaps)} gap(s) across {n_vessels} vessel(s), {total_hrs:,.1f}h total AIS off."
        return _table(gaps), summary, gaps.to_dict("records")

    @app.callback(
        Output("gap-download-csv", "data"),
        Input("gap-btn-export", "n_clicks"),
        State("gap-store", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        return dcc.send_data_frame(pd.DataFrame(store).to_csv, "ais_gaps.csv", index=False)


# ── GFW FUNCTIONS (async) ────────────────────────────────────────────────────

async def search_vessel(query, client):
    """Identique a pages/ports.py -- recherche de navire GFW."""
    result = await client.vessels.search_vessels(
        query=query,
        datasets=["public-global-vessel-identity:latest"],
        includes=["OWNERSHIP", "MATCH_CRITERIA"],
    )
    df = result.df()
    if df.empty:
        return df

    def _d(x):
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

        for c in (d.get("combined_sources_info") or []):
            c = _d(c)
            vid = c.get("vessel_id")
            gears = c.get("gear_types") or []
            ships = c.get("ship_types") or []
            if gears:
                gear_by_vid[vid] = _d(gears[-1]).get("name")
            if ships:
                ship_by_vid[vid] = _d(ships[-1]).get("name")

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


def _flatten_gap_fields(df):
    """Ajoute duration_hrs / distance_km / off_lat / off_lon / on_lat / on_lon
    depuis le sous-objet 'gap' (ou en fallback depuis les colonnes a plat),
    et formatte start/end. Modifie et renvoie df."""

    def _d(x):
        if hasattr(x, "model_dump"):
            return x.model_dump()
        return x if isinstance(x, dict) else {}

    def _gap_field(g, key):
        g = _d(g)
        return g.get(key) if isinstance(g, dict) else None

    def _position(g, side, coord):
        g = _d(g)
        if not isinstance(g, dict):
            return None
        pos = g.get(f"{side}_position") or g.get(f"{side}Position") or {}
        pos = _d(pos)
        return pos.get(coord) if isinstance(pos, dict) else None

    gap_col = "gap" if "gap" in df.columns else None

    if gap_col:
        df["duration_hrs"] = df[gap_col].apply(lambda g: _gap_field(g, "duration_hrs"))
        df["distance_km"] = df[gap_col].apply(lambda g: _gap_field(g, "distance_km"))
        df["off_lat"] = df[gap_col].apply(lambda g: _position(g, "off", "lat"))
        df["off_lon"] = df[gap_col].apply(lambda g: _position(g, "off", "lon"))
        df["on_lat"] = df[gap_col].apply(lambda g: _position(g, "on", "lat"))
        df["on_lon"] = df[gap_col].apply(lambda g: _position(g, "on", "lon"))
    else:
        for col in ("duration_hrs", "distance_km"):
            if col not in df.columns:
                df[col] = None
        if "lat" in df.columns:
            df["off_lat"] = df["lat"]
        if "lon" in df.columns:
            df["off_lon"] = df["lon"]

    df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df["end"] = pd.to_datetime(df["end"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df["duration_hrs"] = pd.to_numeric(df["duration_hrs"], errors="coerce").round(1)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").round(1)
    return df


async def load_ais_gaps(vessel_ids, start, end, client):
    """Coupures AIS pour un ou plusieurs vessel_id precis (mode single)."""
    if isinstance(vessel_ids, str):
        vessel_ids = [vessel_ids]

    events = await client.events.get_all_events(
        datasets=["public-global-gaps-events:latest"],
        vessels=vessel_ids,
        start_date=start,
        end_date=end,
        limit=99999,
    )

    df = events.df()
    if df.empty:
        return df

    df = _flatten_gap_fields(df.copy())

    keep = ["start", "end", "duration_hrs", "distance_km",
            "off_lat", "off_lon", "on_lat", "on_lon"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values("start")


async def load_ais_gaps_bulk(flags, vessel_types, gear_types, start, end, client):
    """
    Coupures AIS pour TOUS les navires correspondant a des filtres
    pavillon(s) / type(s), sur la zone Egee (AEGEAN_GEOJSON), sans liste
    de vessel_id explicite -- meme logique geographique que load_VP_data
    dans gfw.py.

    NOTE : la syntaxe exacte des filtres acceptes par l'endpoint 'events'
    de GFW n'a pas pu etre verifiee ici (pas d'acces a la doc API en
    direct). Le code tente d'abord un filtre SQL-like ("flag IN (...) AND
    vessel_type IN (...)"), sur le meme modele que create_ais_presence_report
    (fourwings) utilise dans load_VP_data. Si l'API rejette ce parametre,
    l'erreur GFW exacte remontera dans gap-status -- a ajuster en fonction
    du message retourne.
    """
    filter_parts = []
    if flags:
        if isinstance(flags, list) and len(flags) > 1:
            flags_joined = ", ".join(f"'{f}'" for f in flags)
            filter_parts.append(f"flag IN ({flags_joined})")
        elif isinstance(flags, list) and flags:
            filter_parts.append(f"flag = '{flags[0]}'")

    if vessel_types:
        if len(vessel_types) == 1:
            filter_parts.append(f"vessel_type = '{vessel_types[0].lower()}'")
        else:
            vt = ", ".join(f"'{t.lower()}'" for t in vessel_types)
            filter_parts.append(f"vessel_type IN ({vt})")

    filter_string = " AND ".join(filter_parts)

    events = await client.events.get_all_events(
        datasets=["public-global-gaps-events:latest"],
        start_date=start,
        end_date=end,
        geojson=AEGEAN_GEOJSON,
        filters=[filter_string] if filter_string else [],
        limit=99999,
    )

    df = events.df()
    if df.empty:
        return df

    df = df.copy()
    df = _flatten_gap_fields(df)

    # Colonnes navire, si l'API les fournit a plat sur chaque evenement.
    rename_map = {"ship_name": "ship_name", "shipname": "ship_name",
                  "ssvid": "mmsi", "vessel_type": "vessel_type", "flag": "flag"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    if "gear_type" not in df.columns:
        df["gear_type"] = None

    # Filtre gear_type applique cote client (l'API ne le supporte pas au
    # telechargement, meme limitation que pour le Vessel Presence -- voir
    # commentaire GEAR_TYPES dans gfw.py).
    if gear_types:
        if "gear_type" in df.columns:
            df = df[df["gear_type"].isin(gear_types)]

    keep = ["ship_name", "mmsi", "flag", "vessel_type", "gear_type",
            "start", "end", "duration_hrs", "distance_km",
            "off_lat", "off_lon", "on_lat", "on_lon"]
    keep = [c for c in keep if c in df.columns]
    if not keep:
        return pd.DataFrame()
    return df[keep].sort_values("start")