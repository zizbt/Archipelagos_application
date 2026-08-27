"""
pages/vessel_report.py
=======================
Page "Vessel Report" -- recherche un navire (nom / MMSI / IMO), le
selectionne, choisit une periode, puis genere un rapport complet :

  - Port visits    (public-global-port-visits-events:latest)
  - AIS gaps        (public-global-gaps-events:latest)
  - Fishing events  (public-global-fishing-events:latest)
  - Loitering events(public-global-loitering-events:latest)
  - Encounters      (public-global-encounters-events:latest)

Chaque section est recuperee independamment (5 appels GFW separes) : si un
dataset echoue (nom incorrect, non disponible sur la cle...), seule cette
section affiche l'erreur -- les autres s'affichent normalement.

Les champs de chaque type d'evenement sont extraits dynamiquement (pas de
noms de colonnes codes en dur au-dela de start/end/lat/lon) via un
"flatten" generique du sous-objet pydantic renvoye par l'API, pour rester
robuste si le schema exact differe de ce qui est suppose ici.
"""

import asyncio
from datetime import date

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl, card
from config import YEARS
from gfw import get_gfw_client
from api_key import get_api_key

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

EVENT_DATASETS = {
    "port_visit": "public-global-port-visits-events:latest",
    "gap":        "public-global-gaps-events:latest",
    "fishing":    "public-global-fishing-events:latest",
    "loitering":  "public-global-loitering-events:latest",
    "encounter":  "public-global-encounters-events:latest",
}

SECTION_LABELS = {
    "port_visit": "Port visits",
    "gap":        "AIS gaps",
    "fishing":    "Fishing events",
    "loitering":  "Loitering events",
    "encounter":  "Encounters",
}


# ── Appel sync GFW (evite de bloquer la boucle Dash) ────────────────────────

def do_search_vessel(query, api_key):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(search_vessel(query, client))
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def do_full_report(vessel_ids, start, end, api_key):
    """Renvoie un dict {event_key: (df, error_or_None)} pour les 5 types."""
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = {}
        for key, dataset in EVENT_DATASETS.items():
            try:
                df = loop.run_until_complete(
                    load_events(dataset, key, vessel_ids, start, end, client))
                results[key] = (df, None)
            except Exception as e:
                results[key] = (pd.DataFrame(), str(e)[:120])
    finally:
        loop.close()
    return results


def _group_results_by_mmsi(df):
    """Identique aux autres pages (ports.py / ais_gap.py)."""
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
            "mmsi": mmsi, "imo": first_valid("imo"), "flag": flag,
            "vessel_type": first_valid("vessel_type"),
            "gear_type": first_valid("gear_type"),
            "length_m": first_valid("length_m"),
        })
    return entries


# ── LAYOUT ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="vr-search-store", data=None),
        dcc.Store(id="vr-report-store", data=None),
        dcc.Download(id="vr-download-csv"),

        html.Div([
            html.H6("Vessel report", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P("Search a vessel, select it, choose a period, and get a full "
                   "activity report: port visits, AIS gaps, fishing, loitering "
                   "and encounters.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Vessel name / MMSI / IMO"),
            dcc.Input(id="vr-query", type="text", placeholder="Vessel name / MMSI / IMO",
                debounce=True,
                style={"width": "100%", "padding": "0.4rem", "marginBottom": "0.5rem",
                       "borderRadius": "5px", "border": "1px solid " + BDR,
                       "background": PANEL, "color": MAIN}),
            html.Button("Search", id="vr-btn-search", n_clicks=0,
                style={"width": "100%", "padding": "0.45rem",
                       "background": "linear-gradient(135deg," + ACC + ",#0d4a7a)",
                       "color": "white", "border": "none", "borderRadius": "6px",
                       "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.8rem"}),

            lbl("Vessels found"),
            dcc.Loading(type="dot", color=ACC,
                children=html.Div(
                    dcc.RadioItems(id="vr-vessel-selector", options=[], value=None,
                        labelStyle={"display": "block", "marginBottom": "5px",
                                    "fontSize": "0.7rem", "color": SOFT, "cursor": "pointer"}),
                    style={"maxHeight": "200px", "overflowY": "auto",
                           "border": "1px solid " + BDR, "borderRadius": "6px",
                           "padding": "0.5rem", "marginBottom": "0.4rem", "background": BG},
                )),
            html.Div(id="vr-selected", style={"fontSize": "0.72rem", "color": ACC,
                                               "fontWeight": "600", "marginBottom": "0.8rem"}),

            lbl("Start date"),
            dcc.DatePickerSingle(id="vr-start", date=date(YEARS[-1], 1, 1),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="vr-end", date=date(YEARS[-1], 12, 31),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "1rem"}),

            html.Button("Generate Report", id="vr-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": "linear-gradient(135deg,#d15400,#a03e00)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            html.Div(id="vr-status", style={"fontSize": "0.72rem", "color": SOFT}),

        # Pas de hauteur figee / overflowY force ici (ca dependait d'une
        # hauteur de nav devinee et fausse). "position: sticky" fait rester
        # la sidebar visible pendant le scroll de la page, sans avoir besoin
        # de connaitre la hauteur exacte du bandeau au-dessus.
        ], style={"width": "320px", "minWidth": "320px", "padding": "1rem",
                   "background": BG, "borderRight": "1px solid " + BDR,
                   "flexShrink": "0", "position": "sticky", "top": "0",
                   "alignSelf": "flex-start", "maxHeight": "100vh", "overflowY": "auto"}),

        # minWidth:0 est essentiel : sans lui, un flex-item refuse de retrecir
        # en dessous de la largeur de son contenu. Le tableau (beaucoup de
        # colonnes) forcait donc TOUTE la ligne (sidebar comprise) a deborder
        # horizontalement au lieu de rester dans son propre scroll interne.
        html.Div([
            html.Div(
                html.Button("Export full CSV", id="vr-btn-export", n_clicks=0,
                    style={"border": "none",
                           "background": "linear-gradient(135deg," + ACC + ",#0d4a7a)",
                           "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                           "fontWeight": "600", "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": "1px solid " + BDR, "flexShrink": "0",
                       "display": "flex", "justifyContent": "flex-end", "position": "sticky",
                       "top": "0", "zIndex": "5"},
            ),
            html.Div(
                dcc.Loading(type="circle", color=ACC,
                    children=html.Div(id="vr-report",
                        children=html.P("Search a vessel, select it, choose dates, "
                                        "then click Generate Report.",
                                        style={"color": DIM, "fontSize": "0.8rem"}))),
                style={"padding": "1rem", "minWidth": "0", "overflowX": "auto"},
            ),
        ], style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "alignItems": "flex-start"})


# ── Rendu du rapport ─────────────────────────────────────────────────────────

def _pick_display_cols(df, extra_keywords=("duration", "distance", "speed", "km", "hour", "confidence")):
    """Choisit un sous-ensemble de colonnes lisibles pour l'affichage,
    sans presumer des noms exacts au-dela de start/end/lat/lon.

    Exclut les colonnes qui contiennent en realite un blob JSON (ex:
    "..._distances" est une LISTE de dicts que json_normalize ne peut
    pas aplatir plus loin ; _sanitize_for_display la convertit en texte
    JSON tres long, ce qui rend la colonne illisible et fait deborder
    le tableau horizontalement bien au-dela de l'ecran)."""
    priority = [c for c in ("start", "end", "lat", "lon") if c in df.columns]
    others = []
    for c in df.columns:
        if c in priority:
            continue
        if not any(k in c.lower() for k in extra_keywords):
            continue
        sample = df[c].dropna().astype(str)
        if not sample.empty and sample.str.len().mean() > 40:
            continue  # colonne JSON/liste brute -> on ne l'affiche pas
        others.append(c)
    cols = priority + others
    return cols if cols else list(df.columns)[:8]


def _section_table(key, df, error):
    label = SECTION_LABELS[key]
    if error:
        return html.Div([
            html.H6(label, style={"color": MAIN, "fontSize": "0.85rem", "marginBottom": "0.3rem"}),
            html.P("Error: " + error, style={"color": "#e07070", "fontSize": "0.72rem"}),
        ], style={"marginBottom": "1.2rem"})

    if df is None or df.empty:
        return html.Div([
            html.H6(label, style={"color": MAIN, "fontSize": "0.85rem", "marginBottom": "0.3rem"}),
            html.P("None found for this period.", style={"color": SOFT, "fontSize": "0.75rem"}),
        ], style={"marginBottom": "1.2rem"})

    cols = _pick_display_cols(df)
    table = dash_table.DataTable(
        data=df[cols].to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        sort_action="native", filter_action="native", page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": "1px solid " + BDR,
                    "fontSize": "0.73rem", "padding": "4px 8px",
                    "maxWidth": "260px", "overflow": "hidden", "textOverflow": "ellipsis"},
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
    )
    return html.Div([
        html.H6(f"{label} ({len(df)})", style={"color": MAIN, "fontSize": "0.85rem", "marginBottom": "0.3rem"}),
        table,
    ], style={"marginBottom": "1.2rem", "minWidth": "0"})


def _summary_bar(info, start, end, results):
    def count(key):
        df, err = results.get(key, (pd.DataFrame(), None))
        return "err" if err else len(df)

    stats = [
        ("Port visits", count("port_visit")),
        ("AIS gaps", count("gap")),
        ("Fishing events", count("fishing")),
        ("Loitering events", count("loitering")),
        ("Encounters", count("encounter")),
    ]
    boxes = [
        html.Div([
            html.Div(str(v), style={"fontSize": "1.3rem", "fontWeight": "700",
                                     "color": ACC if v != "err" else "#e07070"}),
            html.Div(k, style={"fontSize": "0.68rem", "color": SOFT}),
        ], style={"textAlign": "center", "flex": "1", "minWidth": "100px"})
        for k, v in stats
    ]
    return html.Div(card([
        html.H5(info["name"], style={"color": MAIN, "marginBottom": "0.1rem"}),
        html.P(f"MMSI {info['mmsi']} | IMO {info.get('imo') or '?'} | "
               f"Flag {info.get('flag') or '?'} | {start} -> {end}",
               style={"color": DIM, "fontSize": "0.75rem", "marginBottom": "0.8rem"}),
        html.Div(boxes, style={"display": "flex", "gap": "0.5rem", "flexWrap": "wrap"}),
    ]))


def _render_report(info, start, end, results):
    return html.Div([
        _summary_bar(info, start, end, results),
        html.Div([_section_table(key, *results[key]) for key in EVENT_DATASETS],
                 style={"minWidth": "0"}),
    ], style={"minWidth": "0"})


# ── CALLBACKS ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("vr-vessel-selector", "options"),
        Output("vr-vessel-selector", "value"),
        Output("vr-search-store", "data"),
        Output("vr-status", "children"),
        Input("vr-btn-search", "n_clicks"),
        State("vr-query", "value"),
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
        Output("vr-selected", "children"),
        Input("vr-vessel-selector", "value"),
        State("vr-search-store", "data"),
        prevent_initial_call=True,
    )
    def _selected(idx, entries):
        if idx is None or not entries:
            return ""
        info = entries[int(idx)]
        return f"Selected: {info['label']}"

    @app.callback(
        Output("vr-report", "children"),
        Output("vr-status", "children", allow_duplicate=True),
        Output("vr-report-store", "data"),
        Input("vr-btn-run", "n_clicks"),
        State("vr-vessel-selector", "value"),
        State("vr-search-store", "data"),
        State("vr-start", "date"),
        State("vr-end", "date"),
        prevent_initial_call=True,
    )
    def _run(n, idx, entries, start, end):
        if not n:
            raise dash.exceptions.PreventUpdate
        api_key = get_api_key()
        if not api_key:
            return dash.no_update, "No API key saved.", None
        if idx is None or not entries:
            return dash.no_update, "Select a vessel first.", None
        if not start or not end:
            return dash.no_update, "Please choose a start and end date.", None

        info = entries[int(idx)]
        results = do_full_report(info["ids"], start, end, api_key)

        # Store combine pour l'export : chaque df avec une colonne event_type
        combined_frames = []
        for key, (df, err) in results.items():
            if err or df is None or df.empty:
                continue
            d = df.copy()
            d.insert(0, "event_type", SECTION_LABELS[key])
            combined_frames.append(d)
        store = (pd.concat(combined_frames, ignore_index=True, sort=False)
                 .to_dict("records")) if combined_frames else None

        n_errors = sum(1 for _, (_, e) in results.items() if e)
        status = (f"Report generated ({n_errors} section(s) failed)."
                  if n_errors else "Report generated.")

        return _render_report(info, start, end, results), status, store

    @app.callback(
        Output("vr-download-csv", "data"),
        Input("vr-btn-export", "n_clicks"),
        State("vr-report-store", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        return dcc.send_data_frame(pd.DataFrame(store).to_csv, "vessel_report.csv", index=False)


# ── GFW FUNCTIONS (async) ────────────────────────────────────────────────────

async def search_vessel(query, client):
    """Identique aux autres pages -- recherche de navire GFW."""
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


def _flatten_generic(df, type_hint):
    """
    Flatten generique du sous-objet specifique a l'evenement (ex: colonne
    'fishing', 'loitering', 'encounter', 'port_visit' ou 'gap'), sans
    presumer des noms de champs exacts -- json_normalize expose tout ce que
    l'API renvoie, prefixe par type_hint.
    """
    def _d(x):
        if hasattr(x, "model_dump"):
            return x.model_dump()
        return x if isinstance(x, dict) else {}

    candidates = [c for c in df.columns if c not in ("start", "end", "lat", "lon", "id", "type")
                  and df[c].apply(lambda x: hasattr(x, "model_dump") or isinstance(x, dict)).any()]

    sub_col = None
    for c in candidates:
        if type_hint.replace("_", "") in c.lower().replace("_", ""):
            sub_col = c
            break
    if sub_col is None and candidates:
        sub_col = candidates[0]

    if sub_col:
        flat = pd.json_normalize(df[sub_col].apply(_d))
        flat.index = df.index
        flat = flat.add_prefix(f"{type_hint}_")
        df = pd.concat([df.drop(columns=[sub_col]), flat], axis=1)

    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    return df


def _sanitize_for_display(df):
    """
    Certains champs GFW imbriquent un dict a l'interieur d'une LISTE
    (ex: 'distances': [{...}]). json_normalize aplatit les dicts directs
    mais pas les dicts caches dans une liste -- ces cellules restent des
    objets Python bruts, que React/Dash ne peuvent pas afficher tels quels
    (erreur React #31 "object with keys ..."). On les convertit en texte
    JSON lisible avant tout affichage ou export.
    """
    import json as _json
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].apply(
                lambda v: _json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    return df


async def load_events(dataset, type_hint, vessel_ids, start, end, client):
    """Charge un type d'evenement GFW pour un ou plusieurs vessel_id."""
    if isinstance(vessel_ids, str):
        vessel_ids = [vessel_ids]

    events = await client.events.get_all_events(
        datasets=[dataset],
        vessels=vessel_ids,
        start_date=start,
        end_date=end,
        limit=99999,
    )

    df = events.df()
    if df.empty:
        return df

    df = _flatten_generic(df.copy(), type_hint)
    df = _sanitize_for_display(df)
    if "start" in df.columns:
        df = df.sort_values("start")
    return df