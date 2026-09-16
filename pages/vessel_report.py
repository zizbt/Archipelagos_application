"""
pages/vessel_report.py
=======================
Page "Vessel Report" -- importe un CSV de positions AIS, liste les
navires trouves dedans, en selectionne un, puis genere un rapport :

  - Port visits     (detecte localement -- proximite d'un point de port
                      connu, voir port_zones.py, regroupee en visites)
  - AIS gaps         (detecte localement -- diff des positions
                      consecutives du navire, classe suspicious/normal
                      via le buffer de couverture AIS, meme methode que
                      pages/ais_gap.py / pages/alerts.py)
  - Loitering events (pages/loitering.get_loitering_dataframe)
  - Encounters       (pages/encounter.get_encounters_dataframe, sur
                      l'ensemble du CSV -- un "encounter" implique un
                      autre navire -- puis filtre sur celui selectionne)

v2 -- SINGLE data source now: an imported CSV, exactly like
pages/heatmap.py (drag & drop + server-side cache). No more live GFW
vessel search (name/MMSI/IMO) and no more date-range filter -- the
"vessels found" list comes from the imported CSV's own vessel_id/
ship_name columns, and the whole file is analyzed for the selected
vessel (no date pickers).

NOTE on "Fishing events": this section is INTENTIONALLY DROPPED in this
version. It existed before as GFW's own "public-global-fishing-events"
dataset (a proprietary speed/heading classification model run
server-side on GFW's data) -- there is no equivalent local algorithm in
this app to reproduce it from raw imported AIS positions, and making one
up would silently present unreliable results as if they were a real
detection. If you need fishing-event detection, that still requires a
live GFW call (see the old GFW-search-based version of this page, or
pages/ais_gap.py / pages/report.py's AFE report for GFW-side fishing
data).
"""

import base64
import io

import dash
import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl, card
from gfw import load_ais_buffer_polygon, classify_gap_status
from port_zones import port_mask_from_xy, PORT_RADIUS_M_DEFAULT, has_ports
from pages.loitering import get_loitering_dataframe
from pages.encounter import get_encounters_dataframe

GAP_THRESHOLD_HOURS = 3    # any gap longer than this counts as an AIS blackout
DEFAULT_BUFFER_NM = 3      # AIS coverage buffer distance -- must match pages/ais_gap.py's
VISIT_GAP_THRESHOLD_HOURS = 3  # gap between two "in port" points before it's a new visit

SECTION_LABELS = {
    "port_visit": "Port visits",
    "gap":        "AIS gaps",
    "loitering":  "Loitering events",
    "encounter":  "Encounters",
}

PLACEHOLDER_STYLE = {"color": DIM, "padding": "2rem", "fontSize": "0.85rem", "fontStyle": "italic"}

# Cache serveur du CSV importé, comme _CSV_CACHE dans pages/heatmap.py --
# évite l'aller-retour du dataframe par le navigateur.
_CSV_CACHE = {"df": None, "filename": None}


def _parse_uploaded_csv(contents, filename):
    """Identique à heatmap._parse_uploaded_csv (dcc.Upload -> DataFrame)."""
    if contents is None:
        return None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
    return pd.read_csv(io.BytesIO(decoded))


def _find_vessels_in_csv(df):
    """Liste les navires distincts presents dans le CSV importe, meme
    forme d'entree que l'ancienne _group_results_by_mmsi (label, ids,
    name, flag...) mais construite depuis les colonnes du CSV plutot
    qu'une recherche GFW."""
    entries = []
    if df is None or df.empty or "vessel_id" not in df.columns:
        return entries

    for vid, grp in df.groupby("vessel_id"):
        def first_valid(col):
            if col not in grp.columns:
                return None
            s = grp[col].dropna()
            return s.iloc[0] if not s.empty else None

        name = first_valid("ship_name") or "Unknown"
        mmsi = first_valid("mmsi") or "?"
        flag = first_valid("flag") or "?"
        label = f"{name} | MMSI {mmsi} | {flag}"
        entries.append({
            "label": label, "ids": [vid], "name": name, "mmsi": mmsi, "flag": flag,
            "imo": first_valid("imo"), "vessel_type": first_valid("vessel_type"),
            "gear_type": first_valid("gear_type"),
        })
    entries.sort(key=lambda e: str(e["name"]))
    return entries


def _project_xy(df):
    """Ajoute des colonnes x/y en metres (EPSG:32634), reutilise par la
    detection de port visits et le calcul de gaps."""
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat),
                           crs="EPSG:4326").to_crs("EPSG:32634")
    df = df.copy()
    df["x"] = gdf.geometry.x.values
    df["y"] = gdf.geometry.y.values
    return df


def _detect_gaps_local(df_vessel, gap_threshold_hours=GAP_THRESHOLD_HOURS, buffer_nm=DEFAULT_BUFFER_NM):
    """Detecte les trous AIS d'un navire par simple diff des positions
    consecutives (le CSV importe a sa propre resolution temporelle --
    pas d'hypothese sur un pas fixe). Chaque trou est classe suspicious/
    normal via le meme buffer de couverture AIS que pages/ais_gap.py."""
    empty = pd.DataFrame(columns=["start", "end", "duration_hrs", "off_lat", "off_lon",
                                  "on_lat", "on_lon", "status"])
    d = df_vessel.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "lat", "lon"]).sort_values("date")
    if len(d) < 2:
        return empty

    d["prev_date"] = d["date"].shift()
    d["prev_lat"] = d["lat"].shift()
    d["prev_lon"] = d["lon"].shift()
    d["gap_hours"] = (d["date"] - d["prev_date"]).dt.total_seconds() / 3600

    gaps = d[d["gap_hours"] >= gap_threshold_hours].copy()
    if gaps.empty:
        return empty

    buffer_geom = load_ais_buffer_polygon(buffer_nm)
    if buffer_geom is not None:
        gaps["status"] = gaps.apply(
            lambda r: classify_gap_status(r["prev_lat"], r["prev_lon"], r["lat"], r["lon"], buffer_geom),
            axis=1)
    else:
        gaps["status"] = "gap"  # unclassified -- no buffer file found

    out = pd.DataFrame({
        "start": gaps["prev_date"].dt.strftime("%Y-%m-%d %H:%M"),
        "end": gaps["date"].dt.strftime("%Y-%m-%d %H:%M"),
        "duration_hrs": gaps["gap_hours"].round(2),
        "off_lat": gaps["prev_lat"].round(5),
        "off_lon": gaps["prev_lon"].round(5),
        "on_lat": gaps["lat"].round(5),
        "on_lon": gaps["lon"].round(5),
        "status": gaps["status"],
    })
    return out.sort_values("start").reset_index(drop=True)


def _detect_port_visits_local(df_vessel, port_radius_m=PORT_RADIUS_M_DEFAULT):
    """Detecte les visites de port d'un navire : positions a moins de
    port_radius_m d'un point de port connu (port_zones.py), regroupees
    en visites distinctes des qu'un ecart de plus de
    VISIT_GAP_THRESHOLD_HOURS separe deux positions "in port"."""
    empty = pd.DataFrame(columns=["start", "end", "duration_hrs", "n_points"])
    if not has_ports():
        return empty

    d = df_vessel.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "lat", "lon"]).sort_values("date")
    if d.empty:
        return empty

    d = _project_xy(d)
    d["in_port"] = port_mask_from_xy(d["x"].values, d["y"].values, radius_m=port_radius_m)
    d = d[d["in_port"]]
    if d.empty:
        return empty

    d["gap"] = d["date"].diff()
    d["visit_id"] = (d["gap"] > pd.Timedelta(hours=VISIT_GAP_THRESHOLD_HOURS)).cumsum()

    rows = []
    for _, visit in d.groupby("visit_id"):
        start, end = visit["date"].min(), visit["date"].max()
        rows.append({
            "start": start.strftime("%Y-%m-%d %H:%M"),
            "end": end.strftime("%Y-%m-%d %H:%M"),
            "duration_hrs": round((end - start).total_seconds() / 3600, 2),
            "n_points": len(visit),
        })
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def _upload_zone():
    return dcc.Upload(
        id="vr-csv-upload",
        children=html.Div([
            "Drag a CSV here, or ",
            html.A("browse", style={"color": ACC, "textDecoration": "underline"}),
        ]),
        style={
            "width": "100%", "padding": "1rem 0.5rem",
            "textAlign": "center", "cursor": "pointer",
            "border": f"1px dashed {BDR}", "borderRadius": "6px",
            "color": SOFT, "fontSize": "0.75rem",
            "marginBottom": "0.5rem",
        },
        multiple=False,
    )


# ── LAYOUT ───────────────────────────────────────────────────────────────────

def layout():
    return html.Div([
        dcc.Store(id="vr-search-store", data=None),
        dcc.Store(id="vr-csv-loaded", data=None),
        dcc.Store(id="vr-report-store", data=None),
        dcc.Download(id="vr-download-csv"),

        html.Div([
            html.H6("Vessel report", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P("Import a CSV, select a vessel found in it, and get a full "
                   "activity report: port visits, AIS gaps, loitering and "
                   "encounters -- all computed from the imported positions.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            html.Div([
                html.H6("Import a CSV", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
                _upload_zone(),
                html.Div("No file selected", id="vr-csv-filename",
                          style={"fontSize": "0.72rem", "color": DIM,
                                 "fontStyle": "italic", "marginBottom": "0.6rem"}),
            ], style={"marginBottom": "1rem", "paddingBottom": "1rem",
                       "borderBottom": "1px solid " + BDR}),

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

            html.P("The whole imported file is analyzed for the selected vessel -- "
                   "there's no date-range filter here.",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "0.6rem"}),

            html.Button("Generate Report", id="vr-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": "linear-gradient(135deg,#d15400,#a03e00)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            html.Div(id="vr-status", style={"fontSize": "0.72rem", "color": SOFT}),

        ], style={"width": "320px", "minWidth": "320px", "padding": "1rem",
                   "background": BG, "borderRight": "1px solid " + BDR,
                   "flexShrink": "0", "position": "sticky", "top": "0",
                   "alignSelf": "flex-start", "maxHeight": "100vh", "overflowY": "auto"}),

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
                        children=html.P("Import a CSV, select a vessel, then click "
                                        "\"Generate Report\".", style=PLACEHOLDER_STYLE))),
                style={"padding": "1rem", "minWidth": "0", "overflowX": "auto"},
            ),
        ], style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "alignItems": "flex-start"})


# ── Rendu du rapport ─────────────────────────────────────────────────────────

def _section_table(key, df):
    label = SECTION_LABELS[key]
    if df is None or df.empty:
        return html.Div([
            html.H6(label, style={"color": MAIN, "fontSize": "0.85rem", "marginBottom": "0.3rem"}),
            html.P("None found for this vessel.", style={"color": SOFT, "fontSize": "0.75rem"}),
        ], style={"marginBottom": "1.2rem"})

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
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


def _summary_bar(info, results):
    stats = [(SECTION_LABELS[k], len(results.get(k, pd.DataFrame()))) for k in SECTION_LABELS]
    boxes = [
        html.Div([
            html.Div(str(v), style={"fontSize": "1.3rem", "fontWeight": "700", "color": ACC}),
            html.Div(k, style={"fontSize": "0.68rem", "color": SOFT}),
        ], style={"textAlign": "center", "flex": "1", "minWidth": "100px"})
        for k, v in stats
    ]
    return html.Div(card([
        html.H5(info["name"], style={"color": MAIN, "marginBottom": "0.1rem"}),
        html.P(f"MMSI {info['mmsi']} | IMO {info.get('imo') or '?'} | Flag {info.get('flag') or '?'}",
               style={"color": DIM, "fontSize": "0.75rem", "marginBottom": "0.8rem"}),
        html.Div(boxes, style={"display": "flex", "gap": "0.5rem", "flexWrap": "wrap"}),
        html.P("Fishing events aren't included: detecting genuine fishing activity needs "
               "GFW's own classification model, not just raw AIS positions.",
               style={"color": DIM, "fontSize": "0.68rem", "fontStyle": "italic", "marginTop": "0.6rem"}),
    ]))


def _render_report(info, results):
    return html.Div([
        _summary_bar(info, results),
        html.Div([_section_table(key, results.get(key)) for key in SECTION_LABELS],
                 style={"minWidth": "0"}),
    ], style={"minWidth": "0"})


# ── CALLBACKS ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # Parse le CSV dès qu'il est déposé -- peuple la liste des navires
    # trouvés. Le rapport lui-même n'est calculé qu'au clic sur
    # "Generate Report", plus bas.
    @app.callback(
        Output("vr-csv-filename", "children"),
        Output("vr-vessel-selector", "options"),
        Output("vr-vessel-selector", "value"),
        Output("vr-search-store", "data"),
        Output("vr-csv-loaded", "data"),
        Input("vr-csv-upload", "contents"),
        State("vr-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv_uploaded(contents, filename):
        if not contents:
            raise dash.exceptions.PreventUpdate
        try:
            df = _parse_uploaded_csv(contents, filename)
        except Exception as e:
            _CSV_CACHE["df"] = None
            return f"Error: {e}", [], None, None, None

        required = {"lat", "lon", "vessel_id", "date"}
        if not required.issubset(df.columns):
            _CSV_CACHE["df"] = None
            missing = ", ".join(sorted(required - set(df.columns)))
            return f'"{filename}" is missing required column(s): {missing}.', [], None, None, None

        _CSV_CACHE["df"] = df
        _CSV_CACHE["filename"] = filename

        entries = _find_vessels_in_csv(df)
        if not entries:
            return f'"{filename}" loaded, but no vessel_id found in it.', [], None, None, None
        opts = [{"label": e["label"], "value": str(i)} for i, e in enumerate(entries)]
        return (f"Loaded: {len(df):,} rows from \"{filename}\" ({len(entries)} vessel(s)).",
                opts, None, entries, "loaded")

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
        prevent_initial_call=True,
    )
    def _run(n, idx, entries):
        if not n:
            raise dash.exceptions.PreventUpdate
        full_df = _CSV_CACHE.get("df")
        if full_df is None or full_df.empty:
            return dash.no_update, "Import a CSV first.", None
        if idx is None or not entries:
            return dash.no_update, "Select a vessel first.", None

        info = entries[int(idx)]
        vessel_id = info["ids"][0]
        df_vessel = full_df[full_df["vessel_id"] == vessel_id]
        if df_vessel.empty:
            return dash.no_update, "No rows for this vessel in the imported CSV.", None

        results = {}
        results["port_visit"] = _detect_port_visits_local(df_vessel)
        results["gap"] = _detect_gaps_local(df_vessel)

        loi = get_loitering_dataframe(df_vessel)
        results["loitering"] = loi.drop(columns=["vessel_id"], errors="ignore") if not loi.empty else loi

        enc_cols = ["lat", "lon", "vessel_id", "ship_name", "date"]
        if set(enc_cols).issubset(full_df.columns):
            enc = get_encounters_dataframe(full_df[enc_cols])
            if not enc.empty:
                enc = enc[(enc["vessel_1_id"] == vessel_id) | (enc["vessel_2_id"] == vessel_id)]
            results["encounter"] = enc
        else:
            results["encounter"] = pd.DataFrame()

        # Store combine pour l'export : chaque df avec une colonne event_type
        combined_frames = []
        for key, df in results.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d.insert(0, "event_type", SECTION_LABELS[key])
            combined_frames.append(d)
        store = (pd.concat(combined_frames, ignore_index=True, sort=False)
                 .to_dict("records")) if combined_frames else None

        status = "Report generated."
        return _render_report(info, results), status, store

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