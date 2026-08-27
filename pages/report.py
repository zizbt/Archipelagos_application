"""
pages/report.py
====================
Page "Reports" -- two independent report generators, side by side (same
visual pattern as the Data (download) page):

- LEFT panel  : VP Report    -- per-vessel AIS-gap / suspicious-gap /
                tracked-activity / encounter-events report, from a
                downloaded "Vessel Presence" CSV.
- RIGHT panel : AFE Report   -- fishing effort by zone/country, from a
                downloaded "Fishing Effort" CSV (needs an 'hours' column).

Both panels list every CSV already downloaded via the Data page (no
automatic filtering by content -- the user picks whichever file is
relevant for the report they want, exactly like the two download panels
on the Data page).

NOTE on the VP report:
The original standalone script (VP_report.py) imported
`get_encounter_events` from a module called `VP_bulk_map`, which was
never migrated into this app and isn't available. Instead, this page
reuses the encounter-detection logic already built and used on the
"Encounters" page (`pages/encounter.get_encounters_dataframe`, same
500m / 2h thresholds), and counts encounter events per vessel_id
directly from its output -- more robust than the original's approach of
counting name occurrences in generated popup text.

NOTE on the AIS buffer file:
Assumes 'map_files/ais_buffer_{buffer_dis}nm.geojson' exists at
ROOT / "map_files" / ..., same relative layout as the original standalone
scripts. Adjust BUFFER_DIS or the path below if that's not where it
lives in this project.
"""

import dash
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl, GFW_DOWNLOAD_DIR
from config import ROOT, ZONES, FLAG_NAMES
from gfw import load_csv


def _open_native_csv_dialog(initial_dir):
    """
    Opens the OS's native "Open file" dialog (Windows Explorer style,
    same as any desktop app) restricted to CSV files, and returns the
    chosen path as a string, or None if the user cancelled.

    Only works when the Dash app is run locally (server and browser on
    the same machine) since tkinter needs a display on the machine
    where this code executes.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Ouvrir",
            initialdir=str(initial_dir) if Path(initial_dir).exists() else str(ROOT),
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
    finally:
        root.destroy()
    return path or None

STANDARD_CRS = "EPSG:4326"

# Fixed AIS-buffer radius used for the VP report (matches the original
# standalone script's default). No UI control for this -- change here if
# you need a different buffer distance.
BUFFER_DIS = 3

# Zones covered by the AFE report, in display order.
REPORT_ZONE_KEYS = [
    "greece_territorial",
    "turkey_territorial",
    "italy_territorial",
    "malta_fmz",
    "malta_national",
]


# SHARED HELPERS
def _load_zone_polygon(zone_key):
    """Load the shapefile for the given zone key and return a unified polygon (or None)."""
    cfg = ZONES.get(zone_key)
    if not cfg or not cfg.get("shp"):
        return None
    path = ROOT / cfg["shp"]
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path).to_crs(STANDARD_CRS)
        return gdf.geometry.union_all()
    except Exception:
        return None


def _load_ais_buffer(buffer_dis=BUFFER_DIS):
    path = ROOT / "data" / "gis" / f"ais_buffer_{buffer_dis}nm.geojson"
    if not path.exists():
        return None
    try:
        return gpd.read_file(path).to_crs(STANDARD_CRS)
    except Exception:
        return None


# AFE REPORT (fishing effort by zone/country)
def build_report(df):
    """
    Returns two DataFrames: a long one (one row per vessel per zone) and a
    matrix one (one block per zone, one column per country, one row per
    vessel) -- the matrix is what gets exported to CSV.
    """
    if df is None or df.empty or "lon" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    gdf = gpd.GeoDataFrame(df.copy(),
                           geometry=gpd.points_from_xy(df.lon, df.lat),
                           crs=STANDARD_CRS)

    zone_masks = {}
    for key in REPORT_ZONE_KEYS:
        poly = _load_zone_polygon(key)
        if poly is not None:
            zone_masks[key] = gdf.geometry.within(poly)

    if not zone_masks:
        return pd.DataFrame(), pd.DataFrame()

    long_rows = []
    blocks = []

    for key, mask in zone_masks.items():
        label = ZONES[key].get("label", key)
        zone_df = gdf[mask]
        if zone_df.empty:
            continue

        hours_col = f"Fishing Hours ({label})"
        col_headers = ["Ship Name", "MMSI", "Country", hours_col]
        pieces = [
            pd.DataFrame([[f"--- {label.upper()} ---", "", "", ""]], columns=col_headers),
            pd.DataFrame([["", "", "", ""]], columns=col_headers),
        ]

        country_order = (zone_df.groupby("flag")["hours"].sum()
                         .sort_values(ascending=False).index.tolist())

        for country in country_order:
            c_df = zone_df[zone_df["flag"] == country]
            vsum = (c_df.groupby("vessel_id")
                    .agg({"ship_name": "first", "mmsi": "first",
                          "flag": "first", "hours": "sum"})
                    .reset_index()
                    .sort_values("hours", ascending=False))

            for _, r in vsum.iterrows():
                long_rows.append({
                    "Zone": label,
                    "Country": FLAG_NAMES.get(country, country),
                    "Ship Name": r["ship_name"],
                    "MMSI": str(r["mmsi"]),
                    "Fishing Hours": round(float(r["hours"]), 2),
                })

            disp = vsum[["ship_name", "mmsi", "flag", "hours"]].copy()
            disp.columns = col_headers
            disp["MMSI"] = disp["MMSI"].astype(str)
            disp[hours_col] = disp[hours_col].apply(lambda x: f"{x:.2f}")

            pieces.extend([
                pd.DataFrame([[f"=== COUNTRY: {str(country).upper()} ===", "", "", ""]], columns=col_headers),
                pd.DataFrame([col_headers], columns=col_headers),
                disp,
                pd.DataFrame([[f"Total {country} Vessels: {len(vsum)}", "", "", ""]], columns=col_headers),
                pd.DataFrame([[f"Total {country} Hours: {vsum['hours'].sum():.2f}", "", "", ""]], columns=col_headers),
                pd.DataFrame([["", "", "", ""]], columns=col_headers),
            ])

        pieces.extend([
            pd.DataFrame([["", "", "", ""]], columns=col_headers),
            pd.DataFrame([["=== GRAND TOTALS FOR REGION ===", "", "", ""]], columns=col_headers),
            pd.DataFrame([[f"Grand Total Vessels: {zone_df['vessel_id'].nunique()}", "", "", ""]], columns=col_headers),
            pd.DataFrame([[f"Grand Total Fishing Hours: {zone_df['hours'].sum():.2f}", "", "", ""]], columns=col_headers),
        ])
        blocks.append(pd.concat(pieces, ignore_index=True))

    long_df = pd.DataFrame(long_rows)

    def spacer():
        s = pd.DataFrame()
        s[" "] = ""
        s["  "] = ""
        return s

    if blocks:
        parts = []
        for i, b in enumerate(blocks):
            parts.append(b)
            if i < len(blocks) - 1:
                parts.extend([spacer(), spacer()])
        matrix_df = pd.concat(parts, axis=1).fillna("")
    else:
        matrix_df = pd.DataFrame()

    return long_df, matrix_df


# VP REPORT (per-vessel AIS gaps, tracked activity, encounter events)
def build_vp_report(df, filter_type="All Vessels", buffer_dis=BUFFER_DIS,
                    start_date="N/A", end_date="N/A"):
    """
    Returns (report_df, totals_dict, meta_text).
    report_df has one row per vessel:
      Vessel Id, Vessel Name, MMSI, Country,
      Gap Hours (outside AIS buffer), Total Gap Hours,
      Tracked Activity (Hrs), Encounter Events
    """
    empty_cols = ["Vessel Id", "Vessel Name", "MMSI", "Country",
                  "Gap Hours (outside AIS buffer)", "Total Gap Hours",
                  "Tracked Activity (Hrs)", "Encounter Events",
                  "Risk Score", "Risk Level"]
    if df is None or df.empty or not {"lat", "lon", "vessel_id", "date"}.issubset(df.columns):
        return pd.DataFrame(columns=empty_cols), {}, "No data."

    ais_buffer_gdf = _load_ais_buffer(buffer_dis)
    if ais_buffer_gdf is None:
        return pd.DataFrame(columns=empty_cols), {}, \
            f"Missing AIS buffer file: map_files/ais_buffer_{buffer_dis}nm.geojson"

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["vessel_id", "date"])

    # Time gaps in hours between consecutive positions of the same vessel
    df["gap_hours"] = df.groupby("vessel_id")["date"].diff().dt.total_seconds() / 3600
    df["gap_hours"] = df["gap_hours"].fillna(0)

    # Spatial status: is each position inside the AIS buffer zone?
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=STANDARD_CRS)
    res = gpd.sjoin(gdf, ais_buffer_gdf, how="left", predicate="within")
    df["is_inside"] = ~res["index_right"].isna()
    df["prev_is_inside"] = df.groupby("vessel_id")["is_inside"].shift(1)

    # A "suspicious" gap: a gap of more than 3h that happened while the
    # vessel was outside the buffer both before and after the gap (i.e.
    # not explained by simply crossing in/out of the tracked zone).
    df["is_gap"] = df["gap_hours"] > 3
    df["is_suspicious"] = (
        df["is_gap"] & (df["is_inside"] == False) & (df["prev_is_inside"] == False)
    )

    # Encounter events -- reuse the app's existing encounter-detection
    # logic (Encounters page), then count occurrences per vessel_id.
    encounter_counts = {}
    try:
        from pages.encounter import get_encounters_dataframe
        enc_cols = ["lat", "lon", "vessel_id", "ship_name", "date"]
        if set(enc_cols).issubset(df.columns):
            enc = get_encounters_dataframe(df[enc_cols])
            if not enc.empty:
                # Keep only encounters whose midpoint falls OUTSIDE the AIS buffer
                enc_gdf = gpd.GeoDataFrame(enc, geometry=gpd.points_from_xy(enc.lon, enc.lat),
                                           crs=STANDARD_CRS)
                enc_outside = gpd.sjoin(enc_gdf, ais_buffer_gdf, how="left", predicate="within")
                enc = enc_outside[enc_outside["index_right"].isna()]
                for col in ["vessel_1_id", "vessel_2_id"]:
                    for vid in enc[col]:
                        encounter_counts[vid] = encounter_counts.get(vid, 0) + 1
    except Exception:
        # Encounter detection is a bonus metric here -- if it fails for any
        # reason (missing columns, etc.), the rest of the VP report still
        # gets produced with Encounter Events = 0 everywhere.
        pass

    selection_label = "Trawlers Only" if filter_type == "Trawlers Only" else "All Vessels"
    if filter_type == "Trawlers Only" and "gear_type" in df.columns:
        working = df[df["gear_type"].astype(str).str.upper() == "TRAWLERS"].copy()
    else:
        working = df.copy()

    if working.empty:
        return pd.DataFrame(columns=empty_cols), {}, "No vessels match this filter."

    working["ship_name"] = working["ship_name"].fillna("UNKNOWN NAME") if "ship_name" in working.columns else "UNKNOWN NAME"
    working["mmsi"] = working["mmsi"].fillna("UNKNOWN MMSI") if "mmsi" in working.columns else "UNKNOWN MMSI"
    working["flag"] = working["flag"].fillna("Unknown") if "flag" in working.columns else "Unknown"

    working["only_gap_hrs"] = working["gap_hours"].where(working["is_gap"], 0)
    working["only_suspicious_hrs"] = working["gap_hours"].where(working["is_suspicious"], 0)

    stats = working.groupby("vessel_id").agg(
        vessel_name=("ship_name", "first"),
        mmsi=("mmsi", "first"),
        flag=("flag", "first"),
        total_span_hrs=("date", lambda x: (x.max() - x.min()).total_seconds() / 3600),
        total_gap_hrs=("only_gap_hrs", "sum"),
        suspicious_gap_hrs=("only_suspicious_hrs", "sum"),
    )
    stats["total_activity_hrs"] = (stats["total_span_hrs"] - stats["total_gap_hrs"]).clip(lower=0)
    stats["encounter_events"] = stats.index.map(lambda vid: encounter_counts.get(vid, 0))

    report = stats.reset_index()[[
        "vessel_id", "vessel_name", "mmsi", "flag",
        "suspicious_gap_hrs", "total_gap_hrs", "total_activity_hrs", "encounter_events",
    ]].sort_values("suspicious_gap_hrs", ascending=False)

    report.columns = empty_cols
    for c in ["Gap Hours (outside AIS buffer)", "Total Gap Hours", "Tracked Activity (Hrs)"]:
        report[c] = report[c].round(2)

    risk_raw = (
        report["Gap Hours (outside AIS buffer)"] * 1.5
        + report["Total Gap Hours"] * 0.5
        + report["Encounter Events"] * 8
    )
    report["Risk Score"] = risk_raw.clip(lower=0, upper=100).round(0).astype(int)
    report["Risk Level"] = pd.cut(
        report["Risk Score"],
        bins=[-1, 34, 69, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    report = report.sort_values([
        "Risk Score",
        "Encounter Events",
        "Gap Hours (outside AIS buffer)",
    ], ascending=False).reset_index(drop=True)

    totals = {
        "Gap Hours (outside AIS buffer)": round(float(report["Gap Hours (outside AIS buffer)"].sum()), 2),
        "Total Gap Hours": round(float(report["Total Gap Hours"].sum()), 2),
        "Tracked Activity (Hrs)": round(float(report["Tracked Activity (Hrs)"].sum()), 2),
        "Encounter Events": int(report["Encounter Events"].sum()),
    }

    report_flag = "N/A"
    valid_flags = stats["flag"].replace("Unknown", pd.NA).dropna()
    if not valid_flags.empty:
        report_flag = valid_flags.iloc[0]

    meta = (f"Selection: {selection_label} | Period: {start_date} to {end_date} | "
            f"Global Flag: {report_flag} | Total Vessels: {len(report)}")

    return report, totals, meta


# LAYOUT
def _panel_style():
    return {"background": PANEL, "border": f"1px solid {BDR}", "borderRadius": "10px",
            "padding": "1.2rem", "flex": "1", "minWidth": "320px"}


def _browse_button_style():
    return {"width": "100%", "padding": "0.55rem", "background": PANEL,
            "color": MAIN, "border": f"1px solid {BDR}", "borderRadius": "6px",
            "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.4rem",
            "textAlign": "left"}


def _selected_file_style():
    return {"fontSize": "0.75rem", "color": SOFT, "marginBottom": "1rem",
            "wordBreak": "break-all"}


def _classic_layout():
    zones_txt = ", ".join(ZONES[k].get("label", k) for k in REPORT_ZONE_KEYS) or "no zone configured"

    return html.Div([
        dcc.Download(id="vp-report-download"),
        dcc.Download(id="afe-report-download"),
        dcc.Store(id="vp-report-csv"),
        dcc.Store(id="afe-report-csv"),

        html.H5("Reports", style={"color": MAIN, "marginBottom": "0.3rem"}),
        html.P("Build a report from a CSV already downloaded on the Data page. "
               "The report is downloaded directly by your browser.",
               style={"color": DIM, "fontSize": "0.8rem", "marginBottom": "1.2rem"}),

        html.Div([

            # LEFT PANEL: VP Report
            html.Div([
                html.H6("VP Report", style={"color": MAIN, "marginBottom": "0.3rem"}),
                html.P("AIS gaps, tracked activity and encounter events, per vessel.",
                       style={"fontSize": "0.75rem", "color": DIM, "marginBottom": "1rem"}),

                lbl("Downloaded CSV (Vessel Presence)"),
                html.Button([html.Span("📁 "), "Parcourir..."], id="vp-browse-btn",
                    n_clicks=0, style=_browse_button_style()),
                html.Div("Aucun fichier selectionne", id="vp-selected-file",
                    style=_selected_file_style()),

                lbl("Vessel selection"),
                dcc.Dropdown(id="vp-report-filter", value="All Vessels", clearable=False,
                    options=[{"label": "All Vessels", "value": "All Vessels"},
                             {"label": "Trawlers Only", "value": "Trawlers Only"}],
                    style={"color": "#000", "marginBottom": "1rem"}),

                html.Button("Generate VP Report", id="vp-report-run", n_clicks=0,
                    style={"width": "100%", "padding": "0.6rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none", "borderRadius": "6px",
                           "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.6rem"}),

                html.Div(id="vp-report-status",
                         style={"fontSize": "0.75rem", "color": SOFT, "marginTop": "0.8rem"}),
            ], style=_panel_style()),

            # RIGHT PANEL: AFE Report
            html.Div([
                html.H6("AFE Report", style={"color": MAIN, "marginBottom": "0.3rem"}),
                html.P(f"Fishing effort by zone and country. Zones: {zones_txt}.",
                       style={"fontSize": "0.75rem", "color": DIM, "marginBottom": "1rem"}),

                lbl("Downloaded CSV (Fishing Effort)"),
                html.Button([html.Span("📁 "), "Parcourir..."], id="afe-browse-btn",
                    n_clicks=0, style=_browse_button_style()),
                html.Div("Aucun fichier selectionne", id="afe-selected-file",
                    style=_selected_file_style()),

                html.Button("Generate AFE Report", id="afe-report-run", n_clicks=0,
                    style={"width": "100%", "padding": "0.6rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none", "borderRadius": "6px",
                           "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.6rem"}),

                html.Div(id="afe-report-status",
                         style={"fontSize": "0.75rem", "color": SOFT, "marginTop": "0.8rem"}),
            ], style=_panel_style()),

        ], style={"display": "flex", "gap": "1.2rem", "flexWrap": "wrap", "marginBottom": "1.5rem"}),

        html.Div(
            dcc.Loading(children=html.Div(id="report-table")),
            style={"padding": "0 0.2rem"},
        ),

    ], style={"padding": "1.5rem", "background": BG, "minHeight": "calc(100vh - 52px)"})


def layout():
    """Point d'entree de la page 'Report' : un selecteur en haut permet de
    basculer entre le rapport classique (VP/AFE depuis un CSV telecharge)
    et le rapport par navire (recherche + evenements GFW), sans occuper
    deux onglets separes dans la barre de navigation."""
    from pages import vessel_report as page_vessel_report

    return html.Div([
        html.Div([
            lbl("Report type"),
            dcc.RadioItems(
                id="report-mode",
                options=[
                    {"label": " Classic Report (VP / AFE from CSV)", "value": "classic"},
                    {"label": " Vessel Report (search a vessel)", "value": "vessel"},
                ],
                value="classic",
                labelStyle={"display": "inline-block", "marginRight": "1.5rem",
                            "fontSize": "0.8rem", "color": SOFT, "cursor": "pointer"},
            ),
        ], style={"padding": "1rem 1.5rem 0 1.5rem", "background": BG, "flexShrink": "0"}),

        # Pas de hauteur figee ici (ni "calc(100vh - Npx)", ni "100%") : la vraie
        # hauteur de la barre de nav n'est pas fiable a deviner depuis cette page
        # (elle peut faire 52px ou 140px selon le theme/l'ecran), et une valeur
        # fausse fait deborder tout le contenu hors de l'ecran. On laisse la page
        # s'etendre naturellement et le navigateur gerer le defilement vertical.
        html.Div(id="report-mode-content", children=_classic_layout(),
                  style={"minWidth": "0"}),
    ], style={"display": "flex", "flexDirection": "column", "minWidth": "0"})
    if long_df is None or long_df.empty:
        return html.P("No fishing effort found in the configured zones for this CSV.",
                      style={"color": SOFT, "fontSize": "0.85rem"})
    return dash_table.DataTable(
        data=long_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in long_df.columns],
        sort_action="native", filter_action="native", page_size=30,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": f"1px solid {BDR}",
                    "fontSize": "0.75rem", "padding": "4px 8px"},
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
    )


def _vp_table(report_df, totals, meta):
    if report_df is None or report_df.empty:
        return html.P("No vessels found for this CSV / selection.",
                      style={"color": SOFT, "fontSize": "0.85rem"})
    children = [html.P(meta, style={"color": DIM, "fontSize": "0.75rem", "marginBottom": "0.6rem"})]
    children.append(dash_table.DataTable(
        data=report_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in report_df.columns],
        sort_action="native", filter_action="native", page_size=30,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": f"1px solid {BDR}",
                    "fontSize": "0.75rem", "padding": "4px 8px"},
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
    ))
    if totals:
        children.append(html.P(
            f"TOTALS -- Gap Hours (outside AIS buffer): {totals['Gap Hours (outside AIS buffer)']:.2f} | "
            f"Total Gap Hours: {totals['Total Gap Hours']:.2f} | "
            f"Tracked Activity (Hrs): {totals['Tracked Activity (Hrs)']:.2f} | "
            f"Encounter Events: {totals['Encounter Events']}",
            style={"color": SOFT, "fontSize": "0.78rem", "marginTop": "0.6rem", "fontWeight": "600"}
        ))
    return html.Div(children)


# ---------------------------------------------------------------------------
# DIRECT-TO-DISK WRITERS (same pattern as the Data page: no browser save
# dialog, the report CSV is written straight into GFW_DOWNLOAD_DIR)
# ---------------------------------------------------------------------------
def _save_afe_report(matrix_df, source_csv_path):
    stem = Path(source_csv_path).stem
    fname = f"AFE_report_{stem}.csv"
    out_path = GFW_DOWNLOAD_DIR / fname
    matrix_df.to_csv(out_path, index=False, header=False)
    return out_path


def _save_vp_report(report_df, totals, meta, source_csv_path):
    stem = Path(source_csv_path).stem
    fname = f"VP_report_{stem}.csv"
    out_path = GFW_DOWNLOAD_DIR / fname
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(f'"{meta}"\n\n')
        report_df.to_csv(f, index=False)
        f.write("\n\n\n")
        f.write(
            f'"TOTALS",,,{totals["Gap Hours (outside AIS buffer)"]:.2f},'
            f'{totals["Total Gap Hours"]:.2f},{totals["Tracked Activity (Hrs)"]:.2f},'
            f'{totals["Encounter Events"]}\n'
        )
    return out_path


# CALLBACKS
def register_callbacks(app):

    from pages import vessel_report as page_vessel_report
    page_vessel_report.register_callbacks(app)

    @app.callback(
        Output("report-mode-content", "children"),
        Input("report-mode", "value"),
        prevent_initial_call=True,
    )
    def _switch_report_mode(mode):
        if mode == "vessel":
            return page_vessel_report.layout()
        return _classic_layout()

    @app.callback(
        Output("vp-report-csv", "data"),
        Output("vp-selected-file", "children"),
        Input("vp-browse-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _browse_vp_csv(n_clicks):
        path = _open_native_csv_dialog(ROOT / "data")
        if not path:
            raise dash.exceptions.PreventUpdate
        return path, path

    @app.callback(
        Output("afe-report-csv", "data"),
        Output("afe-selected-file", "children"),
        Input("afe-browse-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _browse_afe_csv(n_clicks):
        path = _open_native_csv_dialog(ROOT / "data")
        if not path:
            raise dash.exceptions.PreventUpdate
        return path, path

    @app.callback(
        Output("report-table", "children"),
        Output("afe-report-status", "children"),
        Output("vp-report-status", "children"),
        Output("afe-report-download", "data"),
        Output("vp-report-download", "data"),
        Input("afe-report-run", "n_clicks"),
        Input("vp-report-run", "n_clicks"),
        State("afe-report-csv", "data"),
        State("vp-report-csv", "data"),
        State("vp-report-filter", "value"),
        prevent_initial_call=True,
    )
    def _run(n_afe, n_vp, afe_csv, vp_csv, vp_filter):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "afe-report-run":
            if not afe_csv:
                return dash.no_update, "Please choose a CSV.", dash.no_update, dash.no_update, dash.no_update
            try:
                df = load_csv(afe_csv)
            except Exception as e:
                return dash.no_update, f"CSV error: {str(e)[:80]}", dash.no_update, dash.no_update, dash.no_update

            long_df, matrix_df = build_report(df)
            if long_df.empty:
                return _afe_table(None), "No vessels in the configured zones.", dash.no_update, dash.no_update, dash.no_update

            out_path = _save_afe_report(matrix_df, afe_csv)
            status = (f"{len(long_df)} vessel-rows across {long_df['Zone'].nunique()} zone(s). "
                      f"Downloaded (also saved to {out_path.name}).")
            download = dcc.send_file(str(out_path))
            return _afe_table(long_df), status, dash.no_update, download, dash.no_update

        if trigger == "vp-report-run":
            if not vp_csv:
                return dash.no_update, dash.no_update, "Please choose a CSV.", dash.no_update, dash.no_update
            try:
                df = load_csv(vp_csv)
            except Exception as e:
                return dash.no_update, dash.no_update, f"CSV error: {str(e)[:80]}", dash.no_update, dash.no_update

            date_col = df["date"] if "date" in df.columns else None
            start_txt = str(date_col.min())[:10] if date_col is not None else "N/A"
            end_txt = str(date_col.max())[:10] if date_col is not None else "N/A"

            report_df, totals, meta = build_vp_report(
                df, filter_type=vp_filter, start_date=start_txt, end_date=end_txt)

            if report_df.empty:
                return _vp_table(None, {}, meta), dash.no_update, meta, dash.no_update, dash.no_update

            out_path = _save_vp_report(report_df, totals, meta, vp_csv)
            status = f"{len(report_df)} vessel(s) in report. Downloaded (also saved to {out_path.name})."
            download = dcc.send_file(str(out_path))
            return _vp_table(report_df, totals, meta), dash.no_update, status, dash.no_update, download

        raise dash.exceptions.PreventUpdate