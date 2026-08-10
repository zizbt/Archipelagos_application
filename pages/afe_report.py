"""
pages/afe_report.py
===================
Page "AFE Report" -- rapport of fishing effort by zone and country, based on a downloaded CSV.

"""

import io

import dash
import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl
from config import ROOT, GIS_DIR, ZONES, FLAG_NAMES
from gfw import list_downloaded_csvs, load_csv

STANDARD_CRS = "EPSG:4326"
REPORT_ZONE_KEYS = [k for k in ZONES if "territorial" in k]


# CHARGE OF THE SHAPEFILES
def _load_zone_polygon(zone_key):
    """Charge the shapefile for the given zone key and returns a unified polygon (or None)."""
    cfg = ZONES.get(zone_key)
    if not cfg or not cfg.get("shp"):
        return None
    path = GIS_DIR.parent / cfg["shp"]   # data/ + chemin relatif du shp
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path).to_crs(STANDARD_CRS)
        return gdf.geometry.union_all()
    except Exception:
        return None


# CONSTRUCTION OF THE REPORT
def build_afe_report(df):
    """
    Get back two DataFrames: a long one (one row per vessel per zone) and a matrix one
    (one block per zone, one column per country, one row per vessel).
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


# LAYOUT
def layout():
    zones_txt = ", ".join(ZONES[k].get("label", k) for k in REPORT_ZONE_KEYS) or "aucune zone configurée"
    return html.Div([
        dcc.Store(id="afe-report-matrix", data=None),
        dcc.Download(id="afe-report-download"),

        html.Div([
            html.H6("AFE report", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P(f"Fishing effort by zone and country. Zones: {zones_txt}.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Downloaded CSV"),
            dcc.Dropdown(id="afe-report-csv",
                options=[{"label": f["filename"], "value": f["path"]}
                         for f in list_downloaded_csvs(ROOT / "data")],
                value=None, placeholder="Choose a CSV...",
                style={"color": "#000", "marginBottom": "1rem"}),

            html.Button("Generate report", id="afe-report-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none", "borderRadius": "6px",
                       "cursor": "pointer", "fontWeight": "600", "marginBottom": "0.6rem"}),
            html.Button("Export CSV", id="afe-report-export", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem", "background": PANEL,
                       "color": SOFT, "border": f"1px solid {BDR}",
                       "borderRadius": "6px", "cursor": "pointer", "marginBottom": "1rem"}),

            html.Div(id="afe-report-status", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "300px", "minWidth": "300px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        html.Div(
            dcc.Loading(children=html.Div(id="afe-report-table")),
            style={"flex": "1", "minHeight": 0, "overflowY": "auto", "padding": "1rem"},
        ),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _table(long_df):
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


# CALLBACKS
def register_callbacks(app):

    @app.callback(
        Output("afe-report-table", "children"),
        Output("afe-report-status", "children"),
        Output("afe-report-matrix", "data"),
        Input("afe-report-run", "n_clicks"),
        State("afe-report-csv", "value"),
        prevent_initial_call=True,
    )
    def _run(n, csv_path):
        if not n:
            raise dash.exceptions.PreventUpdate
        if not csv_path:
            return _table(None), "Please choose a CSV.", None
        try:
            df = load_csv(csv_path)
        except Exception as e:
            return _table(None), f"CSV error: {str(e)[:80]}", None

        long_df, matrix_df = build_afe_report(df)
        if long_df.empty:
            return _table(None), "No vessels in the configured zones.", None

        status = f"{len(long_df)} vessel-rows across {long_df['Zone'].nunique()} zone(s)."

        matrix_store = matrix_df.to_dict("split") if not matrix_df.empty else None
        return _table(long_df), status, matrix_store

    @app.callback(
        Output("afe-report-download", "data"),
        Input("afe-report-export", "n_clicks"),
        State("afe-report-matrix", "data"),
        prevent_initial_call=True,
    )
    def _export(n, matrix_store):
        if not n or not matrix_store:
            raise dash.exceptions.PreventUpdate
        matrix_df = pd.DataFrame(**matrix_store)
        csv_str = matrix_df.to_csv(index=False, header=False)
        return dict(content=csv_str, filename="AFE_report.csv")