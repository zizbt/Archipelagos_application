"""
pages/data.py
=============
"Data" page: download from Global Fishing Watch.
No gear_type filter, no selector for existing CSVs here (moved to the
other pages, where each user picks their own file).
"""

import asyncio
import threading
from datetime import date
from pathlib import Path

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl, card, GFW_DOWNLOAD_DIR
from config import FLAG_NAMES
from gfw import get_gfw_client, bulk_load_data_to_csv, test_api_key, GFW_VESSEL_TYPES, COUNTRY_FLAGS


def layout():
    return html.Div([

        html.H5("Download from Global Fishing Watch",
                style={"color": MAIN, "marginBottom": "0.3rem"}),
        html.P("Enter your API key, select filters, then click Download.",
               style={"color": DIM, "fontSize": "0.8rem", "marginBottom": "1.2rem"}),

        card([
            lbl("GFW API Key"),
            # No stored/prefilled value here on purpose: this app is shared
            # between multiple users, so each person enters their own key.
            dcc.Input(id="data-gfw-key", type="password",
                      placeholder="Paste your GFW API key...",
                      style={"width": "100%", "padding": "0.5rem", "background": BG,
                             "color": MAIN, "border": f"1px solid {BDR}",
                             "borderRadius": "6px", "marginBottom": "0.3rem",
                             "fontFamily": "monospace"}),
            html.Div(id="data-key-status", style={"fontSize": "0.7rem", "minHeight": "1rem",
                                                    "marginBottom": "0.8rem"}),

            html.Div([
                html.Div([
                    lbl("Start date"),
                    dcc.DatePickerSingle(id="data-gfw-start", date=date(2024, 7, 1),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=date(2012, 1, 1),
                        max_date_allowed=date(2026, 12, 31)),
                ], style={"marginRight": "2rem"}),
                html.Div([
                    lbl("End date"),
                    dcc.DatePickerSingle(id="data-gfw-end", date=date(2024, 7, 31),
                        display_format="YYYY-MM-DD",
                        min_date_allowed=date(2012, 1, 1),
                        max_date_allowed=date(2026, 12, 31)),
                ]),
            ], style={"display": "flex", "marginBottom": "1rem"}),

            html.Div([
                html.Div([
                    lbl("Country / Flag"),
                    dcc.Dropdown(id="data-gfw-flags",
                        options=[{"label": "ALL countries", "value": "ALL"}] +
                                [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f}
                                 for f in COUNTRY_FLAGS],
                        value=["GRC"], multi=True, placeholder="Select countries...",
                        style={"width": "300px", "color": "#000"}),
                ], style={"marginRight": "1.5rem"}),
                html.Div([
                    lbl("Vessel type (leave empty for ALL)"),
                    dcc.Dropdown(id="data-gfw-vtypes",
                        options=[{"label": t.capitalize(), "value": t}
                                 for t in GFW_VESSEL_TYPES],
                        value=[], multi=True, placeholder="All types...",
                        style={"width": "300px", "color": "#000"}),
                ]),
            ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1.2rem"}),

            html.Div([
                html.Button("Download from GFW", id="data-btn-download", n_clicks=0,
                    style={"padding": "0.6rem 1.5rem", "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none", "borderRadius": "6px",
                           "cursor": "pointer", "fontWeight": "600", "marginRight": "1rem"}),
                # dcc.Loading shows a spinner around the status text for as
                # long as the do_download callback below is running.
                dcc.Loading(
                    id="data-download-loading",
                    type="circle",
                    color=ACC,
                    style={"display": "inline-block"},
                    children=html.Span(id="data-download-status",
                              style={"fontSize": "0.78rem", "color": SOFT, "fontStyle": "italic"}),
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ]),

        html.Div(id="data-active-dataset",
                 style={"marginTop": "1rem", "padding": "0.6rem 1rem",
                        "background": BG, "borderRadius": "6px",
                        "border": f"1px solid {BDR}",
                        "fontSize": "0.78rem", "color": DIM}),

    ], style={"padding": "1.5rem", "background": BG, "minHeight": "calc(100vh - 52px)"})


def register_callbacks(app):

    @app.callback(
        Output("data-key-status", "children"),
        Output("data-key-status", "style"),
        Input("data-gfw-key", "value"),
        prevent_initial_call=True,
    )
    def validate_key(key):
        if not key or len(key) < 10:
            return "", {}
        valid, msg = test_api_key(key)
        color = "#32cd32" if valid else "#ff6b6b"
        return msg, {"fontSize": "0.7rem", "color": color}

    # Clientside callback: runs instantly in the browser on click, before
    # the (potentially long) download even starts on the server side.
    # Gives the user immediate feedback that something is happening.
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) { return window.dash_clientside.no_update; }
            return "Download in progress...";
        }
        """,
        Output("data-download-status", "children", allow_duplicate=True),
        Input("data-btn-download", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("data-download-status", "children", allow_duplicate=True),
        Output("store-csv-path", "data"),
        Output("store-df", "data"),
        Output("data-active-dataset", "children"),
        Input("data-btn-download", "n_clicks"),
        State("data-gfw-key", "value"),
        State("data-gfw-start", "date"),
        State("data-gfw-end", "date"),
        State("data-gfw-flags", "value"),
        State("data-gfw-vtypes", "value"),
        prevent_initial_call=True,
    )
    def do_download(n, key, start, end, flags, vtypes):
        if not key:
            return "Please enter your API key.", dash.no_update, dash.no_update, dash.no_update
        if not start or not end:
            return "Please select dates.", dash.no_update, dash.no_update, dash.no_update

        ds = start[:10]
        de = end[:10]

        if flags and "ALL" in flags:
            resolved_flags = COUNTRY_FLAGS
        else:
            resolved_flags = flags or None

        # The filename is decided BEFORE the download starts (data is
        # streamed straight to disk), so it's based on the requested dates,
        # not the actual dates of the data (only known afterwards).
        flag_tag = "_".join((resolved_flags or ["ALL"])[:2])
        fname = f"VP_{flag_tag}_{ds}_{de}.csv"
        csv_path = str(GFW_DOWNLOAD_DIR / fname)

        result = {"rows": 0, "error": None}

        def _run():
            try:
                client = get_gfw_client(key)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                total_rows, actual_start, actual_end = loop.run_until_complete(
                    bulk_load_data_to_csv(resolved_flags, vtypes or None, ds, de, client, csv_path)
                )
                loop.close()
                if total_rows > 0:
                    result["rows"] = total_rows
                    result["path"] = csv_path
                    result["fname"] = fname
                else:
                    result["error"] = "No data returned. Try different filters or dates."
                    # The file may have been created empty (header only) or
                    # not at all -- clean it up either way.
                    p = Path(csv_path)
                    if p.exists():
                        p.unlink()
            except Exception as e:
                result["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()

        if result.get("error"):
            return f"Error: {result['error']}", dash.no_update, dash.no_update, dash.no_update

        # Success message tells the user how many rows were downloaded and
        # the exact path of the file on disk.
        msg = f"Download complete: {result['rows']:,} records saved to {result['path']}"
        info = f"Active dataset: {result['fname']} ({result['rows']:,} rows) — {result['path']}"
        # NOTE: we no longer send the full dataframe as JSON to the browser
        # (store-df) -- nothing consumed it, and it was tens of MB wasted on
        # every download. The other pages reload the CSV directly from disk
        # through their own file selector.
        return msg, result["path"], True, info
