"""
pages/data.py
=============
"Data" page: download from Global Fishing Watch.

Two side-by-side columns:
  - LEFT  : Vessel Presence (VP) -- AIS presence (existing behaviour)
  - RIGHT : Apparent Fishing Effort (AFE) -- fishing effort

The API key comes from get_api_key() (entered once via the pop-up); there is
no API key field on this page anymore.
"""

import asyncio
import threading
from datetime import date
from pathlib import Path
import tempfile, os

from api_key import get_api_key
import dash
import pandas as pd
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl, card, GFW_DOWNLOAD_DIR
from config import FLAG_NAMES
from gfw import (get_gfw_client, bulk_load_data_to_csv, bulk_load_afe_to_csv,
                GFW_VESSEL_TYPES, COUNTRY_FLAGS, GEAR_TYPES)

def _download_panel(prefix, title, subtitle, show_vtypes=True):
    """prefix = 'vp' or 'afe' -> used to prefix every id in the panel."""
    children = [
        html.H6(title, style={"color": MAIN, "marginBottom": "0.2rem"}),
        html.P(subtitle, style={"color": DIM, "fontSize": "0.75rem", "marginBottom": "1rem"}),

        html.Div([
            html.Div([
                lbl("Start date"),
                dcc.DatePickerSingle(id=f"{prefix}-start", date=date(2026, 1, 1),
                    display_format="YYYY-MM-DD",
                    min_date_allowed=date(2012, 1, 1), max_date_allowed=date(2026, 12, 31)),
            ], style={"marginRight": "1.5rem"}),
            html.Div([
                lbl("End date"),
                dcc.DatePickerSingle(id=f"{prefix}-end", date=date(2026, 1, 31),
                    display_format="YYYY-MM-DD",
                    min_date_allowed=date(2012, 1, 1), max_date_allowed=date(2026, 12, 31)),
            ]),
        ], style={"display": "flex", "marginBottom": "1rem"}),

        lbl("Country / Flag"),
        dcc.Dropdown(id=f"{prefix}-flags",
            options=[{"label": "ALL countries", "value": "ALL"}] +
                    [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f}
                     for f in COUNTRY_FLAGS],
            value=["GRC"], multi=True, placeholder="Select countries...",
            style={"color": "#000", "marginBottom": "1rem"}),
    ]

    if show_vtypes:
        children += [
            lbl("Vessel type (leave empty for ALL)"),
            dcc.Dropdown(id=f"{prefix}-vtypes",
                options=[{"label": t.capitalize(), "value": t} for t in GFW_VESSEL_TYPES],
                value=[], multi=True, placeholder="All types...",
                style={"color": "#000", "marginBottom": "1.2rem"}),
        ]
    else:
        children += [dcc.Store(id=f"{prefix}-vtypes", data=[])]

    # Gear type: NOT filterable server-side (the GFW endpoint doesn't
    # support it), but the API does return a "gear_type" column, so this
    # is applied to the downloaded rows before the file is sent.
    children += [
        lbl("Gear type (leave empty for ALL)"),
        dcc.Dropdown(id=f"{prefix}-gears",
            options=[{"label": g.replace("_", " ").title(), "value": g}
                     for g in GEAR_TYPES],
            value=[], multi=True, placeholder="All gear types...",
            style={"color": "#000", "marginBottom": "0.4rem"}),
        html.P("Applied to the downloaded rows (GFW can't filter gear at "
               "download time).",
               style={"color": DIM, "fontSize": "0.7rem", "marginBottom": "1.2rem"}),
    ]

    children += [
        html.Div([
            html.Button(f"Download {title}", id=f"{prefix}-btn", n_clicks=0,
                style={"padding": "0.6rem 1.4rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none", "borderRadius": "6px",
                       "cursor": "pointer", "fontWeight": "600", "marginRight": "1rem"}),
            dcc.Loading(id=f"{prefix}-loading", type="circle", color=ACC,
                style={"display": "inline-block"},
                children=html.Span(id=f"{prefix}-status",
                    style={"fontSize": "0.76rem", "color": SOFT, "fontStyle": "italic"})),
        ], style={"display": "flex", "alignItems": "center"}),
    ]

    return html.Div(card(children),
                    style={"flex": "1", "minWidth": "340px"})


# LAYOUT -- two columns
def layout():
    return html.Div([
        dcc.Download(id="vp-file-download"),
        dcc.Download(id="afe-file-download"),

        html.H5("Download from Global Fishing Watch",
                style={"color": MAIN, "marginBottom": "0.3rem"}),
        html.P("Select filters, then click Download. Your API key is asked once.",
               style={"color": DIM, "fontSize": "0.8rem", "marginBottom": "1.2rem"}),

        html.Div([
            _download_panel("vp", "Vessel Presence",
                            "AIS presence of vessels (positions).", show_vtypes=True),
            _download_panel("afe", "Fishing Effort (AFE)",
                            "Apparent fishing effort (fishing vessels only).", show_vtypes=True),
        ], style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap"}),

        html.Div(id="data-active-dataset",
                 style={"marginTop": "1rem", "padding": "0.6rem 1rem",
                        "background": BG, "borderRadius": "6px",
                        "border": f"1px solid {BDR}",
                        "fontSize": "0.78rem", "color": DIM}),

    ], style={"padding": "1.5rem", "background": BG, "minHeight": "calc(100vh - 52px)"})


# Download logic (VP or AFE) lives in _do_download() below, called by the callbacks.
def _post_filter_csv(csv_path, vtypes=None, gears=None):
    """Applies the filters GFW can't apply at download time, directly on
    the written CSV: gear_type (never filterable server-side) and, for
    AFE, vessel_type (bulk_load_afe_to_csv takes no vessel_types param).
    Rewrites the file in place. Returns the remaining row count."""
    if not vtypes and not gears:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return 0

    if vtypes and "vessel_type" in df.columns:
        wanted = [str(t).upper() for t in vtypes]
        df = df[df["vessel_type"].astype(str).str.upper().isin(wanted)]
    if gears and "gear_type" in df.columns:
        wanted = [str(g).upper() for g in gears]
        df = df[df["gear_type"].astype(str).str.upper().isin(wanted)]

    df.to_csv(csv_path, index=False)
    return len(df)


def _do_download(kind, start, end, flags, vtypes, gears=None):
    """kind = 'VP' or 'AFE'. Returns (message, csv_path_or_none, info)."""
    key = get_api_key()
    if not key:
        return "No API key saved.", None, None

    ds, de = start[:10], end[:10]
    if flags and "ALL" in flags:
        resolved_flags = COUNTRY_FLAGS
    else:
        resolved_flags = flags or None

    flag_tag = "_".join((resolved_flags or ["ALL"])[:2])
    fname = f"{kind}_{flag_tag}_{ds}_{de}.csv"
    csv_path = os.path.join(tempfile.gettempdir(), fname)

    result = {"rows": 0, "error": None}

    def _run():
        try:
            client = get_gfw_client(key)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            if kind == "VP":
                total_rows, _, _ = loop.run_until_complete(
                    bulk_load_data_to_csv(resolved_flags, vtypes or None, ds, de, client, csv_path))
            else:  # AFE
                loop.run_until_complete(
                    bulk_load_afe_to_csv(resolved_flags, ds, de, client, csv_path))
                try:
                    total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
                except Exception:
                    total_rows = 0
            loop.close()

            if total_rows and total_rows > 0:
                result["rows"] = total_rows
                result["path"] = csv_path
                result["fname"] = fname
            else:
                result["error"] = "No data returned. Try different filters or dates."
                p = Path(csv_path)
                if p.exists():
                    p.unlink()
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()

    if result.get("error"):
        return f"Error: {result['error']}", None, None

    # Filters GFW can't apply server-side: gear_type for both, plus
    # vessel_type for AFE (bulk_load_afe_to_csv takes no vessel_types).
    post_vtypes = vtypes if kind == "AFE" else None
    kept = _post_filter_csv(result["path"], vtypes=post_vtypes, gears=gears)
    if kept is not None:
        if kept == 0:
            p = Path(result["path"])
            if p.exists():
                p.unlink()
            return ("No data left after the vessel type / gear type filter. "
                    "Try widening the filters."), None, None
        result["rows"] = kept

    msg = f"Download complete: {result['rows']:,} records saved."
    info = f"Active dataset: {result['fname']} ({result['rows']:,} rows) — {result['path']}"
    return msg, result["path"], info


# CALLBACKS
def register_callbacks(app):

    # Instant feedback on click (VP and AFE)
    for prefix in ("vp", "afe"):
        app.clientside_callback(
            """
            function(n) {
                if (!n) { return window.dash_clientside.no_update; }
                return "Download in progress...";
            }
            """,
            Output(f"{prefix}-status", "children"),
            Input(f"{prefix}-btn", "n_clicks"),
            prevent_initial_call=True,
        )

    # Download VP
    @app.callback(
        Output("vp-status", "children", allow_duplicate=True),
        Output("vp-file-download", "data"),
        Input("vp-btn", "n_clicks"),
        State("vp-start", "date"), State("vp-end", "date"),
        State("vp-flags", "value"), State("vp-vtypes", "value"),
        State("vp-gears", "value"),
        prevent_initial_call=True,
    )
    def _dl_vp(n, start, end, flags, vtypes, gears):
        if not n:
            raise dash.exceptions.PreventUpdate
        msg, path, info = _do_download("VP", start, end, flags, vtypes, gears)
        if path is None:
            return msg, dash.no_update
        # read the file into memory, send it to the browser, then clean up disk
        send = dcc.send_file(path)
        try:
            os.remove(path)
        except OSError:
            pass
        return msg, send

    # Download AFE
    @app.callback(
        Output("afe-status", "children", allow_duplicate=True),
        Output("afe-file-download", "data"),
        Input("afe-btn", "n_clicks"),
        State("afe-start", "date"), State("afe-end", "date"),
        State("afe-flags", "value"), State("afe-vtypes", "value"),
        State("afe-gears", "value"),
        prevent_initial_call=True,
    )
    def _dl_afe(n, start, end, flags, vtypes, gears):
        if not n:
            raise dash.exceptions.PreventUpdate
        msg, path, info = _do_download("AFE", start, end, flags, vtypes, gears)
        if path is None:
            return msg, dash.no_update
        send = dcc.send_file(path)
        try:
            os.remove(path)
        except OSError:
            pass
        return msg, send