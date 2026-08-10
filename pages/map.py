"""
pages/map.py
============
"Map & Trajectories" page:
- Selection over precomputed data (date range / country / vessel type)
  -> trajectories on the map + vessel list + CSV export.
- Import a personal CSV -> show its trajectories on the map, with an
  optional gear_type filter.

v2: the vessel table is now a FIXED strip at the bottom of the page
(220px, scrollable), the map takes up all the remaining space.
Speed optimizations: vectorized tooltip (no more row-by-row .apply),
and a cap on the number of vessels drawn as trajectories (points are
still all shown as scatter, only the number of lines is capped) to
avoid freezing the browser on a very large selection.

v3: added a map legend (zones + vessel types), the date range is no
longer locked to a single calendar year (a selection can freely span
across two years, e.g. Dec 24 2024 -> Jan 15 2025), and the gear_type
filter now lives next to the CSV import controls, since that's the
only place it actually applies.
"""

import json
from datetime import date

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table

from shared import (
    BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY,
    lbl, card, build_deck, ZONE_LAYERS, FLAG_OPTIONS, GFW_DOWNLOAD_DIR,
)
from config import YEARS, VESSEL_TYPES, TYPE_COLORS, DEFAULT_COLOR, FLAG_NAMES, ROOT, ZONES
from loader import load_trajectories_range
from gfw import list_downloaded_csvs, load_csv, GEAR_TYPES

MAX_POINTS = 60_000    # max number of points shown as scatter (sampled beyond that)
MAX_PATHS_FAST = 500   # max number of vessels drawn as paths in "fast" mode (checkbox unticked)
MAX_PTS_PER_PATH = 80  # max number of points per path (decimation, keeps the overall shape)
TRAJECTORY_COLUMNS = ["lat", "lon", "vessel_id", "ship_name", "date", "flag", "vessel_type", "gear_type"]

# The full range of dates the app has data for -- the start/end date pickers
# are bounded by this instead of by a single calendar year, so a selection
# can freely cross a year boundary (e.g. Dec 2024 -> Jan 2025).
GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

# Server-side cache for the currently filtered selection (used by the
# Export CSV button). This avoids sending the whole dataframe as JSON to
# the browser just so it can be re-uploaded for download -- that used to
# be one of the causes of slowness / freezing on large selections.
_LAST_FILTERED_DF = {"df": None}


def _sidebar_section(title, children):
    return html.Div([
        html.H6(title, style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
        *children,
    ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
              "borderBottom": f"1px solid {BDR}"})


def _legend_swatch(color_rgba, label, shape="square"):
    """One legend row: a small color swatch (square for zones, round dot
    for vessel types) next to its label."""
    r, g, b = color_rgba[0], color_rgba[1], color_rgba[2]
    a = (color_rgba[3] / 255) if len(color_rgba) > 3 else 1
    swatch_style = {
        "width": "11px", "height": "11px", "marginRight": "0.5rem",
        "flexShrink": "0",
        "backgroundColor": f"rgba({r},{g},{b},{a})",
        "border": f"1px solid rgba({r},{g},{b},1)",
        "borderRadius": "2px" if shape == "square" else "50%",
    }
    return html.Div([
        html.Span(style=swatch_style),
        html.Span(label, style={"fontSize": "0.72rem", "color": SOFT}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.3rem"})


def _legend_subtitle(text):
    return html.P(text, style={"fontSize": "0.66rem", "color": DIM, "fontWeight": "600",
                                "textTransform": "uppercase", "letterSpacing": "0.04em",
                                "margin": "0.6rem 0 0.35rem 0"})


def _vessel_type_checklist():
    """Returns a checklist of vessel types with colored dots, plus a "select all"""
    type_options = []
    for t in VESSEL_TYPES:
        c = TYPE_COLORS.get(t, DEFAULT_COLOR)
        dot = html.Span(style={
            "display": "inline-block", "width": "11px", "height": "11px",
            "borderRadius": "50%", "marginRight": "7px",
            "backgroundColor": f"rgba({c[0]},{c[1]},{c[2]},{(c[3]/255) if len(c) > 3 else 1})",
            "verticalAlign": "middle",
        })
        label = html.Span([dot, t.capitalize()],
                          style={"fontSize": "0.72rem", "color": SOFT,
                                 "verticalAlign": "middle"})
        type_options.append({"label": label, "value": t})

    select_all = html.Button(
        "Deselect all",
        id="map-type-select-all",
        n_clicks=0,
        style={"fontSize": "0.7rem", "color": SOFT, "background": "none",
               "border": f"1px solid {BDR}", "borderRadius": "4px",
               "padding": "3px 8px", "marginBottom": "6px", "cursor": "pointer"},
    )

    checklist = dcc.Checklist(
        id="map-type-filter",
        options=type_options,
        value=list(VESSEL_TYPES),
        labelStyle={"display": "flex", "alignItems": "center",
                    "marginBottom": "3px", "cursor": "pointer"},
        inputStyle={"marginRight": "6px"},
    )
    return html.Div([select_all, checklist])


def _legend_section():
    """Legend for zones and vessel types, in the sidebar."""
    zone_rows = [_legend_swatch(z["line_color"], z["label"], shape="square")
                 for z in ZONES.values()]
    return _sidebar_section("Legend", [
        _legend_subtitle("Zones"),
        *zone_rows,
    ])

def layout():
    return html.Div([
        dcc.Store(id="map-store-filtered-df", data=None),
        dcc.Store(id="map-sidebar-open", data=True),
        dcc.Download(id="map-download-csv"),

        # ── Sidebar: spans the FULL height of the side (map + table included) ──
        html.Div([


            _sidebar_section("Precomputed data", [
                lbl("Jump to a year (optional)"),
                dcc.Dropdown(id="map-year", value=None, clearable=True,
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    placeholder="Jump to a year...",
                    style={"color": "#000", "marginBottom": "0.8rem"}),

                lbl("Start date"),
                dcc.DatePickerSingle(id="map-start", date=date(YEARS[-1], 1, 1),
                    display_format="YYYY-MM-DD",
                    min_date_allowed=GLOBAL_MIN_DATE,
                    max_date_allowed=GLOBAL_MAX_DATE,
                    style={"marginBottom": "0.6rem"}),
                lbl("End date"),
                dcc.DatePickerSingle(id="map-end", date=date(YEARS[-1], 1, 31),
                    display_format="YYYY-MM-DD",
                    min_date_allowed=GLOBAL_MIN_DATE,
                    max_date_allowed=GLOBAL_MAX_DATE,
                    style={"marginBottom": "0.6rem"}),
                html.P("The range can span across two years (e.g. Dec 2024 -> Jan 2025).",
                       style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                              "marginBottom": "0.6rem"}),

                lbl("Country (flag)"),
                dcc.Dropdown(id="map-flag", options=FLAG_OPTIONS, value=[], multi=True,
                    placeholder="All countries...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),

                dcc.Dropdown(id="map-vtype",
                    options=[{"label": t.capitalize(), "value": t} for t in VESSEL_TYPES],
                    value=[], multi=True,
                    style={"display": "none"}),
                dcc.Checklist(
                    id="map-show-all-paths",
                    options=[{"label": " Show all trajectories (can be slow on a large selection)",
                              "value": "all"}],
                    value=[],
                    style={"fontSize": "0.72rem", "color": SOFT, "marginBottom": "0.8rem"},
                ),

                lbl("Vessel types (tick to show)"),
                html.Div(_vessel_type_checklist(), style={"marginBottom": "0.8rem"}),

                html.Button("Show", id="map-btn-show", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none",
                           "borderRadius": "6px", "cursor": "pointer",
                           "fontWeight": "600"}),
            ]),

            _sidebar_section("Import a downloaded CSV", [
                dcc.Dropdown(id="map-csv-selector",
                    options=[{"label": f["filename"], "value": f["path"]}
                             for f in list_downloaded_csvs(ROOT / "data")],
                    value=None, placeholder="Choose a CSV...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                lbl("Gear type filter (optional)"),
                html.P("Only applies to CSVs downloaded from GFW, which "
                       "include a gear_type column (e.g. keep only Trawlers).",
                       style={"fontSize": "0.68rem", "color": DIM, "marginBottom": "0.5rem"}),
                dcc.Dropdown(id="map-gear",
                    options=[{"label": g.replace("_", " ").capitalize(), "value": g}
                             for g in GEAR_TYPES],
                    value=[], multi=True, placeholder="All gear types...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                html.Button("Show this CSV", id="map-btn-show-csv", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem",
                           "background": PANEL, "color": SOFT,
                           "border": f"1px solid {BDR}",
                           "borderRadius": "6px", "cursor": "pointer",
                           "marginBottom": "0.4rem"}),
                html.Div(id="map-csv-status",
                         style={"fontSize": "0.72rem", "color": SOFT}),
            ]),

            _legend_section(),

            html.Div(id="map-stats", style={"fontSize": "0.75rem", "color": SOFT}),



        ], id="map-sidebar", style={"width": "260px", "minWidth": "260px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto",
                   "flexShrink": "0"}),

        html.Button("‹", id="map-toggle-sidebar", n_clicks=0, title="Show/hide panel",
            style={"width": "22px", "minWidth": "22px", "border": "none",
                   "background": PANEL, "color": MAIN, "cursor": "pointer",
                   "fontSize": "1.1rem", "fontWeight": "700",
                   "borderRight": f"1px solid {BDR}", "flexShrink": "0"}),

        # ── Right column: map fills the space; Export button top-right ──
        html.Div([
            html.Div(
                html.Button("Export CSV", id="map-btn-export", n_clicks=0,
                    style={"border": "none",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                           "fontWeight": "600",
                           "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": f"1px solid {BDR}", "flexShrink": "0",
                       "display": "flex", "justifyContent": "flex-end"},
            ),

            html.Div([
                dcc.Loading(
                    type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0, "position": "relative"}),

            html.Div(id="map-vessel-list", style={"display": "none"}),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _vessel_list_table(df):
    if df.empty or "vessel_id" not in df.columns:
        return html.P("No vessels in the selection.",
                       style={"color": DIM, "fontStyle": "italic"})
    cols = [c for c in ["vessel_id", "ship_name", "flag", "vessel_type"] if c in df.columns]
    summary = df[cols].drop_duplicates(subset="vessel_id").copy()
    if "flag" in summary.columns:
        summary["flag"] = summary["flag"].map(lambda f: f"{FLAG_NAMES.get(f, f)} ({f})")
    return html.Div([
        html.P(f"{len(summary):,} vessels in the selection",
               style={"color": SOFT, "fontSize": "0.78rem", "marginBottom": "0.4rem"}),
        dash_table.DataTable(
            data=summary.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
            page_size=8,
            export_format="csv",
            export_headers="display",
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": PANEL, "color": MAIN, "border": f"1px solid {BDR}",
                        "fontSize": "0.75rem", "padding": "4px 8px"},
            style_header={"backgroundColor": BG, "color": DIM, "fontWeight": "600"},
        ),
    ])


def _apply_gear_filter(df, gear):
    """Filter by gear_type if the column exists (precomputed trajectories
    don't have it; CSVs downloaded from GFW do)."""
    if df is None or df.empty or not gear:
        return df
    if "gear_type" not in df.columns:
        return df
    gear_upper = [g.upper() for g in gear]
    return df[df["gear_type"].astype(str).str.upper().isin(gear_upper)]


def _build_map(df, show_all_paths=False):
    layers = list(ZONE_LAYERS.values())
    note = None

    if df is not None and not df.empty and "lat" in df.columns and "lon" in df.columns:
        df_plot = df.copy()

        if "vessel_type" in df_plot.columns:
            vtype_upper = df_plot["vessel_type"].astype(str).str.upper()
            df_plot["color"] = vtype_upper.map(lambda t: TYPE_COLORS.get(t, DEFAULT_COLOR))
        else:
            df_plot["color"] = [DEFAULT_COLOR] * len(df_plot)

        # Vectorized tooltip (fast, no row-by-row .apply)
        ship = df_plot["ship_name"].astype(str) if "ship_name" in df_plot.columns else "?"
        flag_lbl = (df_plot["flag"].map(lambda f: FLAG_NAMES.get(f, f))
                    if "flag" in df_plot.columns else "?")
        vtype = df_plot["vessel_type"].astype(str) if "vessel_type" in df_plot.columns else "?"
        df_plot["tooltip"] = ship + " (" + flag_lbl.astype(str) + ") - " + vtype.astype(str)

        # Scatter: sample down if there are too many points (keeps the map smooth)
        scatter_df = df_plot
        if len(scatter_df) > MAX_POINTS:
            scatter_df = scatter_df.sample(MAX_POINTS, random_state=0)
            note = f"Sampled display: {MAX_POINTS:,} / {len(df_plot):,} points"

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=scatter_df,
            get_position=["lon", "lat"], get_fill_color="color",
            get_radius=400, radius_min_pixels=2, radius_max_pixels=8,
            pickable=True, auto_highlight=True,
        ))

        # Trajectories: one line per vessel. By default (checkbox unticked)
        # we cap at MAX_PATHS_FAST vessels to keep the map smooth -- tick
        # "Show all trajectories" to draw all of them regardless of count
        # (can be slow on a large selection).
        if "vessel_id" in df.columns and "date" in df.columns:
            vessel_ids = df_plot["vessel_id"].unique()
            if not show_all_paths and len(vessel_ids) > MAX_PATHS_FAST:
                vessel_ids_kept = vessel_ids[:MAX_PATHS_FAST]
                note = (note + " · " if note else "") + \
                       (f"Trajectories limited to {MAX_PATHS_FAST:,} vessels out of "
                        f"{len(vessel_ids):,} (tick \"Show all trajectories\" to see them all)")
                sub = df_plot[df_plot["vessel_id"].isin(vessel_ids_kept)]
            else:
                sub = df_plot

            sub = sub.sort_values(["vessel_id", "date"])
            paths = []
            for vid, g in sub.groupby("vessel_id", sort=False):
                coords = g[["lon", "lat"]].values.tolist()
                # Decimation: keep at most MAX_PTS_PER_PATH points per vessel
                # (overall shape preserved, payload sent to the browser much
                # lighter -- this is what was slowing down / freezing the map).
                if len(coords) > MAX_PTS_PER_PATH:
                    step = len(coords) / MAX_PTS_PER_PATH
                    coords = [coords[int(i * step)] for i in range(MAX_PTS_PER_PATH)]
                if len(coords) > 1:
                    paths.append({
                        "path": coords,
                        "color": g["color"].iloc[0],
                        "tooltip": g["tooltip"].iloc[0],
                    })
            if paths:
                layers.append(pdk.Layer(
                    "PathLayer", data=paths,
                    get_path="path", get_color="color",
                    get_width=2, width_min_pixels=1.5, pickable=True,
                ))

    deck = build_deck(layers)
    map_widget = dash_deck.DeckGL(data=deck, mapboxKey=MAPBOX_KEY, style={"width": "100%", "height": "100%"})
    return map_widget, note


def register_callbacks(app):

    # Sync the "Select / deselect all" master checkbox with the type list.
    # Bidirectional: clicking the master toggles every type; ticking the
    # individual boxes updates whether the master appears checked.
    @app.callback(
        Output("map-type-filter", "value"),
        Output("map-type-select-all", "value"),
        Input("map-type-select-all", "value"),
        Input("map-type-filter", "value"),
    )
    def _sync_select_all(master, selected):
        trigger = (dash.callback_context.triggered_id
                   if dash.callback_context.triggered else None)
        if trigger == "map-type-select-all":
            if "all" in (master or []):
                return list(VESSEL_TYPES), ["all"]
            return [], []
        master_val = ["all"] if set(selected or []) == set(VESSEL_TYPES) else []
        return selected, master_val
    @app.callback(
        Output("map-type-filter", "value"),
        Output("map-type-select-all", "children"),
        Input("map-type-select-all", "n_clicks"),
        State("map-type-filter", "value"),
        prevent_initial_call=True,
    )
    def _toggle_all_types(n, selected):
        # If anything is selected, clear it; otherwise select everything.
        if selected:
            return [], "Select all"
        return list(VESSEL_TYPES), "Deselect all"
    # "Jump to a year" is just a convenience: it fills in Jan 1 -> Dec 31
    # of the chosen year, but doesn't restrict the pickers -- the user is
    # still free to edit either date afterwards, including across a year
    # boundary (e.g. change the end date to the following January).
    @app.callback(
        Output("map-sidebar", "style"),
        Output("map-toggle-sidebar", "children"),
        Output("map-sidebar-open", "data"),
        Input("map-toggle-sidebar", "n_clicks"),
        State("map-sidebar-open", "data"),
        prevent_initial_call=True,
    )
    def _toggle_sidebar(n, is_open):
        now_open = not is_open
        if now_open:
            style = {"width": "260px", "minWidth": "260px", "padding": "1rem",
                     "background": BG, "borderRight": f"1px solid {BDR}",
                     "height": "calc(100vh - 52px)", "overflowY": "auto",
                     "flexShrink": "0"}
            arrow = "‹"
        else:
            style = {"width": "0px", "minWidth": "0px", "padding": "0",
                     "background": BG, "borderRight": f"1px solid {BDR}",
                     "height": "calc(100vh - 52px)", "overflow": "hidden",
                     "flexShrink": "0"}
            arrow = "›"
        return style, arrow, now_open

    @app.callback(
        Output("map-container", "children"),
        Output("map-vessel-list", "children"),
        Output("map-stats", "children"),
        Output("map-store-filtered-df", "data"),
        Output("map-csv-status", "children"),
        Input("map-btn-show", "n_clicks"),
        Input("map-btn-show-csv", "n_clicks"),
        State("map-start", "date"),
        State("map-end", "date"),
        State("map-flag", "value"),
        State("map-vtype", "value"),
        State("map-gear", "value"),
        State("map-show-all-paths", "value"),
        State("map-csv-selector", "value"),
        State("map-type-filter", "value"),
        prevent_initial_call=False,
    )
    def update_map(n1, n2, start, end, flags, vtypes, gear, show_all, csv_path,
                   type_filter):
        trigger = dash.callback_context.triggered_id if dash.callback_context.triggered else None
        show_all_paths = "all" in (show_all or [])

        if trigger == "map-btn-show-csv":
            if not csv_path:
                map_c, _ = _build_map(pd.DataFrame())
                return map_c, _vessel_list_table(pd.DataFrame()), "", None, "Choose a CSV first."
            try:
                df = load_csv(csv_path)
            except Exception as e:
                map_c, _ = _build_map(pd.DataFrame())
                return map_c, _vessel_list_table(pd.DataFrame()), "", None, f"Error: {e}"

            df = _apply_gear_filter(df, gear)
            map_c, note = _build_map(df, show_all_paths)
            n_v = df["vessel_id"].nunique() if "vessel_id" in df.columns else len(df)
            stats_children = [html.P(f"{len(df):,} positions · {n_v:,} vessels (imported CSV)",
                                      style={"color": SOFT, "fontSize": "0.78rem"})]
            if note:
                stats_children.append(html.P(note, style={"color": DIM, "fontSize": "0.7rem", "fontStyle": "italic"}))
            status = f"CSV loaded: {len(df):,} rows"
            _LAST_FILTERED_DF["df"] = df
            return map_c, _vessel_list_table(df), html.Div(stats_children), (True if not df.empty else None), status

        if not start or not end:
            map_c, _ = _build_map(pd.DataFrame())
            return map_c, _vessel_list_table(pd.DataFrame()), "Select a date range.", None, ""

        # Be forgiving if the user picks the dates in the "wrong" order
        # (e.g. end date before start date) -- just swap them instead of
        # erroring out or returning an empty result.
        if pd.to_datetime(start) > pd.to_datetime(end):
            start, end = end, start

        df = load_trajectories_range(start, end, vtypes or None, flags or None, columns=TRAJECTORY_COLUMNS)
        #Filter by vessel type if the column exists (precomputed trajectories have it, but imported CSVs may not)
        if type_filter is not None and "vessel_type" in df.columns:
            df = df[df["vessel_type"].astype(str).str.upper().isin(
                [t.upper() for t in type_filter])]
        df = _apply_gear_filter(df, gear)
        map_c, note = _build_map(df, show_all_paths)
        if df.empty:
            stats = html.P("No data for this selection.", style={"color": DIM})
        else:
            n_v = df["vessel_id"].nunique()
            stats_children = [html.P(f"{len(df):,} positions · {n_v:,} vessels",
                                      style={"color": SOFT, "fontSize": "0.78rem"})]
            if note:
                stats_children.append(html.P(note, style={"color": DIM, "fontSize": "0.7rem", "fontStyle": "italic"}))
            stats = html.Div(stats_children)
        _LAST_FILTERED_DF["df"] = df
        return map_c, _vessel_list_table(df), stats, (True if not df.empty else None), ""

    @app.callback(
        Output("map-download-csv", "data"),
        Input("map-btn-export", "n_clicks"),
        State("map-store-filtered-df", "data"),
        prevent_initial_call=True,
    )
    def export_csv(n, has_data):
        if not has_data or _LAST_FILTERED_DF["df"] is None or _LAST_FILTERED_DF["df"].empty:
            return dash.no_update
        df = _LAST_FILTERED_DF["df"].copy()
        if "gear_type" not in df.columns:
            df["gear_type"] = ""
        cols = list(df.columns)
        if "gear_type" in cols and "vessel_type" in cols:
            cols.remove("gear_type")
            idx = cols.index("vessel_type") + 1
            cols.insert(idx, "gear_type")
            df = df[cols]
        return dcc.send_data_frame(df.to_csv, "selection_trajectories.csv", index=False)