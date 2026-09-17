"""
pages/ais_gap.py
=================
"AIS Gaps" page -- finds vessels that stopped transmitting AIS for an
unusually long time (a possible sign of intentional AIS disabling),
either:
  - across all vessels matching a flag / vessel-type filter, or
  - for a single vessel found via search (name / MMSI / IMO), same
    search UX as pages/ports.py.

Two data sources (see gfw.py for the full rationale, right above
load_gap_events / compute_client_side_gap_events):
  - "reconstructed" (default): diffs consecutive hourly AIS-presence
    pings client-side. No distance-from-shore cutoff, so it actually
    sees near-coast gaps -- important in the Aegean, an archipelago
    where the official dataset below returns close to nothing.
  - "official": GFW's own "public-global-gaps-events" dataset. Cleaner
    (server-computed, real timestamped events, includes distance_km /
    implied_speed_knots) but only publishes gaps starting >=50 nautical
    miles from shore.

Every gap is also classified "suspicious" / "normal" against the AIS
coverage buffer (data/gis/ais_buffer_<N>nm.geojson, see
classify_gap_status in gfw.py): "suspicious" means BOTH the off and on
positions fall outside the buffer (genuinely offshore -- the real
signal), "normal" means at least one is near the buffer edge (more
likely a reception artifact than real AIS disabling). "gap" means the
buffer file wasn't found on disk, i.e. unclassified.
"""

import asyncio
from datetime import date, timedelta

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table

from shared import (
    BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, GOOD, BAD, WARN, MAPBOX_KEY,
    lbl, build_deck, ZONE_LAYERS,
)
from config import YEARS, FLAG_NAMES
from gfw import (
    get_gfw_client, COUNTRY_FLAGS, GFW_VESSEL_TYPES,
    bulk_load_gap_events_dataframe, bulk_load_client_gap_events_dataframe,
    load_ais_buffer_polygon, classify_gap_status,
)
from api_key import get_api_key
from pages.ports import do_search_vessel, _group_results_by_identity

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

# Default window kept short on purpose: the "reconstructed" source
# downloads raw hourly pings month by month (see bulk_load_vp_dataframe
# in gfw.py) -- a long period combined with "all flags / all types" can
# be very slow. Narrow flags/types or use the official source for a
# wider scan.

BUFFER_OPTIONS = [1, 3, 5, 10]

FLAG_OPTIONS_GFW = [{"label": f"{FLAG_NAMES.get(f, f)} ({f})", "value": f} for f in COUNTRY_FLAGS]
VESSEL_TYPE_OPTIONS_GFW = [{"label": t.replace("_", " ").capitalize(), "value": t} for t in GFW_VESSEL_TYPES]


# CALL SYNC GFW FUNCTIONS FROM A FRESH ASYNCIO LOOP (same pattern as pages/ports.py)
def do_load_gaps(source, flags, vessel_types, vessel_ids, start, end, api_key, min_hours):
    client = get_gfw_client(api_key)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if source == "official":
            df = loop.run_until_complete(
                bulk_load_gap_events_dataframe(flags, vessel_types, start, end, client,
                                                vessel_ids=vessel_ids)
            )
            if df is not None and not df.empty and "duration_hrs" in df.columns and min_hours:
                df = df[df["duration_hrs"] >= float(min_hours)]
        else:
            df = loop.run_until_complete(
                bulk_load_client_gap_events_dataframe(flags, vessel_types, start, end, client,
                                                       vessel_ids=vessel_ids,
                                                       min_hours=float(min_hours or 0))
            )
    finally:
        loop.close()
    return df if df is not None else pd.DataFrame()


def _classify(df, buffer_nm):
    """Adds a 'status' column (suspicious / normal / gap) to every row,
    using the same AIS-coverage buffer as pages/alerts.py."""
    if df is None or df.empty:
        return df
    buffer_geom = load_ais_buffer_polygon(buffer_nm)
    df = df.copy()
    df["status"] = df.apply(
        lambda r: classify_gap_status(r.get("off_lat"), r.get("off_lon"),
                                       r.get("on_lat"), r.get("on_lon"), buffer_geom),
        axis=1,
    )
    return df


# LAYOUT

def _sidebar_section(title, children, extra_style=None):
    style = {"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
             "borderBottom": f"1px solid {BDR}"}
    if extra_style:
        style.update(extra_style)
    return html.Div([
        html.H6(title, style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
        *children,
    ], style=style)


def layout():
    return html.Div([
        dcc.Store(id="gap-search-store", data=None),
        dcc.Store(id="gap-store", data=None),
        dcc.Download(id="gap-download-csv"),

        html.Div([
            html.H6("AIS gap detection", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P("Finds vessels that stopped transmitting AIS for an unusually "
                   "long time -- a possible sign of intentional disabling.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            _sidebar_section("Scope", [
                dcc.RadioItems(id="gap-mode",
                    options=[
                        {"label": " All vessels (flag / type)", "value": "flags"},
                        {"label": " Single vessel", "value": "vessel"},
                    ],
                    value="flags",
                    labelStyle={"display": "block", "marginBottom": "5px",
                                "fontSize": "0.75rem", "color": SOFT, "cursor": "pointer"}),
            ]),

            html.Div(id="gap-flags-block", children=[
                lbl("Flag(s)"),
                dcc.Dropdown(id="gap-flags", options=FLAG_OPTIONS_GFW, value=[], multi=True,
                    placeholder="All flags...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                lbl("Vessel type(s)"),
                dcc.Dropdown(id="gap-vessel-types", options=VESSEL_TYPE_OPTIONS_GFW, value=[], multi=True,
                    placeholder="All vessel types...",
                    style={"color": "#000", "marginBottom": "0.4rem"}),
                html.P("Leave both empty to scan every vessel in the region -- "
                       "correct, but the slowest option.",
                       style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                              "marginBottom": "0.6rem"}),
            ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
                      "borderBottom": f"1px solid {BDR}"}),

            html.Div(id="gap-vessel-block", children=[
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
                        style={"maxHeight": "180px", "overflowY": "auto",
                               "border": "1px solid " + BDR, "borderRadius": "6px",
                               "padding": "0.5rem", "marginBottom": "0.4rem", "background": BG},
                    )),
                html.Div(id="gap-selected", style={"fontSize": "0.72rem", "color": ACC,
                                                    "fontWeight": "600", "marginBottom": "0.4rem"}),
            ], style={"display": "none", "marginBottom": "1.2rem", "paddingBottom": "1.2rem",
                      "borderBottom": f"1px solid {BDR}"}),

            _sidebar_section("Period & sensitivity", [
                lbl("Start date"),
                dcc.DatePickerSingle(id="gap-start", date=DEFAULT_START,
                    display_format="YYYY-MM-DD",
                    min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                    style={"marginBottom": "0.6rem"}),
                lbl("End date"),
                dcc.DatePickerSingle(id="gap-end", date=DEFAULT_END,
                    display_format="YYYY-MM-DD",
                    min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                    style={"marginBottom": "0.8rem"}),

                lbl("Minimum gap duration (hours)"),
                dcc.Input(id="gap-min-hours", type="number", min=0, step=0.5, value=2,
                    style={"width": "100%", "padding": "0.4rem", "marginBottom": "0.6rem",
                           "borderRadius": "5px", "border": "1px solid " + BDR,
                           "background": PANEL, "color": MAIN}),

                lbl("AIS coverage buffer"),
                dcc.Dropdown(id="gap-buffer-nm", clearable=False,
                    options=[{"label": f"{n} nm", "value": n} for n in BUFFER_OPTIONS],
                    value=3, style={"color": "#000", "marginBottom": "0.6rem"}),
            ]),

            _sidebar_section("Data source", [
                dcc.RadioItems(id="gap-source",
                    options=[
                        {"label": " Reconstructed (recommended)", "value": "reconstructed"},
                        {"label": " Official GFW dataset (>=50nm offshore only)", "value": "official"},
                    ],
                    value="reconstructed",
                    labelStyle={"display": "block", "marginBottom": "5px",
                                "fontSize": "0.75rem", "color": SOFT, "cursor": "pointer"}),
                html.P("Reconstructed is the slowest option: GFW builds one detailed "
                       "hourly report per month of the period, sequentially -- there's "
                       "no way around that server-side. Narrow flag(s)/vessel type(s) "
                       "above and shorten the period to cut this down; switch to "
                       "\"Official\" for a near-instant (but coast-blind) result.",
                       style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                              "marginTop": "0.4rem"}),
            ]),

            html.Button("Get AIS gaps", id="gap-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": "linear-gradient(135deg,#d15400,#a03e00)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),

            dcc.Loading(id="gap-loading", type="dot", color=ACC,
                children=html.Div(id="gap-status", style={"fontSize": "0.72rem", "color": SOFT, "minHeight": "1rem"})),
            html.Div(id="gap-summary", style={"fontSize": "0.75rem", "color": SOFT, "marginTop": "0.4rem"}),

        ], style={"width": "320px", "minWidth": "320px", "padding": "1rem",
                   "background": BG, "borderRight": "1px solid " + BDR,
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        html.Div([
            html.Div([
                dcc.Checklist(id="gap-filter-suspicious",
                    options=[{"label": " Suspicious gaps only", "value": "suspicious_only"}],
                    value=[],
                    style={"fontSize": "0.75rem", "color": SOFT}),
                html.Div([
                    html.Button("Export CSV", id="gap-btn-export", n_clicks=0,
                        style={"border": "none",
                               "background": "linear-gradient(135deg," + ACC + ",#0d4a7a)",
                               "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                               "fontWeight": "600", "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                ]),
            ], style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": "1px solid " + BDR, "flexShrink": "0",
                       "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),

            html.Div(
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="gap-map-container", style={"height": "100%", "width": "100%"},
                        children=html.P("Choose a scope and period, then click \"Get AIS gaps\" -- "
                                        "the map will appear automatically, with the AIS coverage "
                                        "zones as reference.",
                                        style={"color": DIM, "fontSize": "0.8rem", "padding": "1rem"}))),
                style={"flex": "1", "minHeight": 0,
                       "borderBottom": "1px solid " + BDR, "position": "relative"},
            ),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _table(df):
    if df is None or df.empty:
        return html.P("No AIS gap found for this selection.", style={"color": SOFT, "fontSize": "0.8rem"})
    show = df.copy()
    if "flag" in show.columns:
        show["flag"] = show["flag"].map(lambda f: FLAG_NAMES.get(f, f) if pd.notna(f) else "?")
    cols = ["ship_name", "mmsi", "flag", "vessel_type", "gear_type", "start", "end",
            "duration_hrs", "distance_km", "implied_speed_knots", "status"]
    cols = [c for c in cols if c in show.columns]
    return dash_table.DataTable(
        data=show[cols].to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        sort_action="native", filter_action="native", page_size=30,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": "1px solid " + BDR,
                    "fontSize": "0.75rem", "padding": "4px 8px"},
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
        style_data_conditional=[
            {"if": {"filter_query": '{status} = "suspicious"'}, "color": BAD, "fontWeight": "600"},
            {"if": {"filter_query": '{status} = "normal"'}, "color": GOOD},
            {"if": {"filter_query": '{status} = "gap"'}, "color": WARN},
        ],
    )


def _build_gap_map(df):
    if df is None or df.empty:
        return html.P("No gap to display for the current filter.",
                       style={"color": DIM, "fontSize": "0.8rem", "padding": "1rem"})
    pts = df.dropna(subset=["off_lat", "off_lon", "on_lat", "on_lon"]).copy()
    if pts.empty:
        return html.P("No positioned gap in the current selection.",
                       style={"color": DIM, "fontSize": "0.8rem", "padding": "1rem"})
    if "status" not in pts.columns:
        pts["status"] = "gap"

    pts["tooltip"] = pts.apply(
        lambda r: (f"{r.get('ship_name') or 'Unknown vessel'} "
                   f"({FLAG_NAMES.get(r.get('flag'), r.get('flag')) or '?'})\n"
                   f"{float(r.get('duration_hrs') or 0):.1f}h gap -- {r.get('status', '?')}"),
        axis=1,
    )

    susp = pts[pts["status"] == "suspicious"]
    other = pts[pts["status"] != "suspicious"]

    # Zone overlays (territorial waters / EEZ / protected areas) as
    # context, same layers as pages/map.py -- this is what gives the
    # dashed-boundary look of the old desktop app's folium map.
    layers = list(ZONE_LAYERS.values())

    # Line connecting each gap's off -> on position.
    if not pts.empty:
        layers.append(pdk.Layer("LineLayer", data=pts.to_dict("records"),
            get_source_position="[off_lon, off_lat]", get_target_position="[on_lon, on_lat]",
            get_color=[224, 176, 112, 120], get_width=2, pickable=False))

    # "Normal" gaps: small, dim -- background noise, not the signal.
    if not other.empty:
        other_records = other.to_dict("records")
        layers.append(pdk.Layer("ScatterplotLayer", data=other_records,
            get_position="[off_lon, off_lat]", get_fill_color=[224, 112, 112, 110],
            get_radius=500, pickable=True))
        layers.append(pdk.Layer("ScatterplotLayer", data=other_records,
            get_position="[on_lon, on_lat]", get_fill_color=[97, 211, 155, 110],
            get_radius=500, pickable=True))

    # "Suspicious" gaps: highlighted -- bigger, brighter, white outline,
    # drawn last so they always sit on top of everything else.
    if not susp.empty:
        susp_records = susp.to_dict("records")
        layers.append(pdk.Layer("ScatterplotLayer", data=susp_records,
            get_position="[off_lon, off_lat]", get_fill_color=[224, 112, 112, 235],
            get_radius=1300, stroked=True, get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=1.5, pickable=True, auto_highlight=True))
        layers.append(pdk.Layer("ScatterplotLayer", data=susp_records,
            get_position="[on_lon, on_lat]", get_fill_color=[97, 211, 155, 235],
            get_radius=1300, stroked=True, get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=1.5, pickable=True, auto_highlight=True))

    deck = build_deck(layers)
    return html.Div([
        dash_deck.DeckGL(data=deck, mapboxKey=MAPBOX_KEY, style={"width": "100%", "height": "100%"}),
        html.Div([
            html.Span("\u25cf off (gap start)", style={"color": BAD, "fontSize": "0.68rem", "marginRight": "1rem"}),
            html.Span("\u25cf on (gap end)", style={"color": GOOD, "fontSize": "0.68rem", "marginRight": "1rem"}),
            html.Span("large + white outline = suspicious",
                      style={"color": SOFT, "fontSize": "0.66rem", "fontStyle": "italic"}),
        ], style={"position": "absolute", "bottom": "8px", "left": "8px",
                   "background": "rgba(7,17,29,0.78)", "padding": "0.3rem 0.6rem",
                   "borderRadius": "6px"}),
    ], style={"position": "relative", "height": "100%", "width": "100%"})


# CALLBACKS

def register_callbacks(app):

    @app.callback(
        Output("gap-flags-block", "style"),
        Output("gap-vessel-block", "style"),
        Input("gap-mode", "value"),
    )
    def _toggle_mode(mode):
        base = {"marginBottom": "1.2rem", "paddingBottom": "1.2rem", "borderBottom": "1px solid " + BDR}
        shown = {**base, "display": "block"}
        hidden = {**base, "display": "none"}
        if mode == "vessel":
            return hidden, shown
        return shown, hidden

    @app.callback(
        Output("gap-vessel-selector", "options"),
        Output("gap-vessel-selector", "value"),
        Output("gap-search-store", "data"),
        Output("gap-status", "children", allow_duplicate=True),
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
        Output("gap-store", "data"),
        Output("gap-status", "children"),
        Input("gap-btn-run", "n_clicks"),
        State("gap-mode", "value"),
        State("gap-source", "value"),
        State("gap-flags", "value"),
        State("gap-vessel-types", "value"),
        State("gap-vessel-selector", "value"),
        State("gap-search-store", "data"),
        State("gap-start", "date"),
        State("gap-end", "date"),
        State("gap-min-hours", "value"),
        State("gap-buffer-nm", "value"),
        prevent_initial_call=True,
    )
    def _run(n, mode, source, flags, vessel_types, idx, entries, start, end, min_hours, buffer_nm):
        if not n:
            raise dash.exceptions.PreventUpdate
        api_key = get_api_key()
        if not api_key:
            return None, "No API key saved."
        if not start or not end:
            return None, "Choose a start and end date."

        vessel_ids = None
        if mode == "vessel":
            if idx is None or not entries:
                return None, "Search and select a vessel first."
            info = entries[int(idx)]
            vessel_ids = info["ids"]
            # IMPORTANT: still pass the vessel's own flag/vessel_type as a
            # server-side narrowing hint (see load_VP_data_by_vessel in
            # gfw.py). Without this, a single-vessel lookup downloads every
            # vessel's raw hourly pings for the whole region/period before
            # filtering down to the one requested -- correct, but so slow
            # it looks like the app is stuck ("Updating..." forever).
            flags = [info["flag"]] if info.get("flag") and info["flag"] != "?" else None
            vessel_types = [info["vessel_type"]] if info.get("vessel_type") else None

        try:
            df = do_load_gaps(source, flags or None, vessel_types or None, vessel_ids,
                               start, end, api_key, min_hours)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, "GFW error: " + str(e)[:200]

        if df is None or df.empty:
            return [], f"No AIS gap found for this selection ({start} -> {end})."

        df = _classify(df, buffer_nm or 3)
        return df.to_dict("records"), f"{len(df)} gap event(s) loaded."

    def _filtered_and_summary(store, suspicious_only):
        """Shared by the table and map renderers: applies the 'suspicious
        only' toggle and computes the summary line, without re-querying
        GFW (pure client-side, on the already-fetched gap-store)."""
        full = pd.DataFrame(store)
        n_total = len(full)
        n_susp = int((full["status"] == "suspicious").sum()) if "status" in full.columns else 0
        n_norm = int((full["status"] == "normal").sum()) if "status" in full.columns else 0
        n_unk = n_total - n_susp - n_norm

        df = full
        if suspicious_only and "suspicious_only" in suspicious_only and "status" in full.columns:
            df = full[full["status"] == "suspicious"]

        summary = f"{n_total} gap(s) total -- {n_susp} suspicious, {n_norm} normal"
        if n_unk:
            summary += f", {n_unk} unclassified (no AIS buffer file found)"
        summary += "."
        return df, summary

    @app.callback(
        Output("gap-summary", "children"),
        Input("gap-store", "data"),
        Input("gap-filter-suspicious", "value"),
    )
    def _update_summary(store, suspicious_only):
        if not store:
            return ""
        _, summary = _filtered_and_summary(store, suspicious_only)
        return summary

    @app.callback(
        Output("gap-map-container", "children"),
        Input("gap-store", "data"),
        Input("gap-filter-suspicious", "value"),
        prevent_initial_call=True,
    )
    def _render_map(store, suspicious_only):
        if not store:
            return html.P("No gap loaded yet -- click \"Get AIS gaps\" first.",
                           style={"color": DIM, "fontSize": "0.8rem", "padding": "1rem"})
        df, _ = _filtered_and_summary(store, suspicious_only)
        return _build_gap_map(df)

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