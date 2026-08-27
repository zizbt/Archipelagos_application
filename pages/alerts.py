"""
pages/alerts.py
================
Alert page -- computes suspicious-vessel alerts LIVE from the precomputed
trajectories over a chosen date range. No dependency on saved report
files (those can be deleted at any time, which made the old
watchlist/alerts/dossier trio built on top of them unreliable).

Signals combined per vessel:
  1. AIS gaps (>3h)   -- same time-gap logic as the VP report. This is now
                          the GATE: a vessel with zero AIS gaps in the
                          period never appears in the alerts, no matter
                          how much loitering/encounters it has. AIS gap
                          is also the single biggest score contributor.
  2. AFE (fishing hrs) -- Apparent Fishing Effort, fetched live from GFW
                          for the (small) set of vessels that passed the
                          AIS-gap gate. Second-biggest score contributor.
                          Degrades gracefully to 0 if no API key is saved
                          -- the rest of the page keeps working.
  3. Loitering         -- pages/loitering.get_loitering_dataframe
  4. Encounters         -- pages/encounter.get_encounters_dataframe

Each vessel gets a Suspicion Score (0-100) and a Suspicion Level:
  Suspicious        (score >= 15)
  Very Suspicious    (score >= 40)
  Critical           (score >= 70)
Vessels with zero AIS gaps, or scoring below MIN_SCORE_SHOWN, are not
shown -- this page is meant to surface what needs attention, not list
every vessel.
"""

import asyncio
import json
import threading
from datetime import date

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl, AEGEAN_CENTER, FLAG_OPTIONS
from config import YEARS, FLAG_NAMES
from loader import load_trajectories_range
from pages.loitering import get_loitering_dataframe
from pages.encounter import get_encounters_dataframe
from api_key import get_api_key
from gfw import get_gfw_client, bulk_load_afe_dataframe

TRAJECTORY_COLUMNS = ["lat", "lon", "vessel_id", "ship_name", "date", "flag"]

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

GAP_THRESHOLD_HOURS = 3  # any gap longer than this counts as an AIS blackout

# Scoring weights -- tune here if the mix needs adjusting.
# AIS gap is the dominant signal (and the gate: 0 gaps -> vessel excluded).
GAP_POINTS_PER_EVENT = 20
GAP_POINTS_CAP = 50

# AFE (fishing hours) is the second biggest signal.
AFE_POINTS_PER_HOUR = 1.5
AFE_POINTS_CAP = 35

# Loitering / encounters are secondary signals.
LOITER_POINTS_PER_EVENT = 5
LOITER_POINTS_CAP = 20
ENCOUNTER_POINTS_PER_EVENT = 8
ENCOUNTER_POINTS_CAP = 20

LEVEL_THRESHOLDS = [
    (70, "Critical", [220, 20, 60, 220]),      # crimson
    (40, "Very Suspicious", [255, 140, 0, 210]),  # orange
    (15, "Suspicious", [255, 215, 0, 190]),     # gold
]
MIN_SCORE_SHOWN = 15


def _suspicion_level(score):
    for threshold, label, _color in LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Normal"


def _level_color(level):
    for _threshold, label, color in LEVEL_THRESHOLDS:
        if label == level:
            return color
    return [150, 150, 150, 150]


# ---------------------------------------------------------------------------
# AFE (fishing hours) -- live GFW call, only for the vessels that already
# passed the AIS-gap gate (small set), so this stays fast.
# ---------------------------------------------------------------------------
def _fetch_afe_hours(vessel_ids, start, end):
    """
    Renvoie (dict {vessel_id: total_hours}, warning_or_None).
    Se degrade proprement (dict vide + message) si pas de cle API ou si
    l'appel echoue -- le reste de la page continue de fonctionner.
    """
    if not vessel_ids:
        return {}, None

    api_key = get_api_key()
    if not api_key:
        return {}, "AFE signal skipped: no API key saved."

    result = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            client = get_gfw_client(api_key)
            df = loop.run_until_complete(
                bulk_load_afe_dataframe(None, start, end, client, vessel_ids=vessel_ids))
            result["df"] = df
        except Exception as e:
            result["error"] = str(e)
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join()

    if result.get("error"):
        return {}, f"AFE signal skipped: {result['error'][:100]}"

    df = result.get("df")
    if df is None or df.empty or "vessel_id" not in df.columns or "hours" not in df.columns:
        return {}, None

    hours_by_vessel = df.groupby("vessel_id")["hours"].sum().to_dict()
    return hours_by_vessel, None


# ---------------------------------------------------------------------------
# ALERT COMPUTATION
# ---------------------------------------------------------------------------
def compute_alerts(df, speed_threshold=1.5, loiter_duration=2.0,
                   dist_threshold=500, encounter_duration=2.0,
                   gap_threshold=GAP_THRESHOLD_HOURS,
                   afe_start=None, afe_end=None, fetch_afe=True):
    """
    Combines AIS gaps (gate + top signal), AFE hours, loitering, and
    encounters into one per-vessel suspicion table. Returns
    (DataFrame, afe_warning_or_None). Only vessels with >=1 AIS gap over
    the period are considered at all.
    """
    empty_cols = ["vessel_id", "ship_name", "flag", "loitering_events", "loitering_hours",
                  "encounters", "ais_gaps", "total_gap_hours", "afe_hours",
                  "suspicion_score", "suspicion_level", "lat", "lon"]
    if df is None or df.empty or "vessel_id" not in df.columns:
        return pd.DataFrame(columns=empty_cols), None

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["lat", "lon", "date", "vessel_id"]).sort_values(["vessel_id", "date"])

    # Vessel name / flag lookup + a rough position (mean lat/lon) to place
    # each vessel on the summary map.
    name_lookup = d.groupby("vessel_id")["ship_name"].first()
    flag_lookup = d.groupby("vessel_id")["flag"].first() if "flag" in d.columns else pd.Series(dtype=object)
    position_lookup = d.groupby("vessel_id")[["lat", "lon"]].mean()

    # --- Signal 1 (gate + top weight): AIS gaps (blackouts) --------------------
    d["gap_hours"] = d.groupby("vessel_id")["date"].diff().dt.total_seconds() / 3600
    d["gap_hours"] = d["gap_hours"].fillna(0)
    d["is_gap"] = d["gap_hours"] > gap_threshold
    gap_stats = d.groupby("vessel_id").agg(
        ais_gaps=("is_gap", "sum"),
        total_gap_hours=("gap_hours", lambda x: x[x > gap_threshold].sum()),
    )

    # Only vessels with at least one AIS gap are candidates -- everything
    # else is skipped entirely (gap detection is by far the cheapest
    # signal to compute, so filtering on it first saves real work below).
    gated_ids = gap_stats[gap_stats["ais_gaps"] > 0].index
    if len(gated_ids) == 0:
        return pd.DataFrame(columns=empty_cols), None
    d_gated = d[d["vessel_id"].isin(gated_ids)]

    # --- Signal 2: AFE (fishing hours), live GFW call, gated vessels only ------
    afe_warning = None
    afe_hours_by_vessel = {}
    if fetch_afe:
        start_txt = afe_start or d["date"].min().strftime("%Y-%m-%d")
        end_txt = afe_end or d["date"].max().strftime("%Y-%m-%d")
        afe_hours_by_vessel, afe_warning = _fetch_afe_hours(list(gated_ids), start_txt, end_txt)

    # --- Signal 3: loitering (gated vessels only) -------------------------------
    loi = get_loitering_dataframe(d_gated, speed_threshold_knots=speed_threshold,
                                  min_duration_hours=loiter_duration)
    if not loi.empty:
        loiter_stats = loi.groupby("vessel_id").agg(
            loitering_events=("vessel_id", "size"),
            loitering_hours=("duration_hours", "sum"),
        )
    else:
        loiter_stats = pd.DataFrame(columns=["loitering_events", "loitering_hours"])

    # --- Signal 4: encounters (gated vessels only) -------------------------------
    enc = get_encounters_dataframe(d_gated, dist_threshold_meters=dist_threshold,
                                   time_threshold_hours=encounter_duration)
    encounter_counts = {}
    if not enc.empty:
        for col in ["vessel_1_id", "vessel_2_id"]:
            for vid in enc[col]:
                encounter_counts[vid] = encounter_counts.get(vid, 0) + 1

    # --- Combine ----------------------------------------------------------------
    rows = []
    for vid in gated_ids:
        loitering_events = int(loiter_stats["loitering_events"].get(vid, 0)) if not loiter_stats.empty else 0
        loitering_hours = float(loiter_stats["loitering_hours"].get(vid, 0)) if not loiter_stats.empty else 0.0
        encounters = int(encounter_counts.get(vid, 0))
        ais_gaps = int(gap_stats["ais_gaps"].get(vid, 0))
        total_gap_hours = float(gap_stats["total_gap_hours"].get(vid, 0))
        afe_hours = float(afe_hours_by_vessel.get(vid, 0))

        score = 0
        score += min(ais_gaps * GAP_POINTS_PER_EVENT, GAP_POINTS_CAP)
        score += min(afe_hours * AFE_POINTS_PER_HOUR, AFE_POINTS_CAP)
        score += min(loitering_events * LOITER_POINTS_PER_EVENT, LOITER_POINTS_CAP)
        score += min(encounters * ENCOUNTER_POINTS_PER_EVENT, ENCOUNTER_POINTS_CAP)
        score = min(round(score), 100)

        if score < MIN_SCORE_SHOWN:
            continue

        pos = position_lookup.loc[vid]
        rows.append({
            "vessel_id": vid,
            "ship_name": name_lookup.get(vid, str(vid)),
            "flag": flag_lookup.get(vid, "?") if not flag_lookup.empty else "?",
            "loitering_events": loitering_events,
            "loitering_hours": round(loitering_hours, 2),
            "encounters": encounters,
            "ais_gaps": ais_gaps,
            "total_gap_hours": round(total_gap_hours, 2),
            "afe_hours": round(afe_hours, 2),
            "suspicion_score": score,
            "suspicion_level": _suspicion_level(score),
            "lat": round(float(pos["lat"]), 5),
            "lon": round(float(pos["lon"]), 5),
        })

    if not rows:
        return pd.DataFrame(columns=empty_cols), afe_warning

    result = pd.DataFrame(rows).sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    return result, afe_warning


# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
def layout():
    return html.Div([
        dcc.Store(id="alerts-store", data=None),
        dcc.Download(id="alerts-download-csv"),

        html.Div([
            html.H6("Vessel alerts", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
            html.P("AIS gap is the primary signal: vessels with zero gaps are not shown. "
                   "AFE (fishing hours), loitering, and encounters refine the score.",
                   style={"fontSize": "0.7rem", "color": DIM, "marginBottom": "1rem"}),

            lbl("Jump to a year (optional)"),
            dcc.Dropdown(id="alerts-year", value=None, clearable=True,
                options=[{"label": str(y), "value": y} for y in YEARS],
                placeholder="Jump to a year...",
                style={"color": "#000", "marginBottom": "0.6rem"}),
            lbl("Start date"),
            dcc.DatePickerSingle(id="alerts-start", date=date(YEARS[-1], 1, 1),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "0.6rem"}),
            lbl("End date"),
            dcc.DatePickerSingle(id="alerts-end", date=date(YEARS[-1], 1, 31),
                display_format="YYYY-MM-DD",
                min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE,
                style={"marginBottom": "1rem"}),

            lbl("Flag (country)"),
            dcc.Dropdown(id="alerts-flag-filter", options=FLAG_OPTIONS, value=[], multi=True,
                placeholder="All flags...",
                style={"color": "#000", "marginBottom": "1rem"}),

            lbl("Suspicion level"),
            dcc.Dropdown(id="alerts-level-filter",
                options=[{"label": lvl, "value": lvl} for _, lvl, _ in LEVEL_THRESHOLDS],
                value=[lvl for _, lvl, _ in LEVEL_THRESHOLDS], multi=True,
                style={"color": "#000", "marginBottom": "1rem"}),

            lbl("Search vessel (name)"),
            dcc.Input(id="alerts-vessel-search", type="text", debounce=True,
                placeholder="Filter by ship name...",
                style={"width": "100%", "padding": "0.4rem", "marginBottom": "1rem",
                       "borderRadius": "5px", "border": f"1px solid {BDR}",
                       "background": PANEL, "color": MAIN}),

            html.P("Tip: keep the range short (days/weeks). This runs AIS-gap + AFE + "
                   "loitering + encounter detection together, which is heavy.",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "0.6rem"}),

            html.Button("Analyze", id="alerts-btn-run", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none",
                       "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
                       "marginBottom": "0.6rem"}),
            html.Button("Export CSV", id="alerts-btn-export", n_clicks=0,
                style={"width": "100%", "padding": "0.5rem",
                       "background": PANEL, "color": SOFT,
                       "border": f"1px solid {BDR}", "borderRadius": "6px",
                       "cursor": "pointer", "marginBottom": "1rem"}),

            html.Div(id="alerts-status", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "300px", "minWidth": "300px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "flexShrink": "0", "position": "sticky", "top": "0",
                   "alignSelf": "flex-start", "maxHeight": "100vh", "overflowY": "auto"}),

        html.Div([
            html.Div([
                dcc.Loading(type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="alerts-map-container", style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0}),

            html.Div(
                dcc.Loading(children=html.Div(id="alerts-table")),
                style={"height": "300px", "flexShrink": "0", "overflowY": "auto", "minWidth": "0",
                       "borderTop": f"1px solid {BDR}", "padding": "0.5rem 1rem", "background": BG},
            ),
        ], style={"flex": "1", "minWidth": "0", "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "alignItems": "flex-start"})


# ---------------------------------------------------------------------------
# HELPERS: map + table
# ---------------------------------------------------------------------------
def _build_map(alerts_df):
    layers = []
    if alerts_df is not None and not alerts_df.empty:
        plot = alerts_df.copy()
        plot["tooltip"] = (plot["ship_name"].astype(str) + " [" + plot["flag"].astype(str) + "] -- "
                           + plot["suspicion_level"].astype(str)
                           + " (score " + plot["suspicion_score"].astype(str) + ")")
        plot["color"] = plot["suspicion_level"].apply(_level_color)
        plot["radius"] = plot["suspicion_score"].clip(lower=10) * 60

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius", radius_min_pixels=5, radius_max_pixels=45,
            pickable=True, auto_highlight=True, opacity=0.7,
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=AEGEAN_CENTER["lat"], longitude=AEGEAN_CENTER["lon"],
            zoom=6, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    )
    deck_json = json.loads(deck.to_json())
    return dash_deck.DeckGL(data=deck_json, mapboxKey=MAPBOX_KEY,
                            style={"width": "100%", "height": "100%"})


def _table(alerts_df):
    if alerts_df is None or alerts_df.empty:
        return html.P("No alerts for this period.", style={"color": SOFT, "fontSize": "0.8rem"})
    show = alerts_df.copy()
    show["Flag"] = show["flag"].apply(lambda f: FLAG_NAMES.get(f, f))
    show = show.rename(columns={
        "ship_name": "Ship Name", "loitering_events": "Loitering Events",
        "loitering_hours": "Loitering Hours", "encounters": "Encounters",
        "ais_gaps": "AIS Blackouts (>3h)", "total_gap_hours": "Total Gap Hours",
        "afe_hours": "AFE Hours", "suspicion_score": "Suspicion Score",
        "suspicion_level": "Suspicion Level",
    })
    cols = ["Ship Name", "Flag", "Suspicion Level", "Suspicion Score",
            "AIS Blackouts (>3h)", "Total Gap Hours", "AFE Hours",
            "Loitering Events", "Loitering Hours", "Encounters"]
    return dash_table.DataTable(
        data=show[cols].to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols],
        sort_action="native", filter_action="native", page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": BG, "color": SOFT, "border": f"1px solid {BDR}",
                    "fontSize": "0.72rem", "padding": "4px 8px"},
        style_data_conditional=[
            {"if": {"filter_query": '{Suspicion Level} = "Critical"', "column_id": "Suspicion Level"},
             "backgroundColor": "rgba(220,20,60,0.18)", "color": "#e07070", "fontWeight": "800"},
            {"if": {"filter_query": '{Suspicion Level} = "Very Suspicious"', "column_id": "Suspicion Level"},
             "backgroundColor": "rgba(255,140,0,0.18)", "color": "#e0b070", "fontWeight": "800"},
            {"if": {"filter_query": '{Suspicion Level} = "Suspicious"', "column_id": "Suspicion Level"},
             "backgroundColor": "rgba(255,215,0,0.14)", "color": "#e0d070", "fontWeight": "800"},
        ],
        style_header={"backgroundColor": PANEL, "color": MAIN, "fontWeight": "600"},
    )


def _apply_filters(df, level_filter, search_text):
    """Filtre le DataFrame en cache -- pas de recalcul (niveau + texte)."""
    if df is None or df.empty:
        return df
    if level_filter:
        df = df[df["suspicion_level"].isin(level_filter)]
    if search_text:
        needle = str(search_text).strip().lower()
        if needle:
            df = df[df["ship_name"].astype(str).str.lower().str.contains(needle, na=False)]
    return df


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------
def register_callbacks(app):

    @app.callback(
        Output("alerts-start", "date"),
        Output("alerts-end", "date"),
        Input("alerts-year", "value"),
        prevent_initial_call=True,
    )
    def _jump_year(year):
        if not year:
            raise dash.exceptions.PreventUpdate
        return date(year, 1, 1), date(year, 1, 31)

    @app.callback(
        Output("alerts-map-container", "children"),
        Output("alerts-table", "children"),
        Output("alerts-status", "children"),
        Output("alerts-store", "data"),
        Input("alerts-btn-run", "n_clicks"),
        Input("alerts-level-filter", "value"),
        Input("alerts-vessel-search", "value"),
        State("alerts-start", "date"),
        State("alerts-end", "date"),
        State("alerts-flag-filter", "value"),
        State("alerts-store", "data"),
        prevent_initial_call=True,
    )
    def _run(n, level_filter, search_text, start, end, flag_filter, cached):
        trigger = dash.ctx.triggered_id

        # Filtering by level/vessel-name alone shouldn't re-run the whole
        # detection -- only re-filter the cached result.
        if trigger in ("alerts-level-filter", "alerts-vessel-search") and cached is not None:
            df = pd.DataFrame(cached)
            df = _apply_filters(df, level_filter, search_text)
            status = f"{len(df)} vessel(s) shown (filtered)."
            return _build_map(df), _table(df), status, dash.no_update

        if not n:
            raise dash.exceptions.PreventUpdate

        df = load_trajectories_range(start, end, None, flag_filter or None, columns=TRAJECTORY_COLUMNS)
        if df is None or df.empty:
            return _build_map(None), _table(None), "No trajectory data for this range/filter.", None

        alerts, afe_warning = compute_alerts(df, afe_start=start[:10], afe_end=end[:10])
        store = alerts.to_dict("records") if not alerts.empty else None

        shown = _apply_filters(alerts, level_filter, search_text)
        if alerts.empty:
            summary = "No vessel with AIS gaps found for this period/filter."
        else:
            summary = f"{len(alerts)} vessel(s) flagged ({start} -> {end})."
        if afe_warning:
            summary += f" ({afe_warning})"
        return _build_map(shown), _table(shown), summary, store

    @app.callback(
        Output("alerts-download-csv", "data"),
        Input("alerts-btn-export", "n_clicks"),
        State("alerts-store", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        out = pd.DataFrame(store)
        return dcc.send_data_frame(out.to_csv, "vessel_alerts.csv", index=False)