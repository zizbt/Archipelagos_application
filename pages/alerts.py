"""
pages/alerts.py
================
Alert page -- computes suspicious-vessel alerts LIVE from the precomputed
trajectories over a chosen date range. No dependency on saved report
files (those can be deleted at any time, which made the old
watchlist/alerts/dossier trio built on top of them unreliable).

Signals combined per vessel:
  1. AIS gaps (>3h)   -- reconstructed CLIENT-SIDE from raw hourly
                          AIS-presence pings (gfw.bulk_load_client_gap_events_dataframe),
                          same method as pages/ais_gap.py, for every
                          vessel present in the trajectory selection.
                          Each gap over the threshold is classified via
                          the AIS coverage buffer (gfw.classify_gap_status,
                          shared with ais_gap.py): only "suspicious"
                          gaps (both endpoints genuinely offshore) count
                          for the gate/score -- "normal" near-coast gaps
                          are very likely reception artifacts, not real
                          AIS-off behaviour. This is the GATE: a vessel
                          with zero suspicious AIS gaps in the period
                          never appears in the alerts, no matter how much
                          loitering/encounters it has. AIS gap is also
                          the single biggest score contributor. Degrades
                          to "no alerts" (+ a warning) if no API key is
                          saved, since the gate itself now needs GFW.
                          HISTORY (why not simpler alternatives):
                          - A first version reconstructed gaps by diffing
                            consecutive rows of the precomputed trajectory
                            dataset. That was badly wrong: the trajectory
                            dataset only carries ~1 position/day/vessel,
                            so every day-to-day diff was naturally ~24h --
                            comfortably over the 3h threshold -- which
                            flagged an AIS "blackout" on every single day
                            for every vessel, real or not (giveaway:
                            totals landing right around 365*24=8760h).
                          - A second version switched to GFW's official
                            "public-global-gaps-events" dataset (same one
                            ais_gap.py used at the time). That dataset
                            only publishes gaps starting >= 50 nautical
                            miles from shore (GFW's own methodology) --
                            structurally excludes almost the whole Aegean
                            archipelago, so it returned 0 gaps for every
                            Greek vessel tested. Not a bug, just the wrong
                            dataset for a coastal fleet.
                          - This version reconstructs from raw pings
                            instead (no distance-from-shore cutoff), with
                            the buffer classification above to separate
                            real offshore silence from coastal noise.
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
import threading
import uuid
from datetime import date

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State, dash_table

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, lbl, FLAG_OPTIONS
from config import YEARS, FLAG_NAMES, VESSEL_TYPES
from loader import load_trajectories_range
from pages.loitering import get_loitering_dataframe
from pages.encounter import get_encounters_dataframe
from api_key import get_api_key
from gfw import (get_gfw_client, bulk_load_afe_dataframe,
                 bulk_load_client_gap_events_dataframe, load_ais_buffer_polygon,
                 classify_gap_status, GEAR_TYPES)
from port_zones import PORT_RADIUS_M_DEFAULT, has_ports
from zone_filters import get_zone_options, zone_mask, has_any_zone_data

TRAJECTORY_COLUMNS = ["lat", "lon", "vessel_id", "ship_name", "date", "flag", "vessel_type", "gear_type"]

GLOBAL_MIN_DATE = date(YEARS[0], 1, 1)
GLOBAL_MAX_DATE = date(YEARS[-1], 12, 31)

GAP_THRESHOLD_HOURS = 3  # any gap longer than this counts as an AIS blackout
DEFAULT_BUFFER_NM = 3    # AIS coverage buffer distance -- must match pages/ais_gap.py's

# ── Background-run progress tracking ────────────────────────────────────────
# Same fix/rationale as pages/ais_gap.py::_RUN_STATE: computing alerts calls
# _fetch_gap_signal, which downloads raw hourly AIS-presence pings per flag
# group (can be hundreds of vessels for a single flag) -- heavy and slow,
# previously fully synchronous inside the Dash callback with no feedback
# beyond console prints. Now run in a background thread, polled by a
# dcc.Interval, so "Analyze" shows real progress instead of a blind spinner.
_RUN_STATE = {}
_RUN_LOCK = threading.Lock()


def _set_run_state(run_id, **kwargs):
    with _RUN_LOCK:
        _RUN_STATE.setdefault(run_id, {}).update(kwargs)


def _get_run_state(run_id):
    with _RUN_LOCK:
        return dict(_RUN_STATE.get(run_id, {}))


def _load_trajectories(start, end, vessel_types, flags, gear_types):
    """
    Loads precomputed trajectories, filtered by vessel type / flag at load
    time (cheap -- handled by the loader itself). Gear type isn't
    necessarily present on every precomputed dataset, so it's applied as a
    post-filter here with a graceful fallback if the column is missing.
    Returns (df, warning_or_None).
    """
    try:
        df = load_trajectories_range(start, end, vessel_types or None, flags or None,
                                     columns=TRAJECTORY_COLUMNS)
        gear_available = True
    except Exception:
        fallback_cols = [c for c in TRAJECTORY_COLUMNS if c != "gear_type"]
        df = load_trajectories_range(start, end, vessel_types or None, flags or None,
                                     columns=fallback_cols)
        gear_available = False

    warning = None
    if gear_types:
        if gear_available and df is not None and not df.empty and "gear_type" in df.columns:
            df = df[df["gear_type"].isin(gear_types)]
        else:
            warning = "Gear type filter skipped: not available in this trajectory dataset."
    return df, warning


def _apply_zone_filter(df, zone_values):
    """
    Filtre les positions par zone maritime (EEZ, eaux territoriales,
    aires protegees WDPA/Natura2000, eaux internationales). Un point est
    garde s'il tombe dans AU MOINS UNE des zones cochees (union / OR),
    logique la plus utile pour "montre-moi les navires vus en zone
    protegee OU en eaux internationales", par exemple.
    Se degrade proprement (pas de filtrage) si aucune donnee de zone n'est
    chargee -- voir zone_filters.py.
    """
    if not zone_values or df is None or df.empty:
        return df, None
    if not has_any_zone_data():
        return df, "Zone filter skipped: no EEZ/WDPA reference data loaded."
    if "lat" not in df.columns or "lon" not in df.columns:
        return df, "Zone filter skipped: trajectory data has no lat/lon."

    combined_mask = None
    for zv in zone_values:
        m = zone_mask(df["lon"].values, df["lat"].values, zv)
        combined_mask = m if combined_mask is None else (combined_mask | m)

    return df[combined_mask], None

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
# AIS gaps -- reconstructed client-side from raw hourly AIS-presence pings
# (same method as pages/ais_gap.py), then classified suspicious/normal
# via the AIS coverage buffer. Runs for EVERY vessel in the trajectory
# selection, not just an already-gated subset, because AIS gaps are
# themselves the gate -- there's no cheaper signal to filter on first.
# Heavier than a plain events-API call (downloads raw position pings,
# month-by-month) -- expect this to be noticeably slower on a large
# vessel set or a long date range.
# ---------------------------------------------------------------------------
def _fetch_gap_stats(vessel_ids, start, end, gap_threshold=GAP_THRESHOLD_HOURS, flags_by_vessel=None,
                     progress_callback=None):
    """
    Renvoie (DataFrame indexe par vessel_id avec les colonnes
    ['ais_gaps', 'total_gap_hours', 'all_gaps', 'all_gap_hours'],
    warning_or_None).

    'ais_gaps'/'total_gap_hours' = SUSPICIOUS gaps only (both endpoints
    outside the AIS coverage buffer -- genuinely offshore); this is what
    gates/scores a vessel. 'all_gaps'/'all_gap_hours' = every gap over
    gap_threshold regardless of classification, kept only for display in
    the table (near-coast "normal" gaps are very likely reception
    artifacts, not real AIS-off behaviour, so they shouldn't drive an
    alert on their own).

    flags_by_vessel (optional dict {vessel_id: flag}): used to batch the
    download ONE CALL PER FLAG instead of one huge unfiltered call for
    every vessel in the selection. This matters a lot on a wide date
    range (e.g. a full year): the underlying AIS-presence report has no
    valid vessel_id server-side filter (see gfw.load_VP_data_by_vessel),
    so with no flag/vessel_type hint at all it has to download EVERY
    vessel's HOURLY pings for the WHOLE region/period before filtering
    down client-side -- exactly the kind of request GFW's own docs warn
    against ("prefer simple, small regions and shorter time ranges"),
    and in practice it fails/times out on a year-long, many-vessel
    selection, which silently gates out every vessel (0 alerts) instead
    of erroring loudly. Grouping by flag keeps each call's server-side
    filter narrow, the same way the old desktop app downloaded by flag
    and filtered by vessel_id afterward (VP_gfw.py / VP_map.py). Vessels
    with an unknown/missing flag fall back to one unfiltered call.

    A single flag group can still cover hundreds of vessels (e.g. every
    GRC-flagged vessel in the selection), so it's still a heavy
    month-by-month download even after this narrowing -- progress_callback
    (optional, (message, fraction) -> None) is forwarded down into
    bulk_load_client_gap_events_dataframe/bulk_load_vp_dataframe so the
    caller can surface real per-month progress instead of a blind wait.

    If no ais_buffer_<N>nm.geojson file is found on disk, suspicious/
    normal can't be told apart -- every gap over threshold counts instead
    (degrades to the pre-buffer behaviour), with a warning.

    Se degrade proprement (DataFrame vide + message) si pas de cle API ou
    si l'appel echoue -- dans ce cas la page n'affichera aucune alerte
    (le gap-gate ne peut plus etre evalue), mais elle ne plante pas.
    """
    empty = pd.DataFrame(columns=["ais_gaps", "total_gap_hours", "all_gaps", "all_gap_hours"])
    if not vessel_ids:
        return empty, None

    api_key = get_api_key()
    if not api_key:
        return empty, "AIS-gap signal skipped: no API key saved (Alerts now needs one, like the AIS Gaps page)."

    # Group vessel_ids by flag so each download can be narrowed server-side.
    # Vessels with no known/usable flag go in a single unfiltered group.
    flags_by_vessel = flags_by_vessel or {}
    groups = {}  # flag_or_None -> [vessel_id, ...]
    for vid in vessel_ids:
        flag = flags_by_vessel.get(vid)
        if not flag or pd.isna(flag) or str(flag).strip() in ("", "?"):
            flag = None
        groups.setdefault(flag, []).append(vid)

    event_frames = []
    errors = []
    n_groups = len(groups)

    def _run(flag, ids, group_index):
        def sub_progress(message, fraction):
            if progress_callback:
                label = flag or "unflagged"
                overall = (group_index + fraction) / max(n_groups, 1)
                progress_callback(f"AIS gaps [{label}, {len(ids)} vessel(s), "
                                   f"group {group_index + 1}/{n_groups}]: {message}", overall)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            client = get_gfw_client(api_key)
            df = loop.run_until_complete(
                bulk_load_client_gap_events_dataframe(
                    [flag] if flag else None, None, start, end, client,
                    vessel_ids=ids, min_hours=gap_threshold, progress_callback=sub_progress))
            if df is not None and not df.empty:
                event_frames.append(df)
        except Exception as e:
            errors.append(f"{flag or 'unflagged'}: {str(e)[:80]}")
        finally:
            loop.close()

    # Sequential on purpose -- GFW rejects concurrent reports on the same
    # token (see the 429 retry logic in gfw.py's report loaders).
    for i, (flag, ids) in enumerate(groups.items()):
        t = threading.Thread(target=_run, args=(flag, ids, i), daemon=True)
        t.start()
        t.join()

    if not event_frames:
        warning = ("AIS-gap signal failed for all flag group(s): " + "; ".join(errors[:3])) if errors else None
        return empty, warning

    events = pd.concat(event_frames, ignore_index=True)
    warning = ("AIS-gap signal failed for some flag group(s): " + "; ".join(errors[:3])) if errors else None

    if events.empty or "vessel_id" not in events.columns or "duration_hrs" not in events.columns:
        return empty, warning

    # Same safety net as _fetch_afe_hours: filter client-side to the
    # requested vessels rather than trust the server-side filter alone.
    events = events[events["vessel_id"].isin(vessel_ids)]
    events = events[events["duration_hrs"] >= gap_threshold]
    if events.empty:
        return empty, warning

    buffer_geom = load_ais_buffer_polygon(DEFAULT_BUFFER_NM)
    warning = None
    if buffer_geom is not None:
        events = events.copy()
        events["status"] = events.apply(
            lambda r: classify_gap_status(r.get("off_lat"), r.get("off_lon"),
                                          r.get("on_lat"), r.get("on_lon"), buffer_geom),
            axis=1)
        scored = events[events["status"] == "suspicious"]
    else:
        warning = (f"AIS-gap: no ais_buffer_{DEFAULT_BUFFER_NM}nm.geojson found -- "
                   "suspicious/normal split skipped, every gap over threshold counts.")
        scored = events

    all_stats = events.groupby("vessel_id").agg(
        all_gaps=("duration_hrs", "size"),
        all_gap_hours=("duration_hrs", "sum"),
    )
    if scored.empty:
        gap_stats = pd.DataFrame(columns=["ais_gaps", "total_gap_hours"])
    else:
        gap_stats = scored.groupby("vessel_id").agg(
            ais_gaps=("duration_hrs", "size"),
            total_gap_hours=("duration_hrs", "sum"),
        )
    gap_stats = gap_stats.join(all_stats, how="outer").fillna(0)
    return gap_stats, warning


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

    # SAFETY NET: the GFW 4Wings API does not reliably apply the
    # `vessel_id IN (...)` filter server-side -- it can silently return
    # every fishing vessel in the whole AFE region/period instead (confirmed
    # by testing: a single-vessel request returned 100k+ rows across many
    # flags). Without this client-side filter, a vessel's afe_hours would
    # be correct by luck (only if it happens to appear in the oversized
    # response) or simply come back as 0 whenever the response is dropped
    # for being too large / rate-limited. Filtering here makes the result
    # correct regardless of what the server actually filtered on.
    df = df[df["vessel_id"].isin(vessel_ids)]
    if df.empty:
        return {}, None

    hours_by_vessel = df.groupby("vessel_id")["hours"].sum().to_dict()
    return hours_by_vessel, None


# ---------------------------------------------------------------------------
# ALERT COMPUTATION
# ---------------------------------------------------------------------------
def compute_alerts(df, speed_threshold=1.5, loiter_duration=2.0,
                   dist_threshold=500, encounter_duration=2.0,
                   gap_threshold=GAP_THRESHOLD_HOURS,
                   afe_start=None, afe_end=None, fetch_afe=True,
                   exclude_port_events=True, port_radius_m=PORT_RADIUS_M_DEFAULT,
                   progress_callback=None):
    """
    Combines AIS gaps (gate + top signal), AFE hours, loitering, and
    encounters into one per-vessel suspicion table. Returns
    (DataFrame, warning_or_None) -- the warning covers both the live
    AIS-gap call and the live AFE call, joined with "; " if both fire.
    Only vessels with >=1 SUSPICIOUS AIS gap (offshore, both endpoints
    outside the AIS coverage buffer) over the period are considered --
    see _fetch_gap_stats.

    exclude_port_events: if True (default), loitering / encounter events
    that happen within port_radius_m of a known port point (see
    port_zones.py) don't contribute to the score -- a vessel idling or
    meeting another one at a harbour isn't suspicious the way it would be
    at sea. AIS gaps and AFE hours are unaffected by this flag.

    progress_callback (optional, (message, fraction) -> None): forwarded
    into _fetch_gap_stats (the heaviest, most variable-duration step by
    far -- see its docstring) and also called around the other stages so
    a caller polling from a background thread can show real progress
    instead of a blind spinner.
    """
    def progress(message, fraction):
        if progress_callback:
            progress_callback(message, fraction)

    empty_cols = ["vessel_id", "ship_name", "flag", "loitering_events", "loitering_hours",
                  "encounters", "ais_gaps", "total_gap_hours", "all_gaps", "all_gap_hours",
                  "afe_hours", "suspicion_score", "suspicion_level", "lat", "lon"]
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

    start_txt = afe_start or d["date"].min().strftime("%Y-%m-%d")
    end_txt = afe_end or d["date"].max().strftime("%Y-%m-%d")

    # --- Signal 1 (gate + top weight): AIS gaps (blackouts) --------------------
    # Live call to GFW's official gap-events dataset, for every vessel in
    # the current trajectory selection -- see _fetch_gap_stats docstring
    # and the module docstring for why this replaced the old local diff.
    all_vessel_ids = d["vessel_id"].dropna().unique().tolist()
    flags_by_vessel = flag_lookup.to_dict() if not flag_lookup.empty else {}
    progress(f"Fetching AIS gaps for {len(all_vessel_ids)} vessel(s)...", 0.0)

    def gap_progress(message, fraction):
        # Reserve the first 85% of the overall bar for the AIS-gap step --
        # by far the heaviest one -- and leave headroom for AFE/loitering/
        # encounters after it.
        progress(message, 0.85 * fraction)

    gap_stats, gap_warning = _fetch_gap_stats(all_vessel_ids, start_txt, end_txt, gap_threshold,
                                              flags_by_vessel=flags_by_vessel,
                                              progress_callback=gap_progress)

    # Only vessels with at least one real AIS gap are candidates --
    # everything else is skipped entirely.
    gated_ids = gap_stats[gap_stats["ais_gaps"] > 0].index if not gap_stats.empty else pd.Index([])
    if len(gated_ids) == 0:
        return pd.DataFrame(columns=empty_cols), gap_warning
    d_gated = d[d["vessel_id"].isin(gated_ids)]

    # --- Signal 2: AFE (fishing hours), live GFW call, gated vessels only ------
    afe_warning = None
    afe_hours_by_vessel = {}
    if fetch_afe:
        progress(f"Fetching AFE hours for {len(gated_ids)} gated vessel(s)...", 0.87)
        afe_hours_by_vessel, afe_warning = _fetch_afe_hours(list(gated_ids), start_txt, end_txt)

    # Combine any warnings from the two live GFW calls into one message.
    afe_warning = "; ".join(w for w in (gap_warning, afe_warning) if w) or None

    # --- Signal 3: loitering (gated vessels only) -------------------------------
    progress("Computing loitering events...", 0.92)
    loi = get_loitering_dataframe(d_gated, speed_threshold_knots=speed_threshold,
                                  min_duration_hours=loiter_duration,
                                  port_radius_m=port_radius_m)
    if exclude_port_events and not loi.empty:
        loi = loi[~loi["in_port"]]
    if not loi.empty:
        loiter_stats = loi.groupby("vessel_id").agg(
            loitering_events=("vessel_id", "size"),
            loitering_hours=("duration_hours", "sum"),
        )
    else:
        loiter_stats = pd.DataFrame(columns=["loitering_events", "loitering_hours"])

    # --- Signal 4: encounters (gated vessels only) -------------------------------
    progress("Computing encounters...", 0.96)
    enc = get_encounters_dataframe(d_gated, dist_threshold_meters=dist_threshold,
                                   time_threshold_hours=encounter_duration,
                                   port_radius_m=port_radius_m)
    if exclude_port_events and not enc.empty:
        enc = enc[~enc["in_port"]]
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
        all_gaps = int(gap_stats["all_gaps"].get(vid, 0)) if "all_gaps" in gap_stats.columns else ais_gaps
        all_gap_hours = float(gap_stats["all_gap_hours"].get(vid, 0)) if "all_gap_hours" in gap_stats.columns else total_gap_hours
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
            "all_gaps": all_gaps,
            "all_gap_hours": round(all_gap_hours, 2),
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
        dcc.Store(id="alerts-run-id", data=None),
        dcc.Interval(id="alerts-progress-interval", interval=1500, disabled=True),
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

            lbl("Vessel type"),
            dcc.Dropdown(id="alerts-vessel-type-filter",
                options=[{"label": t.capitalize(), "value": t} for t in VESSEL_TYPES],
                value=[], multi=True, placeholder="All types...",
                style={"color": "#000", "marginBottom": "1rem"}),

            lbl("Gear type"),
            dcc.Dropdown(id="alerts-gear-type-filter",
                options=[{"label": g.replace("_", " ").title(), "value": g} for g in GEAR_TYPES],
                value=[], multi=True, placeholder="All gears...",
                style={"color": "#000", "marginBottom": "1rem"}),

            lbl("Water zone"),
            dcc.Dropdown(id="alerts-zone-filter",
                options=get_zone_options(), value=[], multi=True,
                placeholder="All waters..." if get_zone_options()
                            else "No zone reference data loaded",
                disabled=not get_zone_options(),
                style={"color": "#000", "marginBottom": "0.3rem"}),
            html.P("Keep a vessel's positions only if at least one falls in one "
                   "of the selected zones (EEZ, territorial waters, protected "
                   "areas, or international waters).",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "1rem"}),

            lbl("Port filter"),
            html.P("No port reference file found (data/gis/ITA_vessels.geojson) "
                   "-- filter has no effect.",
                   style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                          "marginBottom": "0.4rem", "display": "block" if not has_ports() else "none"}),
            dcc.Checklist(id="alerts-exclude-port",
                options=[{"label": " Exclude in-port loitering/encounters", "value": "exclude"}],
                value=["exclude"],
                labelStyle={"fontSize": "0.75rem", "color": SOFT, "cursor": "pointer"},
                style={"marginBottom": "0.4rem"}),
            html.Div([
                lbl("Port radius (m)"),
                dcc.Slider(id="alerts-port-radius", min=200, max=3000, step=100,
                    value=PORT_RADIUS_M_DEFAULT,
                    marks={200: "200", 1500: "1500", 3000: "3000"},
                    tooltip={"placement": "bottom", "always_visible": False}),
            ], style={"marginBottom": "1rem"}),

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
            dcc.Loading(children=html.Div(id="alerts-table")),
        ], style={"flex": "1", "minWidth": "0", "minHeight": 0, "overflowY": "auto",
                   "padding": "1rem", "background": BG}),

    ], style={"display": "flex", "alignItems": "flex-start", "height": "calc(100vh - 52px)"})


# ---------------------------------------------------------------------------
# HELPERS: table
# ---------------------------------------------------------------------------
def _table(alerts_df):
    if alerts_df is None or alerts_df.empty:
        return html.P("No alerts for this period.", style={"color": SOFT, "fontSize": "0.8rem"})
    show = alerts_df.copy()
    show["Flag"] = show["flag"].apply(lambda f: FLAG_NAMES.get(f, f))
    show = show.rename(columns={
        "ship_name": "Ship Name", "loitering_events": "Loitering Events",
        "loitering_hours": "Loitering Hours", "encounters": "Encounters",
        "ais_gaps": "Suspicious Gaps (offshore)", "total_gap_hours": "Suspicious Gap Hours",
        "all_gaps": "AIS Blackouts (>3h, all)", "all_gap_hours": "Total Gap Hours (all)",
        "afe_hours": "AFE Hours", "suspicion_score": "Suspicion Score",
        "suspicion_level": "Suspicion Level",
    })
    cols = ["Ship Name", "Flag", "Suspicion Level", "Suspicion Score",
            "Suspicious Gaps (offshore)", "Suspicious Gap Hours",
            "AIS Blackouts (>3h, all)", "Total Gap Hours (all)", "AFE Hours",
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

    def _background_alerts_run(run_id, start, end, flag_filter, vessel_type_filter,
                                gear_type_filter, zone_filter, exclude_port, port_radius):
        """
        Runs the actual (potentially long) trajectory load + alert
        computation off the Dash request thread, writing progress and the
        final result into _RUN_STATE[run_id] as it goes. Mirrors exactly
        what the old synchronous _run callback did for the button-click
        path -- only the execution/reporting model changed.
        """
        try:
            def progress(message, fraction):
                _set_run_state(run_id, status="running", message=message, fraction=fraction)

            progress("Loading trajectories...", 0.0)
            df, gear_warning = _load_trajectories(start, end, vessel_type_filter,
                                                  flag_filter, gear_type_filter)
            if df is None or df.empty:
                msg = "No trajectory data for this range/filter."
                if gear_warning:
                    msg += f" ({gear_warning})"
                _set_run_state(run_id, status="done", alerts=None, summary=msg)
                return

            progress("Applying zone filter...", 0.0)
            df, zone_warning = _apply_zone_filter(df, zone_filter)
            if df is None or df.empty:
                msg = "No trajectory data left after zone filter."
                if zone_warning:
                    msg += f" ({zone_warning})"
                _set_run_state(run_id, status="done", alerts=None, summary=msg)
                return

            alerts, afe_warning = compute_alerts(
                df, afe_start=start[:10], afe_end=end[:10],
                exclude_port_events=bool(exclude_port),
                port_radius_m=port_radius or PORT_RADIUS_M_DEFAULT,
                progress_callback=progress)

            if alerts.empty:
                summary = "No vessel with AIS gaps found for this period/filter."
            else:
                port_txt = " (in-port loitering/encounters excluded)" if exclude_port else ""
                summary = f"{len(alerts)} vessel(s) flagged ({start} -> {end}){port_txt}."
            if afe_warning:
                summary += f" ({afe_warning})"
            if gear_warning:
                summary += f" ({gear_warning})"
            if zone_warning:
                summary += f" ({zone_warning})"
            _set_run_state(run_id, status="done", alerts=alerts, summary=summary)

        except Exception as e:
            _set_run_state(run_id, status="error", message="Error: " + str(e)[:150])

    @app.callback(
        Output("alerts-table", "children"),
        Output("alerts-status", "children"),
        Output("alerts-store", "data"),
        Input("alerts-level-filter", "value"),
        Input("alerts-vessel-search", "value"),
        State("alerts-store", "data"),
        prevent_initial_call=True,
    )
    def _refilter(level_filter, search_text, cached):
        # Filtering by level/vessel-name alone shouldn't re-run the whole
        # detection -- only re-filter the cached result.
        if cached is None:
            raise dash.exceptions.PreventUpdate
        df = pd.DataFrame(cached)
        df = _apply_filters(df, level_filter, search_text)
        status = f"{len(df)} vessel(s) shown (filtered)."
        return _table(df), status, dash.no_update

    @app.callback(
        Output("alerts-run-id", "data"),
        Output("alerts-progress-interval", "disabled"),
        Output("alerts-status", "children", allow_duplicate=True),
        Output("alerts-btn-run", "disabled"),
        Input("alerts-btn-run", "n_clicks"),
        State("alerts-start", "date"),
        State("alerts-end", "date"),
        State("alerts-flag-filter", "value"),
        State("alerts-vessel-type-filter", "value"),
        State("alerts-gear-type-filter", "value"),
        State("alerts-zone-filter", "value"),
        State("alerts-exclude-port", "value"),
        State("alerts-port-radius", "value"),
        prevent_initial_call=True,
    )
    def _launch_run(n, start, end, flag_filter, vessel_type_filter, gear_type_filter,
                    zone_filter, exclude_port, port_radius):
        if not n:
            raise dash.exceptions.PreventUpdate

        run_id = str(uuid.uuid4())
        _set_run_state(run_id, status="running", message="Starting...", fraction=0.0)
        t = threading.Thread(
            target=_background_alerts_run,
            args=(run_id, start, end, flag_filter, vessel_type_filter, gear_type_filter,
                  zone_filter, exclude_port, port_radius),
            daemon=True,
        )
        t.start()
        return run_id, False, "Starting...", True

    @app.callback(
        Output("alerts-table", "children", allow_duplicate=True),
        Output("alerts-status", "children", allow_duplicate=True),
        Output("alerts-store", "data", allow_duplicate=True),
        Output("alerts-progress-interval", "disabled", allow_duplicate=True),
        Output("alerts-btn-run", "disabled", allow_duplicate=True),
        Input("alerts-progress-interval", "n_intervals"),
        State("alerts-run-id", "data"),
        State("alerts-level-filter", "value"),
        State("alerts-vessel-search", "value"),
        prevent_initial_call=True,
    )
    def _poll_run(n_intervals, run_id, level_filter, search_text):
        if not run_id:
            raise dash.exceptions.PreventUpdate
        state = _get_run_state(run_id)
        if not state:
            raise dash.exceptions.PreventUpdate

        status = state.get("status")
        if status == "running":
            fraction = state.get("fraction") or 0.0
            message = state.get("message") or "Working..."
            status_text = f"{message} ({fraction * 100:.0f}%)"
            return dash.no_update, status_text, dash.no_update, False, True

        # status in ("done", "error") -- stop polling and release the button
        with _RUN_LOCK:
            _RUN_STATE.pop(run_id, None)

        if status == "error":
            return _table(None), state.get("message", "Error."), None, True, False

        alerts = state.get("alerts")
        summary = state.get("summary", "")
        store = alerts.to_dict("records") if alerts is not None and not alerts.empty else None
        shown = _apply_filters(alerts, level_filter, search_text) if alerts is not None else alerts
        return _table(shown), summary, store, True, False

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