"""
gfw.py
======
Global Fishing Watch API client.
Restauration de la version qui fonctionnait (celle basee sur VP_gfw.py
du collegue), simplement sans le parametre gear_type.
"""

import asyncio
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon
from shapely.geometry import mapping


# ── Region — Aegean Sea ─────────────────────────────────────────────────────────

AEGEAN_LAT_LONS = [
    (19.5, 34.0),
    (30.5, 34.0),
    (30.5, 41.5),
    (19.5, 41.5),
    (19.5, 34.0),
]
AEGEAN_POLYGON = Polygon(AEGEAN_LAT_LONS)
AEGEAN_GEOJSON = mapping(AEGEAN_POLYGON)

_AFE_REGION_POLY = Polygon([
    (11.0, 33.0), (30.5, 33.0), (30.5, 36.5),
    (26.5, 41.5), (22.0, 41.5), (11.0, 36.5), (11.0, 33.0),
])

# Region actually used by the old app (VP_gfw.py) for VP / gap-related
# queries -- wider than AEGEAN_GEOJSON (lon 11.0->30.5, lat 33.0->41.5:
# includes the Ionian Sea / Malta and south of Crete, not just the
# Aegean box). Restored here so vessel-presence and AIS-gap queries
# cover the same area the old app did.
AFE_REGION_GEOJSON = mapping(_AFE_REGION_POLY)

# Options for the UI 

GFW_VESSEL_TYPES = [
    "fishing",
    "carrier",
    "bunker",
    "cargo",
    "passenger",
    "other",
    "seismic_vessel",
    "gear",
]

COUNTRY_FLAGS = [
    "GRC", "TUR", "ITA", "MLT", "TUN", "CYP", "DZA", "ALB", "FRA",
    "ESP", "HRV", "MNE", "LBY", "EGY", "LBN", "SYR", "RUS", "UKR",
    "ROU", "BGR", "GEO", "ISR", "LBR", "PAN", "BHS", "POL",
]

# NOTE: gear_type n'est PAS filtrable au téléchargement (pas supporté par
# l'endpoint de présence AIS). En revanche, l'API renvoie quand même une
# colonne "gear_type" dans les résultats -> on peut donc filtrer dessus
# APRÈS téléchargement / import, côté affichage (page Map). Cette liste
# sert uniquement d'options pour ce filtre d'affichage.
GEAR_TYPES = [
    "TRAWLERS",
    "PURSE_SEINES",
    "TUNA_PURSE_SEINES",
    "OTHER_PURSE_SEINES",
    "DRIFTING_LONGLINES",
    "SET_LONGLINES",
    "SET_GILLNETS",
    "POLE_AND_LINE",
    "FIXED_GEAR",
    "SEINERS",
    "SQUID_JIGGER",
    "POTS_AND_TRAPS",
    "FISHING",
    "INCONCLUSIVE",
]


def get_gfw_client(api_key):
    import gfwapiclient as gfw
    gfw_client = gfw.Client(access_token=api_key)
    return gfw_client


def load_ais_buffer_polygon(buffer_nm=3):
    """
    Loads data/gis/ais_buffer_<N>nm.geojson (if present) and returns a
    single shapely geometry (union of all its features) usable for
    point-in-polygon tests, e.g. to classify whether an AIS gap happened
    close to shore/coverage limits ("normal") or genuinely offshore
    ("suspicious"). Returns None if the file isn't there -- callers should
    degrade gracefully (skip the suspicious/normal split) rather than
    fail.

    Kept dependency-light on purpose: plain shapely + json, no geopandas/
    CRS reprojection, since the buffer file is already precomputed in
    WGS84 (lon/lat) -- matching the coordinates GFW returns.
    """
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union

    candidates = [
        Path(__file__).parent / "data" / "gis" / f"ais_buffer_{buffer_nm}nm.geojson",
        Path(__file__).parent / "data" / "gis" / f"ais_buffer_{float(buffer_nm)}nm.geojson",
        Path(__file__).parent / "map_files" / f"ais_buffer_{buffer_nm}nm.geojson",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [data]) if isinstance(data, dict) and "features" in data else [data]
    geoms = [shape(feat["geometry"] if "geometry" in feat else feat) for feat in features]
    return unary_union(geoms)


def classify_gap_status(off_lat, off_lon, on_lat, on_lon, buffer_geom):
    """
    Classifies a single AIS gap given its off/on positions:
      - "suspicious": BOTH endpoints fall OUTSIDE the AIS coverage buffer
        (data/gis/ais_buffer_<N>nm.geojson) -- genuinely offshore.
      - "normal": at least one endpoint is INSIDE the buffer -- e.g. near
        a coast/port/AIS-coverage edge, most likely a reception artifact
        rather than real intentional AIS-disabling.
      - "gap": unclassified, when buffer_geom is None (no buffer file
        found on disk) -- callers should treat this as "can't tell", not
        as "confirmed normal".
    Shared by pages/ais_gap.py and pages/alerts.py so both apply the
    exact same rule to gap events, whichever source produced them
    (GFW's official gaps-events dataset or the client-side reconstruction
    below).
    """
    if buffer_geom is None:
        return "gap"

    from shapely.geometry import Point

    def _outside(lat, lon):
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return None
        try:
            return not buffer_geom.contains(Point(float(lon), float(lat)))
        except Exception:
            return None

    off_outside = _outside(off_lat, off_lon)
    on_outside = _outside(on_lat, on_lon)
    return "suspicious" if (off_outside and on_outside) else "normal"


def get_monthly_chunks(start_date, end_date):
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    chunks = []
    current_start = start
    while current_start < end:
        month_end   = current_start + pd.offsets.MonthEnd(0)
        current_end = min(month_end, end)
        chunks.append((
            current_start.strftime('%Y-%m-%d'),
            current_end.strftime('%Y-%m-%d'),
        ))
        current_start = current_end + pd.Timedelta(days=1)
    return chunks


# ── AIS GAPS — official GFW "public-global-gaps-events" dataset ────────────
#
# NOTE: this replaces trying to reconstruct gaps by diffing rows of the
# 4Wings AIS-presence heatmap report (load_VP_data_by_vessel / VP_map.py
# logic, still below for reference). Two problems made that approach
# unreliable:
#   1. `vessel_id` is NOT an officially supported filter on the 4Wings
#      AIS-presence endpoint (create_ais_presence_report only documents
#      `flag`, `vessel_type`, `speed` as allowed filters -- vessel_id
#      isn't one of them).
#   2. That report is a spatio-temporal BINNED heatmap product (a `date`
#      + `hours` of activity per grid cell), not a raw timestamped ping
#      timeline -- diffing consecutive rows doesn't reliably reflect
#      real AIS transmission gaps.
#
# The Events API is built exactly for this: GFW computes each gap
# server-side from the real raw AIS transmissions (off/on positions,
# exact duration, distance, implied speed), and `vessels` (a list of
# vessel_id) IS an officially documented filter there.

def _flatten_gap_event(item):
    """Flattens one gfwapiclient Gap EventItem (nested vessel/gap/position
    pydantic sub-models) into a single flat dict/row."""
    d = item.model_dump()
    vessel = d.get("vessel") or {}
    gap = d.get("gap") or {}
    off_pos = gap.get("off_position") or {}
    on_pos = gap.get("on_position") or {}
    return {
        "event_id": d.get("id"),
        "vessel_id": vessel.get("id"),
        "ship_name": vessel.get("name"),
        "mmsi": vessel.get("ssvid"),
        "flag": vessel.get("flag"),
        "vessel_type": vessel.get("type"),
        "gear_type": None,  # not provided by the gaps-events dataset
        "start": d.get("start"),
        "end": d.get("end"),
        "duration_hrs": gap.get("duration_hours"),
        "distance_km": gap.get("distance_km"),
        "implied_speed_knots": gap.get("implied_speed_knots"),
        "intentional_disabling": gap.get("intentional_disabling"),
        "off_lat": off_pos.get("lat"),
        "off_lon": off_pos.get("lon"),
        "on_lat": on_pos.get("lat"),
        "on_lon": on_pos.get("lon"),
    }


async def load_gap_events(vessel_ids, flags, vessel_types, start, end, client, max_retries=10):
    """
    Downloads real AIS-gap events for the Aegean region/period, straight
    from GFW's official "public-global-gaps-events" dataset. Pass
    vessel_ids for a single/specific-vessel lookup (mirrors
    load_VP_data_by_vessel's calling convention), or flags/vessel_types
    for a broader filter (mirrors load_VP_data). All are optional and
    combinable.

    Returns a flat DataFrame, one row per gap event -- see
    _flatten_gap_event for the columns.
    """
    kwargs = {}
    if vessel_ids:
        kwargs["vessels"] = vessel_ids if isinstance(vessel_ids, list) else [vessel_ids]
    if flags:
        kwargs["flags"] = flags if isinstance(flags, list) else [flags]
    if vessel_types:
        kwargs["vessel_types"] = [str(v).upper() for v in vessel_types]

    all_rows = []
    offset = 0
    page_size = 99999  # gap events are far fewer than raw pings; usually a single page

    while True:
        attempt = 0
        while True:
            try:
                result = await client.events.get_all_events(
                    datasets=["public-global-gaps-events:latest"],
                    types=["GAP"],
                    start_date=start,
                    end_date=end,
                    geometry=AFE_REGION_GEOJSON,
                    limit=page_size,
                    offset=offset,
                    **kwargs,
                )
                break
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "Too Many Requests" in msg or "concurrent report" in msg
                attempt += 1
                if is_rate_limit and attempt <= max_retries:
                    wait_s = min(10 * attempt, 60)
                    print(f"    429 Too Many Requests -- retry {attempt}/{max_retries} in {wait_s}s...")
                    await asyncio.sleep(wait_s)
                    continue
                raise

        items = result.data()
        if items is None:
            items = []
        elif not isinstance(items, list):
            items = [items]
        if not items:
            break

        all_rows.extend(_flatten_gap_event(item) for item in items)
        if len(items) < page_size:
            break
        offset += page_size

    print(f"    {start} -> {end} : {len(all_rows)} gap event(s) | "
          f"vessels={kwargs.get('vessels')} flags={kwargs.get('flags')} vessel_types={kwargs.get('vessel_types')}")

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


async def bulk_load_gap_events_dataframe(flags, vessel_types, start_date, end_date, client,
                                          vessel_ids=None, progress_callback=None):
    """
    Drop-in async entry point used by pages/ais_gap.py. No month-by-month
    chunking needed here (unlike bulk_load_vp_dataframe) -- gap EVENTS
    are orders of magnitude fewer than raw position pings, so a single
    (paginated) call for the whole period is enough.
    """
    if progress_callback:
        progress_callback(f"Gap events {start_date} -> {end_date}...", 1.0)
    return await load_gap_events(vessel_ids, flags, vessel_types, start_date, end_date, client)


# ── OLD: 4Wings AIS-presence report (kept for reference / other pages) ─────

async def load_VP_data(flags, vessel_types, start, end, client, max_retries=10):
    filter_parts = []

    if flags:
        if isinstance(flags, list) and len(flags) > 1:
            flags_joined = ", ".join(f"'{f}'" for f in flags)
            flag_filter = f"flag IN ({flags_joined})"
        elif isinstance(flags, list):
            flag_filter = f"flag = '{flags[0]}'"
        else:
            flag_filter = f"flag = '{flags}'"
        filter_parts.append(flag_filter)

    if vessel_types:
        if len(vessel_types) == 1:
            filter_parts.append(f"vessel_type = '{vessel_types[0].lower()}'")
        else:
            vt = ", ".join(f"'{t.lower()}'" for t in vessel_types)
            filter_parts.append(f"vessel_type IN ({vt})")

    filter_string = " AND ".join(filter_parts)

    # GFW refuse d'exécuter 2 rapports en même temps avec le même token
    # ("Too Many Requests" / 429 "not currently enabled to perform more than
    # one concurrent report"). Cela arrive même en usage normal si un appel
    # précédent n'a pas fini de se clôturer côté serveur GFW. On réessaie
    # donc automatiquement avec un délai croissant avant d'abandonner.
    attempt = 0
    while True:
        try:
            presence_report = await client.fourwings.create_ais_presence_report(
                spatial_resolution="HIGH",
                temporal_resolution="HOURLY",
                group_by="VESSEL_ID",
                filters=[filter_string] if filter_string else [],
                start_date=start,
                end_date=end,
                geojson=AFE_REGION_GEOJSON,
            )
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "Too Many Requests" in msg or "concurrent report" in msg
            attempt += 1
            if is_rate_limit and attempt <= max_retries:
                wait_s = min(10 * attempt, 60)  # 10s, 20s, 30s... plafonné à 60s
                print(f"    429 Too Many Requests -- retry {attempt}/{max_retries} in {wait_s}s...")
                await asyncio.sleep(wait_s)
                continue
            raise

    dataframe = presence_report.df()
    print(f"    {start} -> {end} : {len(dataframe)} rows | filter: {filter_string or 'none'}")
    return dataframe


async def bulk_load_data_to_csv(flags, vessel_types, start_date, end_date, client,
                                 csv_path, progress_callback=None):
    """
    Télécharge mois par mois et écrit directement sur disque au fur et à
    mesure (mode append), au lieu d'accumuler tous les mois en mémoire puis
    de faire un pd.concat géant a la fin. Sur une grosse sélection (ALL
    pays + ALL types + HOURLY + plusieurs mois), l'ancienne approche pouvait
    consommer plusieurs Go de RAM d'un coup et planter (MemoryError).
    Retourne (total_rows, actual_start, actual_end) -- plus de dataframe
    complet retourné, tout est déjà sur disque.
    """
    chunks = get_monthly_chunks(start_date, end_date)
    total = len(chunks)
    total_rows = 0
    min_date, max_date = None, None
    header_written = False

    for i, (start, end) in enumerate(chunks):
        status = f"Loading {start} to {end}..."
        if progress_callback:
            progress_callback(status, i / total)

        df_month = await load_VP_data(flags, vessel_types, start, end, client)

        if not df_month.empty:
            date_col = ('timestamp' if 'timestamp' in df_month.columns
                        else 'date' if 'date' in df_month.columns else None)
            if date_col:
                df_month["date"] = pd.to_datetime(df_month[date_col], errors="coerce")
                df_month["year"] = df_month["date"].dt.year
                df_month["month"] = df_month["date"].dt.month
                ts = df_month["date"]
                mn, mx = ts.min(), ts.max()
                min_date = mn if min_date is None else min(min_date, mn)
                max_date = mx if max_date is None else max(max_date, mx)

            df_month.to_csv(csv_path, mode="a", header=not header_written, index=False)
            header_written = True
            total_rows += len(df_month)
            del df_month  # libère la mémoire immédiatement, avant le mois suivant

        # Pause entre 2 requêtes -- évite de déclencher le 429 "concurrent
        # report" de GFW quand il y a plusieurs mois à charger. Plus généreuse
        # sur les grosses requêtes (ALL pays + ALL types) qui prennent plus de
        # temps à se "libérer" côté serveur GFW.
        if i < total - 1:
            await asyncio.sleep(3)

    actual_start = min_date.strftime('%Y-%m-%d') if min_date is not None else start_date
    actual_end = max_date.strftime('%Y-%m-%d') if max_date is not None else end_date
    return total_rows, actual_start, actual_end


def test_api_key(api_key: str) -> tuple[bool, str]:
    try:
        import httpx
        r = httpx.get(
            "https://gateway.api.globalfishingwatch.org/v3/vessels/search",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"query": "test",
                    "datasets[0]": "public-global-vessel-identity:latest",
                    "limit": 1},
            timeout=10,
        )
        if r.status_code == 200:
            return True, "API key valid"
        elif r.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        else:
            return False, f"API returned status {r.status_code}"
    except Exception as e:
        return False, f"Connection error: {e}"


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = 'timestamp' if 'timestamp' in df.columns else 'date' if 'date' in df.columns else None
    if date_col:
        df["date"]  = pd.to_datetime(df[date_col], errors="coerce")
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month
    return df


def list_downloaded_csvs(data_dir) -> list[dict]:
    csv_dir = Path(data_dir) / "gfw_downloads"
    if not csv_dir.exists():
        return []
    files = []
    for p in sorted(csv_dir.glob("*.csv"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        files.append({
            "filename":      p.name,
            "path":          str(p),
            "size_kb":       p.stat().st_size // 1024,
            "date_modified": datetime.fromtimestamp(
                p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return files


async def load_VP_data_by_vessel(vessel_ids, start, end, client, max_retries=10,
                                 flags=None, vessel_types=None):
    """
    Same underlying report as load_VP_data (create_ais_presence_report,
    HIGH/HOURLY, group_by VESSEL_ID) but narrowed down to one or more
    specific vessel_id(s) -- this is what a single-vessel AIS-gap lookup
    needs: the actual position pings of ONE (or a few) vessel(s) over the
    period, so gaps between consecutive pings can be computed client-side
    (see pages/ais_gap.py's gap reconstruction).

    BUGFIX (was broken): a previous version tried to filter server-side
    with "vessel_id = '...'" / "vessel_id IN (...)" passed through the
    `filters` list, on the assumption that this worked the same way it
    does for load_AFE_data's report. That assumption was WRONG: per the
    installed gfw-api-python-client's own docs,
    create_ais_presence_report only accepts `flag`, `vessel_type` and
    `speed` as filters -- vessel_id is only a valid filter on
    create_fishing_effort_report (a different report/dataset), matching
    the note already above load_gap_events in this file. Passing an
    unsupported field there doesn't raise -- the API just matches nothing,
    so this silently returned an empty dataframe every time, which is why
    the "Single vessel" AIS-gap lookup stopped finding anything.

    FIX: no longer send vessel_id as a server-side filter. Instead:
      - optionally narrow the query server-side with `flags`/`vessel_types`
        (the SAME two fields load_VP_data already filters on, using the
        actual identity info of the vessel(s) being looked up, when the
        caller has it) to keep the download reasonable,
      - then filter the returned dataframe to the requested vessel_id(s)
        CLIENT-SIDE (correct and cheap: the report is already grouped by
        VESSEL_ID, so every row carries a vessel_id column).
    This is slower without a flag/vessel_type hint (downloads every
    vessel's pings for the region/period instead of just one), but it is
    correct, which matters more for gap detection than raw speed --
    pass `flags`/`vessel_types` when available to cut the download down.
    """
    if isinstance(vessel_ids, str):
        vessel_ids = [vessel_ids]
    if not vessel_ids:
        return pd.DataFrame()
    vessel_ids_set = {str(v) for v in vessel_ids}

    filter_parts = []
    if flags:
        if isinstance(flags, list) and len(flags) > 1:
            flags_joined = ", ".join(f"'{f}'" for f in flags)
            filter_parts.append(f"flag IN ({flags_joined})")
        elif isinstance(flags, list):
            filter_parts.append(f"flag = '{flags[0]}'")
        else:
            filter_parts.append(f"flag = '{flags}'")
    if vessel_types:
        if len(vessel_types) == 1:
            filter_parts.append(f"vessel_type = '{vessel_types[0].lower()}'")
        else:
            vt = ", ".join(f"'{t.lower()}'" for t in vessel_types)
            filter_parts.append(f"vessel_type IN ({vt})")
    filter_string = " AND ".join(filter_parts)

    attempt = 0
    while True:
        try:
            presence_report = await client.fourwings.create_ais_presence_report(
                spatial_resolution="HIGH",
                temporal_resolution="HOURLY",
                group_by="VESSEL_ID",
                filters=[filter_string] if filter_string else [],
                start_date=start,
                end_date=end,
                geojson=AFE_REGION_GEOJSON,
            )
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "Too Many Requests" in msg or "concurrent report" in msg
            attempt += 1
            if is_rate_limit and attempt <= max_retries:
                wait_s = min(10 * attempt, 60)
                print(f"    429 Too Many Requests -- retry {attempt}/{max_retries} in {wait_s}s...")
                await asyncio.sleep(wait_s)
                continue
            raise

    dataframe = presence_report.df()
    total_rows = len(dataframe)
    if not dataframe.empty and "vessel_id" in dataframe.columns:
        dataframe = dataframe[dataframe["vessel_id"].astype(str).isin(vessel_ids_set)]
    print(f"    {start} -> {end} : {total_rows} row(s) downloaded, "
          f"{len(dataframe)} kept after client-side vessel_id filter "
          f"({len(vessel_ids_set)} vessel(s)) | server filter: {filter_string or 'none'}")
    return dataframe


async def bulk_load_vp_dataframe(flags, vessel_types, start_date, end_date, client,
                                  vessel_ids=None, progress_callback=None):
    """
    Month-by-month position download, concatenated in memory (no CSV) --
    used by the AIS-gaps page. Pass vessel_ids for a single/specific
    vessel lookup, or leave it None and use flags/vessel_types for the
    "all vessels matching this filter" mode, mirroring
    bulk_load_afe_dataframe's two modes.

    When BOTH vessel_ids and flags/vessel_types are given, flags/
    vessel_types are used as a server-side narrowing hint for the
    vessel_id lookup (see load_VP_data_by_vessel) -- vessel_id itself is
    always applied client-side, since it isn't a supported filter on this
    report.
    """
    chunks = get_monthly_chunks(start_date, end_date)
    all_months = []

    for i, (start, end) in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Positions {start} -> {end}...", (i + 1) / max(len(chunks), 1))
        if vessel_ids:
            df_month = await load_VP_data_by_vessel(vessel_ids, start, end, client,
                                                     flags=flags, vessel_types=vessel_types)
        else:
            df_month = await load_VP_data(flags, vessel_types, start, end, client)
        if df_month is not None and not df_month.empty:
            all_months.append(df_month)
        if i < len(chunks) - 1:
            await asyncio.sleep(3)

    if not all_months:
        return pd.DataFrame()
    return pd.concat(all_months, ignore_index=True)


# ── AIS GAPS — client-side reconstruction from raw hourly VP pings ─────────
#
# WHY THIS EXISTS ALONGSIDE load_gap_events: GFW's official
# "public-global-gaps-events" dataset only publishes a gap if it starts
# at least 50 NAUTICAL MILES FROM SHORE (see GFW's own data-caveats doc:
# https://globalfishingwatch.org/our-apis/documentation/docs/v3/general-api-doc/data-caveats).
# That's a deliberate methodological choice on GFW's side (near-shore
# AIS gaps are usually reception interference, not real disabling), but
# it structurally excludes almost the entire Aegean fleet: the Aegean is
# an archipelago, so most of it is well within 50nm of some coast.
# Testing confirmed this -- the official dataset returned 0 gap events
# for every Greek vessel tried, even ones with known long AIS silences.
#
# This function reconstructs gaps the same way the old Tkinter app did
# (VP_map.py / VP_bulk_map.py): diff consecutive hourly position pings
# per vessel from the 4Wings AIS-presence report, and flag any gap over
# a (very low) sanity threshold. It has NO distance-from-shore cutoff, so
# it can actually see near-coast AIS silence. The tradeoff (see the note
# above load_gap_events) is that it's a heuristic on a spatio-temporal
# BINNED product, not a validated official GFW event -- expect more
# false positives from reception artifacts near the coast. That's
# exactly what the buffer-based classify_gap_status() split is for:
# treat "suspicious" (both ends outside the buffer) as the real signal,
# and "normal" (near the buffer) as likely noise.

def compute_client_side_gap_events(vp_df, min_hours=0.0):
    """
    Reconstructs AIS-gap "events" from a raw 4Wings AIS-presence
    dataframe (load_VP_data / load_VP_data_by_vessel / bulk_load_vp_dataframe),
    by diffing consecutive position timestamps per vessel.

    Returns a flat DataFrame, one row per consecutive-ping gap of at
    least `min_hours`, with the SAME columns as load_gap_events's output
    (vessel_id, ship_name, mmsi, flag, vessel_type, gear_type, start, end,
    duration_hrs, off_lat, off_lon, on_lat, on_lon) -- so it's a drop-in
    replacement anywhere a gap-events dataframe is expected, e.g.
    pages/ais_gap.py::_process_gap_events. `gear_type` is included when
    present in vp_df (unlike the official dataset, which never carries it).

    min_hours lets a caller pre-trim obviously-uninteresting rows before
    any UI threshold is applied (e.g. alerts.py, which only ever cares
    about gaps above its fixed GAP_THRESHOLD_HOURS) -- leave it at 0 to
    keep every consecutive-ping gap and let the caller decide the cutoff
    (e.g. ais_gap.py, whose threshold is user-adjustable).
    """
    base_cols = ["vessel_id", "ship_name", "mmsi", "flag", "vessel_type", "gear_type",
                 "start", "end", "duration_hrs", "off_lat", "off_lon", "on_lat", "on_lon"]
    if vp_df is None or vp_df.empty or "vessel_id" not in vp_df.columns:
        return pd.DataFrame(columns=base_cols)

    date_col = "date" if "date" in vp_df.columns else ("timestamp" if "timestamp" in vp_df.columns else None)
    if date_col is None or "lat" not in vp_df.columns or "lon" not in vp_df.columns:
        return pd.DataFrame(columns=base_cols)

    d = vp_df.copy()
    d["_date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=["_date", "lat", "lon", "vessel_id"]).sort_values(["vessel_id", "_date"])

    rows = []
    for vessel_id, grp in d.groupby("vessel_id", sort=False):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue

        def _first(col):
            if col not in grp.columns:
                return None
            s = grp[col].dropna()
            return s.iloc[0] if not s.empty else None

        ship_name = _first("ship_name")
        mmsi = _first("mmsi")
        flag = _first("flag")
        vessel_type = _first("vessel_type")
        gear_type = _first("gear_type")

        dates = grp["_date"].values
        lats = grp["lat"].values
        lons = grp["lon"].values
        for i in range(len(grp) - 1):
            gap_hours = (dates[i + 1] - dates[i]) / pd.Timedelta(hours=1)
            if gap_hours < min_hours:
                continue
            rows.append({
                "vessel_id": vessel_id,
                "ship_name": ship_name,
                "mmsi": mmsi,
                "flag": flag,
                "vessel_type": vessel_type,
                "gear_type": gear_type,
                "start": grp["_date"].iloc[i],
                "end": grp["_date"].iloc[i + 1],
                "duration_hrs": gap_hours,
                "off_lat": lats[i],
                "off_lon": lons[i],
                "on_lat": lats[i + 1],
                "on_lon": lons[i + 1],
            })

    if not rows:
        return pd.DataFrame(columns=base_cols)
    return pd.DataFrame(rows)


async def bulk_load_client_gap_events_dataframe(flags, vessel_types, start_date, end_date, client,
                                                  vessel_ids=None, progress_callback=None, min_hours=0.0):
    """
    Client-side counterpart to bulk_load_gap_events_dataframe: downloads
    raw hourly AIS-presence pings (bulk_load_vp_dataframe -- month-by-
    month, same as before) and reconstructs gap events locally
    (compute_client_side_gap_events) instead of querying GFW's official
    gaps-events dataset. See the comment above compute_client_side_gap_events
    for why.

    When vessel_ids is given, flags/vessel_types (if also given) are used
    purely to narrow the server-side download -- the actual vessel_id
    match is always applied client-side (see load_VP_data_by_vessel for
    why: vessel_id isn't a supported filter on this report).

    PERFORMANCE NOTE: this downloads raw position pings, which is much
    heavier than the official events-API call (few, pre-computed events)
    -- expect it to be noticeably slower on a large vessel set or a long
    date range, since it now needs the same month-by-month chunking as
    any other raw VP download. Without a flags/vessel_types hint, a
    single-vessel lookup downloads every vessel's pings for the whole
    region/period before filtering down to the one requested -- correct,
    but the heaviest case; pass flags/vessel_types when known to cut this
    down a lot.
    """
    vp_df = await bulk_load_vp_dataframe(flags, vessel_types, start_date, end_date, client,
                                          vessel_ids=vessel_ids, progress_callback=progress_callback)
    return compute_client_side_gap_events(vp_df, min_hours=min_hours)


# =============================================================================
# AFE (Apparent Fishing Effort) -- porté depuis l'ancienne app Tkinter.
# _AFE_REGION_POLY est déjà défini plus haut dans ce fichier.
# =============================================================================

async def load_AFE_data(flags, start, end, client, vessel_types=None, vessel_ids=None,
                        max_retries=10):
    """
    Télécharge l'effort de pêche apparent (AFE) sur la région Égée élargie,
    avec filtres optionnels et combinables :
      - flags        : pavillon(s) (liste ou string). Peut être None/vide si
                        vessel_ids est fourni (mode single vessel : le
                        vessel_id suffit, pas besoin de filtrer par pavillon).
      - vessel_types : type(s) de navire GFW (ex: "fishing"), meme logique
                        que load_VP_data.
      - vessel_ids   : identifiant(s) GFW precis (mode single vessel), meme
                        principe que le filtre flag/vessel_type -- une
                        clause "vessel_id = '...'" / "vessel_id IN (...)"
                        est ajoutee a la liste de filtres. NOTE : la
                        colonne vessel_id est bien presente dans les
                        resultats (group_by="VESSEL_ID"), mais la
                        possibilite de filtrer DESSUS cote API n'a pas pu
                        etre confirmee avec la doc en direct -- si l'API
                        rejette ce filtre, l'erreur GFW remontera telle
                        quelle a l'appelant (a ajuster si besoin, sur le
                        meme modele que la remarque equivalente dans
                        pages/ais_gap.py::load_ais_gaps_bulk).
    Ne garde que les navires de pêche (vessel_type == FISHING) SAUF si
    vessel_ids est fourni : dans ce cas le navire demande est garde tel
    quel, meme si GFW ne le categorise pas FISHING.
    Même gestion des 429 (rate limit) que load_VP_data.
    """
    region_geometry = AFE_REGION_GEOJSON

    api_filters = ["distance_from_port_km > 1"]

    if flags:
        if isinstance(flags, list) and len(flags) > 1:
            api_filters.append("flag IN (" + ", ".join(f"'{f}'" for f in flags) + ")")
        elif isinstance(flags, list) and flags:
            api_filters.append(f"flag = '{flags[0]}'")
        elif isinstance(flags, str):
            api_filters.append(f"flag = '{flags}'")

    if vessel_types:
        if len(vessel_types) == 1:
            api_filters.append(f"vessel_type = '{vessel_types[0].lower()}'")
        else:
            vt = ", ".join(f"'{t.lower()}'" for t in vessel_types)
            api_filters.append(f"vessel_type IN ({vt})")

    if vessel_ids:
        if isinstance(vessel_ids, str):
            vessel_ids = [vessel_ids]
        if len(vessel_ids) == 1:
            api_filters.append(f"vessel_id = '{vessel_ids[0]}'")
        else:
            vids = ", ".join(f"'{v}'" for v in vessel_ids)
            api_filters.append(f"vessel_id IN ({vids})")

    for attempt in range(max_retries):
        try:
            report = await client.fourwings.create_fishing_effort_report(
                spatial_resolution="HIGH",
                temporal_resolution="HOURLY",
                group_by="VESSEL_ID",
                filters=api_filters,
                start_date=start,
                end_date=end,
                geojson=region_geometry,
            )
            break
        except Exception as e:
            msg = str(e)
            is_rate_limit = ("429" in msg or "Too Many Requests" in msg
                             or "concurrent report" in msg)
            if is_rate_limit and attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            raise

    df = report.df()
    if df.empty:
        return df
    if "vessel_type" in df.columns and not vessel_ids:
        df = df[df["vessel_type"] == "FISHING"]
    return df


async def bulk_load_afe_to_csv(flags, start_date, end_date, client, csv_path,
                               progress_callback=None):
    """
    Version AFE de bulk_load_data_to_csv : télécharge mois par mois et écrit
    le CSV. Renvoie le chemin du CSV.
    """
    chunks = get_monthly_chunks(start_date, end_date)
    all_months = []

    for i, (start, end) in enumerate(chunks):
        if progress_callback:
            progress_callback(f"AFE {start} -> {end}...", (i + 1) / max(len(chunks), 1))
        df_month = await load_AFE_data(flags, start, end, client)
        if df_month is not None and not df_month.empty:
            all_months.append(df_month)

    if not all_months:
        pd.DataFrame().to_csv(csv_path, index=False)
        return csv_path

    df = pd.concat(all_months, ignore_index=True)
    df.to_csv(csv_path, index=False)
    return csv_path


async def bulk_load_afe_dataframe(flags, start_date, end_date, client,
                                  vessel_types=None, vessel_ids=None,
                                  progress_callback=None):
    """
    Comme bulk_load_afe_to_csv, mais renvoie directement un DataFrame
    concaténé en mémoire (pas d'écriture sur disque) -- utilisé par la
    heatmap AFE (single vessel ou filtrée par flag(s)/type(s)), qui n'a
    besoin que des positions lat/lon, pas d'un fichier CSV persistant.

    flags peut être None (pas de filtre pavillon), utile en mode single
    vessel où seul vessel_ids importe.
    """
    chunks = get_monthly_chunks(start_date, end_date)
    all_months = []

    for i, (start, end) in enumerate(chunks):
        if progress_callback:
            progress_callback(f"AFE {start} -> {end}...", (i + 1) / max(len(chunks), 1))
        df_month = await load_AFE_data(flags, start, end, client,
                                        vessel_types=vessel_types, vessel_ids=vessel_ids)
        if df_month is not None and not df_month.empty:
            all_months.append(df_month)
        if i < len(chunks) - 1:
            await asyncio.sleep(3)

    if not all_months:
        return pd.DataFrame()
    return pd.concat(all_months, ignore_index=True)