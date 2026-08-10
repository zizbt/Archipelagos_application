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
    "ROU", "BGR", "GEO", "ISR", "LBR", "PAN", "BHS",
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
                geojson=AEGEAN_GEOJSON,
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


# =============================================================================
# AFE (Apparent Fishing Effort) -- porté depuis l'ancienne app Tkinter.
# _AFE_REGION_POLY est déjà défini plus haut dans ce fichier.
# =============================================================================

async def load_AFE_data(flags, start, end, client, max_retries=10):
    """
    Télécharge l'effort de pêche apparent (AFE) pour un ou plusieurs pavillons,
    sur la région Égée élargie. Ne garde que les navires de pêche.
    Même gestion des 429 (rate limit) que load_VP_data.
    """
    region_geometry = mapping(_AFE_REGION_POLY)

    if isinstance(flags, list):
        flag_filter = "flag IN (" + ", ".join(["'{}'".format(f) for f in flags]) + ")"
    else:
        flag_filter = f"flag = '{flags}'"

    api_filters = [flag_filter, "distance_from_port_km > 3"]

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
    if "vessel_type" in df.columns:
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