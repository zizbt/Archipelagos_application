"""
pages/stats.py
==============
Page "Statistics" -- v3 : les 5 catégories, calculées EN DIRECT à partir
d'un CSV importé par l'utilisateur (même pattern que pages/heatmap.py :
zone d'upload + cache serveur, aucune donnée précalculée).

  1. Vessel count by type
  2. Vessel activity by year
  3. Seasonal pattern by year
  4. Vessel type composition per season
  5. Protected area — vessels detected inside protected areas, by year
     (test géométrique point-in-polygon sur lat/lon vs MARINE_ZONES)

Il n'y a plus de filtre "années précalculées" ni de sélecteur de CSV déjà
téléchargé sur le disque : tout part du fichier que l'utilisateur dépose,
exactement comme sur la page Heatmaps.
"""

import base64
import io

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MARINE_ZONES, lbl
from config import TYPE_COLORS, DEFAULT_COLOR

# Mêmes saisons météorologiques que pages/heatmap.py, calculées à partir
# de la colonne "date" du CSV -- aucune donnée précalculée par saison.
SEASON_MONTHS = {
    "Winter": {12, 1, 2},
    "Spring": {3, 4, 5},
    "Summer": {6, 7, 8},
    "Fall": {9, 10, 11},
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]

PLACEHOLDER_STYLE = {"color": DIM, "padding": "2rem", "fontSize": "0.85rem", "fontStyle": "italic"}

# Cache serveur du CSV importé, comme _CSV_CACHE dans pages/heatmap.py --
# évite de faire l'aller-retour du dataframe par le navigateur.
_CSV_CACHE = {"df": None, "filename": None}


def _rgb(vtype):
    c = TYPE_COLORS.get(str(vtype).upper(), DEFAULT_COLOR)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _dark_layout(fig, title, height=300):
    fig.update_layout(template="plotly_dark", height=height, title=title,
                       paper_bgcolor=BG, plot_bgcolor=BG, font_color=SOFT,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _parse_uploaded_csv(contents, filename):
    """Identique à heatmap._parse_uploaded_csv (dcc.Upload -> DataFrame)."""
    if contents is None:
        return None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
    return pd.read_csv(io.BytesIO(decoded))


def _assign_year_season(df, date_col="date"):
    """Ajoute les colonnes '_year' et '_season' à partir de df[date_col]."""
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    year = parsed.dt.year
    month = parsed.dt.month
    season = pd.Series(index=df.index, dtype="object")
    for name, months in SEASON_MONTHS.items():
        season[month.isin(months)] = name
    return year, season


def _marine_polygons():
    """Essaie de récupérer une liste de géométries shapely depuis MARINE_ZONES.
    Tolère plusieurs formats (Feature GeoJSON, geometry brute, dict avec clé
    'geometry'/'geojson'...) et ignore silencieusement ce qu'il ne reconnaît
    pas -- on ne veut jamais planter la page pour ça."""
    try:
        from shapely.geometry import shape
    except ImportError:
        return None

    polys = []
    zones = MARINE_ZONES or []
    if isinstance(zones, dict):
        zones = zones.get("features", zones.get("zones", [zones]))

    for item in zones:
        geom = None
        if isinstance(item, dict):
            if "geometry" in item:
                geom = item["geometry"]
            elif "geojson" in item:
                geom = item["geojson"]
            elif "type" in item and "coordinates" in item:
                geom = item
        if geom is None:
            continue
        try:
            polys.append(shape(geom))
        except Exception:
            continue
    return polys or None


def _points_in_protected_area(df):
    """Retourne un masque booléen (index de df) indiquant quelles lignes
    (lat/lon) tombent dans une zone marine protégée. None si le test n'a
    pas pu être fait (pas de shapely, pas de géométrie exploitable, ou pas
    de colonnes lat/lon dans le CSV)."""
    if "lat" not in df.columns or "lon" not in df.columns:
        return None

    polys = _marine_polygons()
    if not polys:
        return None

    try:
        from shapely.ops import unary_union
        from shapely.prepared import prep
        union_geom = prep(unary_union(polys))
    except Exception:
        return None

    try:
        from shapely.vectorized import contains
        import numpy as np
        lon = df["lon"].to_numpy(dtype=float)
        lat = df["lat"].to_numpy(dtype=float)
        # shapely.vectorized veut la géométrie non "prepared"
        raw_union = unary_union(polys)
        mask = contains(raw_union, lon, lat)
        return pd.Series(mask, index=df.index)
    except Exception:
        # Repli plus lent mais robuste : test point par point.
        from shapely.geometry import Point

        def _test(row):
            try:
                return union_geom.contains(Point(row["lon"], row["lat"]))
            except Exception:
                return False

        return df.apply(_test, axis=1)


def _upload_zone():
    return dcc.Upload(
        id="stats-csv-upload",
        children=html.Div([
            "Drag a CSV here, or ",
            html.A("browse", style={"color": ACC, "textDecoration": "underline"}),
        ]),
        style={
            "width": "100%", "maxWidth": "420px", "padding": "1rem 0.5rem",
            "textAlign": "center", "cursor": "pointer",
            "border": f"1px dashed {BDR}", "borderRadius": "6px",
            "color": SOFT, "fontSize": "0.75rem",
        },
        multiple=False,
    )


def layout():
    return html.Div([
        dcc.Store(id="stats-store-csv-df", data=None),

        html.Div([
            html.Div([lbl("Import a CSV"), _upload_zone()],
                     style={"marginRight": "1.5rem"}),

            html.Div([lbl("Year (for seasonal graphs)"),
                dcc.Dropdown(id="stats-year", options=[], value=None,
                    placeholder="Import a CSV first...",
                    clearable=False, style={"width": "160px", "color": "#000"})],
                style={"marginRight": "1rem"}),
        ], style={"display": "flex", "alignItems": "flex-end", "gap": "0.5rem",
                   "flexWrap": "wrap", "marginBottom": "0.5rem"}),

        html.Div("No file selected", id="stats-csv-filename",
                  style={"fontSize": "0.72rem", "color": DIM,
                         "fontStyle": "italic", "marginBottom": "1.2rem"}),

        html.Div(id="stats-container", children=html.P(
            "Import a CSV to see the statistics.", style=PLACEHOLDER_STYLE),
            style={"padding": "0 0.2rem"}),

    ], style={"padding": "1.5rem", "background": BG,
              "height": "calc(100vh - 52px)", "overflowY": "auto"})


def register_callbacks(app):

    # Parse le CSV dès qu'il est déposé, uniquement pour peupler le
    # sélecteur d'années -- les graphiques sont reconstruits juste après
    # via le callback ci-dessous (déclenché par le changement d'année).
    @app.callback(
        Output("stats-csv-filename", "children"),
        Output("stats-year", "options"),
        Output("stats-year", "value"),
        Output("stats-store-csv-df", "data"),
        Input("stats-csv-upload", "contents"),
        State("stats-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv_uploaded(contents, filename):
        if not contents:
            raise dash.exceptions.PreventUpdate
        try:
            df = _parse_uploaded_csv(contents, filename)
        except Exception as e:
            _CSV_CACHE["df"] = None
            return (f"Error: {e}", [], None, None)

        if "date" not in df.columns:
            _CSV_CACHE["df"] = None
            return (f'"{filename}" has no "date" column -- can\'t compute year/season stats.',
                    [], None, None)

        year, _season = _assign_year_season(df)
        df = df.copy()
        df["_year"] = year

        _CSV_CACHE["df"] = df
        _CSV_CACHE["filename"] = filename

        years = sorted(df["_year"].dropna().astype(int).unique(), reverse=True)
        year_opts = [{"label": str(y), "value": y} for y in years]
        default_year = years[0] if years else None

        return (f"Loaded: {len(df):,} rows from \"{filename}\"", year_opts, default_year,
                "loaded")

    @app.callback(
        Output("stats-container", "children"),
        Input("stats-store-csv-df", "data"),
        Input("stats-year", "value"),
        prevent_initial_call=True,
    )
    def _update_stats(_store_flag, year):
        df = _CSV_CACHE.get("df")
        if df is None or df.empty:
            return html.P("Import a CSV to see the statistics.", style=PLACEHOLDER_STYLE)

        sections = []

        row1 = html.Div([
            html.Div(_graph_vessel_count_by_type(df), style={"flex": "1", "minWidth": "380px"}),
            html.Div(_graph_activity_by_year(df), style={"flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1rem"})
        sections.append(row1)

        if year is not None:
            row2 = html.Div([
                html.Div(_graph_seasonal_pattern(df, year), style={"flex": "1", "minWidth": "380px"}),
                html.Div(_graph_type_composition_per_season(df, year), style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1rem"})
            sections.append(row2)

        # ── Protected areas ──────────────────────────────────────────────
        mask = _points_in_protected_area(df)
        if mask is None:
            reason = ("this CSV has no \"lat\"/\"lon\" columns" if
                       ("lat" not in df.columns or "lon" not in df.columns)
                       else "no usable protected-area geometry was found")
            sections.append(html.P(
                f"Protected area stats unavailable: {reason}.",
                style={"color": DIM, "fontStyle": "italic", "marginTop": "0.5rem"}))
        else:
            in_zone = df[mask]
            if in_zone.empty:
                sections.append(html.P(
                    "No detections fall inside a protected area in this CSV.",
                    style={"color": DIM, "fontStyle": "italic", "marginTop": "0.5rem"}))
            else:
                sections.append(html.H6(
                    "Protected areas — vessels detected inside, by year",
                    style={"color": MAIN, "marginTop": "0.5rem", "marginBottom": "0.6rem"}))
                row3 = html.Div([
                    html.Div(_graph_protected_area_totals(in_zone), style={"flex": "1", "minWidth": "380px"}),
                    html.Div(_graph_protected_area_by_type(in_zone), style={"flex": "1", "minWidth": "380px"}),
                ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1rem"})
                sections.append(row3)

        return html.Div(sections)


def _vessel_count(df, group_col):
    if "vessel_id" in df.columns:
        return df.groupby(group_col)["vessel_id"].nunique()
    return df.groupby(group_col).size()


# ── 1. Vessel count by type ─────────────────────────────────────────────────

def _graph_vessel_count_by_type(df):
    if "vessel_type" not in df.columns:
        return html.P("No \"vessel_type\" column in this CSV.", style=PLACEHOLDER_STYLE)
    by_type = _vessel_count(df, "vessel_type").sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=by_type.index, y=by_type.values,
                            marker_color=[_rgb(t) for t in by_type.index]))
    _dark_layout(fig, "Vessel count by type")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 2. Vessel activity by year ───────────────────────────────────────────────

def _graph_activity_by_year(df):
    by_year = df.groupby("_year").size().sort_index()
    by_year.index = by_year.index.astype(int).astype(str)
    fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values, marker_color=ACC))
    _dark_layout(fig, "Vessel activity by year (AIS points)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 3. Seasonal pattern by year ─────────────────────────────────────────────

def _graph_seasonal_pattern(df, year):
    sub = df[df["_year"] == year].copy()
    _, season = _assign_year_season(sub)
    sub["_season"] = season
    by_season = sub.groupby("_season").size().reindex(SEASON_ORDER).fillna(0)
    fig = go.Figure(go.Scatter(x=by_season.index, y=by_season.values, mode="lines+markers",
                                line=dict(color=ACC, width=3), marker=dict(size=9)))
    _dark_layout(fig, f"Seasonal pattern — {year}")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 4. Vessel type composition per season ────────────────────────────────────

def _graph_type_composition_per_season(df, year):
    if "vessel_type" not in df.columns:
        return html.P("No \"vessel_type\" column in this CSV.", style=PLACEHOLDER_STYLE)
    sub = df[df["_year"] == year].copy()
    _, season = _assign_year_season(sub)
    sub["_season"] = season

    if "vessel_id" in sub.columns:
        pivot = sub.pivot_table(index="_season", columns="vessel_type", values="vessel_id",
                                 aggfunc="nunique", fill_value=0).reindex(SEASON_ORDER)
    else:
        pivot = sub.pivot_table(index="_season", columns="vessel_type", values="vessel_type",
                                 aggfunc="count", fill_value=0).reindex(SEASON_ORDER)

    fig = go.Figure()
    for vtype in pivot.columns:
        fig.add_trace(go.Bar(name=vtype, x=pivot.index, y=pivot[vtype], marker_color=_rgb(vtype)))
    fig.update_layout(barmode="stack")
    _dark_layout(fig, f"Vessel type composition per season — {year}")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 5. Protected area — vessels detected inside, by year ────────────────────

def _graph_protected_area_totals(in_zone):
    by_year = _vessel_count(in_zone, "_year").sort_index()
    by_year.index = by_year.index.astype(int).astype(str)
    fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values, marker_color=ACC))
    _dark_layout(fig, "Vessels detected in protected area, by year")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _graph_protected_area_by_type(in_zone):
    if "vessel_type" not in in_zone.columns:
        return html.P("No \"vessel_type\" column in this CSV.", style=PLACEHOLDER_STYLE)
    pivot = in_zone.pivot_table(index="_year",
                                 columns="vessel_type",
                                 values="vessel_id" if "vessel_id" in in_zone.columns else "vessel_type",
                                 aggfunc="nunique" if "vessel_id" in in_zone.columns else "count",
                                 fill_value=0).sort_index()
    pivot.index = pivot.index.astype(int).astype(str)
    fig = go.Figure()
    for vtype in pivot.columns:
        fig.add_trace(go.Bar(name=vtype, x=pivot.index, y=pivot[vtype], marker_color=_rgb(vtype)))
    fig.update_layout(barmode="stack")
    _dark_layout(fig, "Composition by vessel type, by year")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})