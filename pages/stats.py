"""
pages/stats.py
==============
Page "Statistics" -- v2 : les 5 catégories demandées, toutes basées sur
les fichiers précalculés (3.5 ans de données) :

  1. Vessel count by type
  2. Vessel activity by year
  3. Seasonal pattern by year
  4. Vessel type composition per season
  5. Protected area — vessels detected inside protected areas, by year

Note : type_year_season.parquet n'a pas de colonne "flag", donc pas de
filtre pays sur ces stats précalculées (comme convenu). Le filtre pays
reste disponible via l'import d'un CSV personnel (section du bas).
"""

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, GLOBAL_STATS, GFW_DOWNLOAD_DIR, lbl
from config import YEARS, SEASON_ORDER, TYPE_COLORS, DEFAULT_COLOR, FLAG_NAMES, ROOT
from loader import load_type_year_season_stats, load_protected_area_stats
from gfw import list_downloaded_csvs, load_csv


def _rgb(vtype):
    c = TYPE_COLORS.get(str(vtype).upper(), DEFAULT_COLOR)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _dark_layout(fig, title, height=300):
    fig.update_layout(template="plotly_dark", height=height, title=title,
                       paper_bgcolor=BG, plot_bgcolor=BG, font_color=SOFT,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def layout():
    return html.Div([
        dcc.Store(id="stats-store-csv-df", data=None),

        html.Div([
            html.Div([lbl("Years available in precalculated stats"),
                dcc.Dropdown(id="stats-year", value=YEARS[-1], clearable=False,
                    options=[{"label": str(y), "value": y} for y in YEARS],
                    style={"width": "160px", "color": "#000"})],
                style={"marginRight": "1.5rem"}),

            html.Div([lbl("Import a downloaded CSV"),
                dcc.Dropdown(id="stats-csv-selector",
                    options=[{"label": f["filename"], "value": f["path"]}
                             for f in list_downloaded_csvs(ROOT / "data")],
                    value=None, placeholder="Choose a CSV...",
                    style={"width": "320px", "color": "#000"})],
                style={"marginRight": "1rem"}),
            html.Div([lbl(" "),
                html.Button("Load", id="stats-btn-load-csv", n_clicks=0,
                    style={"padding": "0.45rem 1.2rem", "background": PANEL,
                           "color": SOFT, "border": f"1px solid {BDR}",
                           "borderRadius": "6px", "cursor": "pointer"})]),
        ], style={"display": "flex", "alignItems": "flex-end", "gap": "0.5rem",
                   "flexWrap": "wrap", "marginBottom": "1.2rem"}),

        html.Div(id="stats-container", style={"padding": "0 0.2rem"}),

    ], style={"padding": "1.5rem", "background": BG,
              "height": "calc(100vh - 52px)", "overflowY": "auto"})


def register_callbacks(app):

    @app.callback(
        Output("stats-container", "children"),
        Output("stats-store-csv-df", "data"),
        Input("stats-year", "value"),
        Input("stats-btn-load-csv", "n_clicks"),
        State("stats-csv-selector", "value"),
        State("stats-store-csv-df", "data"),
        prevent_initial_call=False,
    )
    def update_stats(year, n_csv, csv_path, current_csv_json):
        trigger = dash.callback_context.triggered_id if dash.callback_context.triggered else None

        csv_json = current_csv_json
        csv_error = None
        if trigger == "stats-btn-load-csv":
            if not csv_path:
                csv_error = "Choisis un CSV d'abord."
            else:
                try:
                    df_csv = load_csv(csv_path)
                    csv_json = df_csv.to_json(orient="records")
                except Exception as e:
                    csv_error = f"Erreur : {e}"

        sections = []
        tys = load_type_year_season_stats()

        # ── Chiffres globaux ─────────────────────────────────────────────────
        if GLOBAL_STATS:
            sections.append(html.Div([
                html.P(k.replace("_", " ").title(),
                       style={"fontSize": "0.7rem", "color": DIM, "margin": "0 0 0.2rem 0"})
                for k, v in [] # placeholder, replaced below
            ]))
            sections[-1] = html.Div([
                html.Div([
                    html.P(k.replace("_", " ").title(),
                           style={"fontSize": "0.7rem", "color": DIM, "margin": "0 0 0.2rem 0"}),
                    html.P(f"{int(v):,}",
                           style={"fontSize": "1.4rem", "fontWeight": "600", "color": MAIN, "margin": 0}),
                ], style={"background": PANEL, "borderRadius": "8px", "padding": "1rem",
                           "border": f"1px solid {BDR}", "flex": "1", "minWidth": "140px"})
                for k, v in GLOBAL_STATS.items() if k not in ("year_min", "year_max")
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1.8rem"})

        if tys.empty:
            sections.append(html.P(
                "Pas de fichier type_year_season.parquet trouvé — impossible d'afficher les statistiques précalculées.",
                style={"color": "#ff6b6b", "fontStyle": "italic"}))
        else:
            row1 = html.Div([
                html.Div(_graph_vessel_count_by_type(tys), style={"flex": "1", "minWidth": "380px"}),
                html.Div(_graph_activity_by_year(tys), style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1rem"})
            sections.append(row1)

            row2 = html.Div([
                html.Div(_graph_seasonal_pattern(tys, year), style={"flex": "1", "minWidth": "380px"}),
                html.Div(_graph_type_composition_per_season(tys, year), style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1rem"})
            sections.append(row2)

        # ── Protected areas ────────────────────────────────────────
        pa_stats = load_protected_area_stats()
        if pa_stats:
            sections.append(html.H6("Protected areas — vessels detected inside, by year",
                                     style={"color": MAIN, "marginTop": "0.5rem", "marginBottom": "0.6rem"}))
            row3 = html.Div([
                html.Div(_graph_protected_area_totals(pa_stats), style={"flex": "1", "minWidth": "380px"}),
                html.Div(_graph_protected_area_by_type(pa_stats), style={"flex": "1", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1rem"})
            sections.append(row3)
        else:
            sections.append(html.P(
                "Pas de fichier protected_areas.json trouvé.",
                style={"color": DIM, "fontStyle": "italic"}))

        # ── Imported CSV stats ─────────────────
        if csv_error:
            sections.append(html.P(csv_error, style={"color": "#ff6b6b", "marginTop": "1rem"}))

        if csv_json:
            df = pd.read_json(csv_json, orient="records")
            if not df.empty:
                sections.append(html.Hr(style={"borderColor": BDR, "margin": "1.5rem 0 1rem 0"}))
                sections.append(html.H6("Statistiques du jeu de données importé",
                                         style={"color": MAIN, "marginBottom": "1rem"}))

                csv_row = []
                if "vessel_type" in df.columns:
                    by_type = df.groupby("vessel_type").size().sort_values(ascending=False)
                    fig = go.Figure(go.Bar(x=by_type.index, y=by_type.values,
                                            marker_color=[_rgb(t) for t in by_type.index]))
                    _dark_layout(fig, "Nombre de positions par type")
                    csv_row.append(html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}),
                                             style={"flex": "1", "minWidth": "380px"}))

                if "flag" in df.columns:
                    by_flag = df.groupby("flag").size().sort_values(ascending=False).head(15)
                    fig2 = go.Figure(go.Bar(x=[FLAG_NAMES.get(f, f) for f in by_flag.index],
                                             y=by_flag.values, marker_color=ACC))
                    _dark_layout(fig2, "Top 15 pays (flag)", height=320)
                    csv_row.append(html.Div(dcc.Graph(figure=fig2, config={"displayModeBar": False}),
                                             style={"flex": "1", "minWidth": "380px"}))

                if csv_row:
                    sections.append(html.Div(csv_row, style={"display": "flex", "gap": "1rem",
                                                              "flexWrap": "wrap", "marginBottom": "1rem"}))

                n_vessels = df["vessel_id"].nunique() if "vessel_id" in df.columns else "-"
                n_flags = df["flag"].nunique() if "flag" in df.columns else "-"
                sections.append(html.Div([
                    html.P(f"{len(df):,} positions AIS", style={"color": SOFT, "fontSize": "0.82rem", "margin": "0.2rem 0"}),
                    html.P(f"{n_vessels:,} bateaux uniques" if n_vessels != "-" else "-",
                           style={"color": SOFT, "fontSize": "0.82rem", "margin": "0.2rem 0"}),
                    html.P(f"{n_flags} pays" if n_flags != "-" else "-",
                           style={"color": SOFT, "fontSize": "0.82rem", "margin": "0.2rem 0"}),
                ]))

        return html.Div(sections), csv_json


# ── 1. Vessel count by type ─────────────────────────

def _graph_vessel_count_by_type(tys):
    by_type = tys.groupby("vessel_type")["n_vessels"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=by_type.index, y=by_type.values,
                            marker_color=[_rgb(t) for t in by_type.index]))
    _dark_layout(fig, "Vessel count by type (toutes années)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 2. Vessel activity by year ──────────────────────────────────────────────────

def _graph_activity_by_year(tys):
    by_year = tys.groupby("year")["n_points"].sum().sort_index()
    fig = go.Figure(go.Bar(x=by_year.index.astype(str), y=by_year.values, marker_color=ACC))
    _dark_layout(fig, "Vessel activity by year (points AIS)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 3. Seasonal pattern by year ────────────────────────────

def _graph_seasonal_pattern(tys, year):
    ty = tys[tys["year"] == year]
    by_season = ty.groupby("season")["n_points"].sum().reindex(SEASON_ORDER).fillna(0)
    fig = go.Figure(go.Scatter(x=by_season.index, y=by_season.values, mode="lines+markers",
                                line=dict(color=ACC, width=3), marker=dict(size=9)))
    _dark_layout(fig, f"Seasonal pattern — {year}")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 4. Vessel type composition per season ────────────────

def _graph_type_composition_per_season(tys, year):
    ty = tys[tys["year"] == year]
    pivot = ty.pivot_table(index="season", columns="vessel_type", values="n_vessels",
                            aggfunc="sum", fill_value=0).reindex(SEASON_ORDER)
    fig = go.Figure()
    for vtype in pivot.columns:
        fig.add_trace(go.Bar(name=vtype, x=pivot.index, y=pivot[vtype], marker_color=_rgb(vtype)))
    fig.update_layout(barmode="stack")
    _dark_layout(fig, f"Vessel type composition per season — {year}")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── 5. Protected area — vessels detected inside, by year ───────────────────────

def _graph_protected_area_totals(pa_stats):
    years = sorted(pa_stats.keys())
    totals = [pa_stats[y].get("total_vessels_in_zone", 0) for y in years]
    fig = go.Figure(go.Bar(x=years, y=totals, marker_color=ACC))
    _dark_layout(fig, "Total vessels detected in protected area, by year")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _graph_protected_area_by_type(pa_stats):
    years = sorted(pa_stats.keys())
    all_types = sorted({t for y in years for t in pa_stats[y].get("by_type", {})})
    fig = go.Figure()
    for t in all_types:
        vals = [pa_stats[y].get("by_type", {}).get(t, 0) for y in years]
        fig.add_trace(go.Bar(name=t, x=years, y=vals, marker_color=_rgb(t)))
    fig.update_layout(barmode="stack")
    _dark_layout(fig, "Composition by vessel type, by year")
    return dcc.Graph(figure=fig, config={"displayModeBar": False})
