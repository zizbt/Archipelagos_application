"""
pages/protected.py
===================
"Protected Area" page -- zoom on the Fourni island (SPA protected zone
under the EU Birds Directive, including Thymaina and Agios Minas + the
marine area, extracted from data/gis/wdpa.geojson).

For a given date range, detects which vessels passed inside the zone's
polygon and shows, per vessel: name, flag, type, number of hours
detected inside, first/last detection. CSV export included.

v2 -- SINGLE data source now: an imported CSV, exactly like
pages/heatmap.py (drag & drop + server-side cache). No more precomputed
trajectories (load_trajectories_range) and no more "choose an
already-downloaded CSV from disk" dropdown -- everything (date range,
vessel types, gear types) is derived live from the file the user drops,
which also means gear_type is always available here (it wasn't on the
precomputed source before).

The date range is not locked to a single calendar year (a selection can
freely span across two years, e.g. Dec 2024 -> Jan 2025), same as
before -- its bounds are simply taken from the imported CSV's own
"date" column instead of a fixed global range.
"""

import base64
import io
import json
from datetime import date

import dash
import pandas as pd
import pydeck as pdk
import dash_deck
from dash import dcc, html, Input, Output, State, dash_table
from shapely.geometry import shape, Point
from shapely.prepared import prep

from shared import BG, PANEL, BDR, DIM, MAIN, SOFT, ACC, MAPBOX_KEY, lbl
from config import VESSEL_TYPES, TYPE_COLORS, DEFAULT_COLOR, FLAG_NAMES, FOURNI_CENTER
from loader import load_geojson

ZONE_KEY = "fourni_protected"
ZONE_FILL = [255, 215, 0, 60]
ZONE_LINE = [255, 215, 0, 230]

PLACEHOLDER_STYLE = {"color": DIM, "padding": "2rem", "fontSize": "0.85rem", "fontStyle": "italic"}

# Cache serveur du CSV importé, comme _CSV_CACHE dans pages/heatmap.py --
# évite l'aller-retour du dataframe par le navigateur.
_CSV_CACHE = {"df": None, "filename": None}


def _load_zone():
    """Load the Fourni protected area polygon (once). This is the zone's
    own boundary definition, not vessel data, so it stays precomputed
    regardless of where the vessel positions come from."""
    gj = load_geojson(ZONE_KEY)
    if not gj or not gj.get("features"):
        return None, None
    feat = gj["features"][0]
    poly = shape(feat["geometry"])
    return poly, gj


_ZONE_POLYGON, _ZONE_GEOJSON = _load_zone()


def _parse_uploaded_csv(contents, filename):
    """Identique à heatmap._parse_uploaded_csv (dcc.Upload -> DataFrame)."""
    if contents is None:
        return None
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith((".tsv", ".txt")):
        return pd.read_csv(io.BytesIO(decoded), sep=None, engine="python")
    return pd.read_csv(io.BytesIO(decoded))


def _filter_points_in_zone(df):
    """Filter positions falling inside the zone's polygon. Fast bounding-box
    pre-filter first, then an exact test only on the already-close subset --
    avoids testing every single point in the full dataset one by one (slow)."""
    if df is None or df.empty or _ZONE_POLYGON is None:
        return pd.DataFrame()
    if "lat" not in df.columns or "lon" not in df.columns:
        return pd.DataFrame()

    minx, miny, maxx, maxy = _ZONE_POLYGON.bounds
    bbox_mask = (df["lon"] >= minx) & (df["lon"] <= maxx) & \
                (df["lat"] >= miny) & (df["lat"] <= maxy)
    sub = df[bbox_mask]
    if sub.empty:
        return sub

    prepared = prep(_ZONE_POLYGON)
    inside = [prepared.contains(Point(lon, lat)) for lon, lat in zip(sub["lon"], sub["lat"])]
    return sub[inside]


def _aggregate_by_vessel(df):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["vessel_id", "date"])

    agg_dict = {
        "n_positions": ("date", "count"),
        "first_seen": ("date", "min"),
        "last_seen": ("date", "max"),
    }
    for col in ["ship_name", "flag", "vessel_type", "gear_type"]:
        if col in df.columns:
            agg_dict[col] = (col, "first")

    result = df.groupby("vessel_id").agg(**agg_dict).reset_index()
    # HOURLY resolution -> 1 position = ~1h detected inside the zone
    result["hours_detected"] = result["n_positions"]
    result["first_seen"] = result["first_seen"].dt.strftime("%Y-%m-%d %H:%M")
    result["last_seen"] = result["last_seen"].dt.strftime("%Y-%m-%d %H:%M")
    if "flag" in result.columns:
        result["flag_label"] = result["flag"].map(lambda f: f"{FLAG_NAMES.get(f, f)} ({f})")

    # Per-visit breakdown: each distinct crossing of the zone, with its
    # exact start timestamp, end timestamp, and duration. Consecutive
    # positions belong to the same visit; a gap larger than
    # VISIT_GAP_THRESHOLD means the vessel left and came back later, so a
    # new visit starts. Formatted as a single human-readable string per
    # vessel, e.g. "2026-01-01 12:03 -> 2026-01-02 13:05 (1h02), 2026-02-20 09:10 -> 2026-02-20 11:40 (2h30)".
    # NOTE: comma-separated (not semicolon) -- semicolons get read as a
    # column delimiter by some Excel locales (e.g. French), which was
    # spilling multiple visits into columns B, C, etc. instead of staying
    # in a single cell. Using a comma is safe here: pandas.to_csv quotes
    # any field that contains the delimiter, so this whole string stays
    # wrapped in double quotes as one CSV field.
    VISIT_GAP_THRESHOLD = pd.Timedelta(hours=3)

    def _format_duration(dur):
        total_minutes = int(dur.total_seconds() // 60)
        h, m = divmod(total_minutes, 60)
        return f"{h}h{m:02d}"

    def _format_visits(group):
        group = group.sort_values("date")
        gap = group["date"].diff()
        visit_id = (gap > VISIT_GAP_THRESHOLD).cumsum()
        parts = []
        for _, visit in group.groupby(visit_id):
            v_start = visit["date"].min()
            v_end = visit["date"].max()
            duration = _format_duration(v_end - v_start)
            parts.append(
                f"{v_start.strftime('%Y-%m-%d %H:%M')} -> "
                f"{v_end.strftime('%Y-%m-%d %H:%M')} ({duration})"
            )
        return ", ".join(parts)

    breakdown = df.groupby("vessel_id").apply(_format_visits, include_groups=False)
    result["dates_and_hours"] = result["vessel_id"].map(breakdown)

    return result.sort_values("hours_detected", ascending=False)


def _upload_zone():
    return dcc.Upload(
        id="prot-csv-upload",
        children=html.Div([
            "Drag a CSV here, or ",
            html.A("browse", style={"color": ACC, "textDecoration": "underline"}),
        ]),
        style={
            "width": "100%", "padding": "1rem 0.5rem",
            "textAlign": "center", "cursor": "pointer",
            "border": f"1px dashed {BDR}", "borderRadius": "6px",
            "color": SOFT, "fontSize": "0.75rem",
            "marginBottom": "0.5rem",
        },
        multiple=False,
    )


def layout():
    return html.Div([
        dcc.Store(id="prot-store-agg", data=None),
        dcc.Store(id="prot-store-csv-loaded", data=None),
        dcc.Download(id="prot-download-csv"),

        # ── Sidebar ──────────────────────────────────────────────────────────
        html.Div([
            html.H6("Protected zone", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.6rem"}),
            dcc.Dropdown(id="prot-zone", value="fourni",
                options=[{"label": "Fourni (Fournoi Korseon)", "value": "fourni"}],
                clearable=False, style={"color": "#000", "marginBottom": "1rem"}),

            html.Div([
                html.H6("Import a CSV", style={"color": MAIN, "fontSize": "0.82rem", "marginBottom": "0.4rem"}),
                _upload_zone(),
                html.Div("No file selected", id="prot-csv-filename",
                          style={"fontSize": "0.72rem", "color": DIM,
                                 "fontStyle": "italic", "marginBottom": "0.6rem"}),
            ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
                       "borderBottom": f"1px solid {BDR}"}),

            html.Div([
                lbl("Jump to a year (optional)"),
                dcc.Dropdown(id="prot-year", value=None, clearable=True,
                    options=[], placeholder="Import a CSV first...",
                    style={"color": "#000", "marginBottom": "0.6rem"}),
                lbl("Start date"),
                dcc.DatePickerSingle(id="prot-start", date=None,
                    display_format="YYYY-MM-DD",
                    style={"marginBottom": "0.6rem"}),
                lbl("End date"),
                dcc.DatePickerSingle(id="prot-end", date=None,
                    display_format="YYYY-MM-DD",
                    style={"marginBottom": "0.6rem"}),
                html.P("The range can span across two years (e.g. Dec 2024 -> Jan 2025).",
                       style={"fontSize": "0.68rem", "color": DIM, "fontStyle": "italic",
                              "marginBottom": "0.6rem"}),
                lbl("Vessel types (tick to show)"),
                html.Button(
                    "Deselect all",
                    id="prot-type-select-all",
                    n_clicks=0,
                    style={"fontSize": "0.7rem", "color": SOFT, "background": "none",
                           "border": f"1px solid {BDR}", "borderRadius": "4px",
                           "padding": "3px 8px", "marginBottom": "6px", "cursor": "pointer"},
                ),
                dcc.Checklist(
                    id="prot-type-filter",
                    options=[
                        {"label": html.Span([
                            html.Span(style={
                                "display": "inline-block", "width": "11px", "height": "11px",
                                "borderRadius": "50%", "marginRight": "7px",
                                "backgroundColor": "rgba({},{},{},{})".format(
                                    *(TYPE_COLORS.get(t, DEFAULT_COLOR)[:3]),
                                    (TYPE_COLORS.get(t, DEFAULT_COLOR)[3] / 255)
                                    if len(TYPE_COLORS.get(t, DEFAULT_COLOR)) > 3 else 1),
                                "verticalAlign": "middle"}),
                            html.Span(t.capitalize(),
                                    style={"fontSize": "0.72rem", "color": SOFT,
                                            "verticalAlign": "middle"})],
                            style={"display": "inline-flex", "alignItems": "center"}),
                        "value": t}
                        for t in VESSEL_TYPES
                    ],
                    value=list(VESSEL_TYPES),
                    labelStyle={"display": "flex", "alignItems": "center",
                                "marginBottom": "3px", "cursor": "pointer"},
                    inputStyle={"marginRight": "6px"},
                    style={"marginBottom": "0.8rem"},
                ),

                # Le filtre gear_type dépend du CSV importé (comme le
                # filtre gear de heatmap.py) -- il n'a de sens que si la
                # colonne existe, donc il est masqué par défaut.
                html.Div(id="prot-gear-wrap", style={"display": "none"}, children=[
                    lbl("Gear type filter (optional)"),
                    dcc.Dropdown(id="prot-gear-filter",
                        options=[], value=[], multi=True,
                        placeholder="All gear types (e.g. Trawlers)...",
                        style={"color": "#000", "marginBottom": "0.6rem"}),
                ]),

                html.Button("Analyze", id="prot-btn-run", n_clicks=0,
                    style={"width": "100%", "padding": "0.5rem",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "border": "none",
                           "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600"}),
                html.Div(id="prot-csv-status", style={"fontSize": "0.72rem", "color": SOFT, "marginTop": "0.4rem"}),
            ], style={"marginBottom": "1.2rem", "paddingBottom": "1.2rem",
                       "borderBottom": f"1px solid {BDR}"}),

            html.Div(id="prot-summary", style={"fontSize": "0.75rem", "color": SOFT}),

        ], style={"width": "280px", "minWidth": "280px", "padding": "1rem",
                   "background": BG, "borderRight": f"1px solid {BDR}",
                   "height": "calc(100vh - 52px)", "overflowY": "auto", "flexShrink": "0"}),

        # ── Map (Fourni zoom), full height + Export button top-right ────────
        html.Div([
            html.Div(
                html.Button("Export CSV", id="prot-btn-export", n_clicks=0,
                    style={"border": "none",
                           "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                           "color": "white", "cursor": "pointer", "fontSize": "0.75rem",
                           "fontWeight": "600", "padding": "0.3rem 1rem", "borderRadius": "5px"}),
                style={"padding": "0.3rem 0.6rem", "background": BG,
                       "borderBottom": f"1px solid {BDR}", "flexShrink": "0",
                       "display": "flex", "justifyContent": "flex-end"},
            ),
            html.Div([
                dcc.Loading(
                    type="circle", color=ACC,
                    parent_style={"height": "100%", "width": "100%"},
                    style={"height": "100%", "width": "100%"},
                    children=html.Div(id="prot-map-container",
                        children=html.P("Import a CSV, then click \"Analyze\".", style=PLACEHOLDER_STYLE),
                        style={"height": "100%", "width": "100%"}),
                ),
            ], style={"flex": "1", "minHeight": 0}),

            # tableau conservé mais caché (les callbacks écrivent encore dedans)
            html.Div(id="prot-vessel-table", style={"display": "none"}),
        ], style={"flex": "1", "minHeight": 0, "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "calc(100vh - 52px)"})


def _build_zone_map(df_inside):
    layers = []
    if _ZONE_GEOJSON:
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=_ZONE_GEOJSON, stroked=True, filled=True,
            get_fill_color=ZONE_FILL, get_line_color=ZONE_LINE,
            line_width_min_pixels=2, pickable=False,
        ))

    if df_inside is not None and not df_inside.empty:
        plot = df_inside.copy()
        if "vessel_type" in plot.columns:
            plot["color"] = plot["vessel_type"].astype(str).str.upper().map(
                lambda t: TYPE_COLORS.get(t, DEFAULT_COLOR))
        else:
            plot["color"] = [DEFAULT_COLOR] * len(plot)
        plot["t_ship"] = plot["ship_name"].astype(str) if "ship_name" in plot.columns else "?"
        plot["t_flag"] = (plot["flag"].map(lambda f: FLAG_NAMES.get(f, f)).astype(str)
                          if "flag" in plot.columns else "?")
        plot["t_type"] = plot["vessel_type"].astype(str) if "vessel_type" in plot.columns else "?"
        plot["t_gear"] = (plot["gear_type"].astype(str).str.replace("_", " ").str.title()
                          if "gear_type" in plot.columns else "-")
        plot["t_date"] = plot["date"].astype(str) if "date" in plot.columns else "-"

        layers.append(pdk.Layer(
            "ScatterplotLayer", data=plot,
            get_position=["lon", "lat"], get_fill_color="color",
            get_radius=150, radius_min_pixels=3, radius_max_pixels=9,
            pickable=True, auto_highlight=True,
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=FOURNI_CENTER["lat"], longitude=FOURNI_CENTER["lon"],
            zoom=10.5, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "html": (
                "<div style='font-size:12px'>"
                "<b>{t_ship}</b><br/>"
                "Flag: {t_flag}<br/>"
                "Type: {t_type}<br/>"
                "Gear: {t_gear}<br/>"
                "Date: {t_date}"
                "</div>"
            ),
            "style": {
                "backgroundColor": "#0d1b2a",
                "color": "#e0e6ed",
                "border": "1px solid #1b3a5b",
                "borderRadius": "6px",
                "padding": "8px 10px",
            },
        },
    )
    deck_json = json.loads(deck.to_json())
    return dash_deck.DeckGL(
        data=deck_json, mapboxKey=MAPBOX_KEY,
        tooltip={
            "html": (
                "<div style='font-size:12px'>"
                "<b>{t_ship}</b><br/>"
                "Flag: {t_flag}<br/>"
                "Type: {t_type}<br/>"
                "Gear: {t_gear}<br/>"
                "Date: {t_date}"
                "</div>"
            ),
            "style": {
                "backgroundColor": "#0d1b2a",
                "color": "#e0e6ed",
                "border": "1px solid #1b3a5b",
                "borderRadius": "6px",
                "padding": "8px 10px",
            },
        },
        style={"width": "100%", "height": "100%"})


def _vessel_table(agg):
    if agg.empty:
        return html.P("No vessel detected in the zone for this period.",
                       style={"color": DIM, "fontStyle": "italic"})
    cols = ["vessel_id"]
    for c in ["ship_name", "flag_label", "vessel_type", "gear_type", "hours_detected",
              "first_seen", "last_seen", "dates_and_hours"]:
        if c in agg.columns:
            cols.append(c)
    display_names = {
        "vessel_id": "Vessel Id", "ship_name": "Ship Name", "flag_label": "Flag",
        "vessel_type": "Vessel Type", "gear_type": "Gear Type",
        "hours_detected": "Hours Detected", "first_seen": "First Seen", "last_seen": "Last Seen",
        "dates_and_hours": "Crossings (start -> end, duration)",
    }
    return dash_table.DataTable(
        data=agg[cols].to_dict("records"),
        columns=[{"name": display_names.get(c, c), "id": c} for c in cols],
        page_size=10, export_format="csv", export_headers="display",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": PANEL, "color": MAIN, "border": f"1px solid {BDR}",
                    "fontSize": "0.75rem", "padding": "4px 8px"},
        style_header={"backgroundColor": BG, "color": DIM, "fontWeight": "600"},
    )


def _summary(agg, note=None):
    if agg.empty:
        return html.P("No data.", style={"color": DIM})
    n_vessels = len(agg)
    total_hours = int(agg["hours_detected"].sum())
    children = [
        html.P(f"{n_vessels:,} vessels detected in the zone", style={"color": SOFT, "margin": "0.2rem 0"}),
        html.P(f"{total_hours:,} cumulative hours detected", style={"color": SOFT, "margin": "0.2rem 0"}),
    ]
    if "vessel_type" in agg.columns:
        by_type = agg["vessel_type"].value_counts()
        children.append(html.Div([
            html.P("By type:", style={"color": DIM, "margin": "0.6rem 0 0.2rem 0", "fontWeight": "600"}),
            *[html.P(f"  {t}: {n}", style={"color": SOFT, "margin": "0.1rem 0", "fontSize": "0.72rem"})
              for t, n in by_type.items()],
        ]))
    if "gear_type" in agg.columns:
        n_trawlers = (agg["gear_type"].astype(str).str.upper() == "TRAWLERS").sum()
        children.append(html.P(f"Of which Trawlers (gear type): {n_trawlers}",
                                style={"color": "#ffd700", "marginTop": "0.5rem", "fontWeight": "600"}))
    if note:
        children.append(html.P(note, style={"color": DIM, "fontSize": "0.68rem", "fontStyle": "italic", "marginTop": "0.5rem"}))
    return html.Div(children)


def register_callbacks(app):

    # Parse le CSV dès qu'il est déposé -- peuple le sélecteur d'années,
    # les bornes des date-pickers (prises dans le fichier lui-même, plus
    # de plage globale fixe) et l'état du filtre gear_type. L'analyse
    # elle-même n'a lieu qu'au clic sur "Analyze", plus bas.
    @app.callback(
        Output("prot-csv-filename", "children"),
        Output("prot-year", "options"),
        Output("prot-year", "value"),
        Output("prot-start", "date"),
        Output("prot-start", "min_date_allowed"),
        Output("prot-start", "max_date_allowed"),
        Output("prot-end", "date"),
        Output("prot-end", "min_date_allowed"),
        Output("prot-end", "max_date_allowed"),
        Output("prot-gear-wrap", "style"),
        Output("prot-gear-filter", "options"),
        Output("prot-store-csv-loaded", "data"),
        Input("prot-csv-upload", "contents"),
        State("prot-csv-upload", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv_uploaded(contents, filename):
        if not contents:
            raise dash.exceptions.PreventUpdate
        try:
            df = _parse_uploaded_csv(contents, filename)
        except Exception as e:
            _CSV_CACHE["df"] = None
            return (f"Error: {e}", [], None, None, None, None, None, None, None,
                    {"display": "none"}, [], None)

        if "date" not in df.columns:
            _CSV_CACHE["df"] = None
            return (f'"{filename}" has no "date" column.', [], None, None, None, None,
                    None, None, None, {"display": "none"}, [], None)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        _CSV_CACHE["df"] = df
        _CSV_CACHE["filename"] = filename

        min_date = df["date"].min()
        max_date = df["date"].max()
        min_d = min_date.date() if pd.notna(min_date) else None
        max_d = max_date.date() if pd.notna(max_date) else None

        years = sorted(df["date"].dt.year.dropna().astype(int).unique(), reverse=True)
        year_opts = [{"label": str(y), "value": y} for y in years]

        gear_style = {"display": "block"} if "gear_type" in df.columns else {"display": "none"}
        gear_opts = []
        if "gear_type" in df.columns:
            gear_opts = [{"label": g.replace("_", " ").title(), "value": g}
                         for g in sorted(df["gear_type"].dropna().astype(str).str.upper().unique())]

        return (f"Loaded: {len(df):,} rows from \"{filename}\"", year_opts, None,
                min_d, min_d, max_d, max_d, min_d, max_d, gear_style, gear_opts, "loaded")

    # "Deselect all / Select all" button: toggles the whole type list and
    # flips its own label to match the resulting state.
    @app.callback(
        Output("prot-type-filter", "value"),
        Output("prot-type-select-all", "children"),
        Input("prot-type-select-all", "n_clicks"),
        State("prot-type-filter", "value"),
        prevent_initial_call=True,
    )
    def _toggle_all_types(n, selected):
        if selected:
            return [], "Select all"
        return list(VESSEL_TYPES), "Deselect all"

    # "Jump to a year" is just a convenience: it fills in Jan 1 -> Dec 31
    # of the chosen year, but doesn't restrict the pickers -- the user can
    # still edit either date afterwards, including across a year boundary.
    @app.callback(
        Output("prot-start", "date", allow_duplicate=True),
        Output("prot-end", "date", allow_duplicate=True),
        Input("prot-year", "value"),
        prevent_initial_call=True,
    )
    def jump_to_year(year):
        if not year:
            raise dash.exceptions.PreventUpdate
        return date(year, 1, 1), date(year, 12, 31)

    @app.callback(
        Output("prot-map-container", "children"),
        Output("prot-vessel-table", "children"),
        Output("prot-summary", "children"),
        Output("prot-csv-status", "children"),
        Output("prot-store-agg", "data"),
        Input("prot-btn-run", "n_clicks"),
        State("prot-start", "date"),
        State("prot-end", "date"),
        State("prot-gear-filter", "value"),
        State("prot-type-filter", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n, start, end, gear_filter, type_filter):
        if not n:
            raise dash.exceptions.PreventUpdate

        if _ZONE_POLYGON is None:
            msg = html.P("Zone not found: data/gis/fourni_protected.geojson is missing.",
                          style={"color": "#ff6b6b"})
            return _build_zone_map(pd.DataFrame()), "", msg, "", None

        df = _CSV_CACHE.get("df")
        if df is None or df.empty:
            return (html.P("Import a CSV first.", style=PLACEHOLDER_STYLE), "", "",
                    "Import a CSV first.", None)

        sub = df
        if start and end:
            s, e = pd.to_datetime(start), pd.to_datetime(end)
            if s > e:
                s, e = e, s
            sub = sub[(sub["date"] >= s) & (sub["date"] <= e + pd.Timedelta(days=1))]

        df_inside = _filter_points_in_zone(sub)
        if type_filter is not None and "vessel_type" in df_inside.columns:
            df_inside = df_inside[df_inside["vessel_type"].astype(str).str.upper().isin(
                [t.upper() for t in type_filter])]
        if gear_filter and "gear_type" in df_inside.columns:
            df_inside = df_inside[df_inside["gear_type"].astype(str).str.upper().isin(
                [g.upper() for g in gear_filter])]

        agg = _aggregate_by_vessel(df_inside)
        filename = _CSV_CACHE.get("filename") or "CSV"
        note = "Analysis based on the imported CSV." + (
            " (gear_type available)" if "gear_type" in df.columns else "")
        status = f"\"{filename}\": {len(sub):,} rows in range -> {len(df_inside):,} positions in the zone"
        return (_build_zone_map(df_inside), _vessel_table(agg), _summary(agg, note), status,
                (agg.to_dict("records") if not agg.empty else None))

    @app.callback(
        Output("prot-download-csv", "data"),
        Input("prot-btn-export", "n_clicks"),
        State("prot-store-agg", "data"),
        prevent_initial_call=True,
    )
    def _export(n, store):
        if not n or not store:
            raise dash.exceptions.PreventUpdate
        out = pd.DataFrame(store)
        if "gear_type" not in out.columns:
            out["gear_type"] = ""
        cols = list(out.columns)
        if "gear_type" in cols and "vessel_type" in cols:
            cols.remove("gear_type")
            cols.insert(cols.index("vessel_type") + 1, "gear_type")
            out = out[cols]
        # sep=";" -- French-locale Excel opens .csv files using semicolon as
        # the column delimiter by default. With the default comma delimiter,
        # Excel-FR doesn't split the file into columns at all (everything
        # lands in column A). Using ";" here matches that default, so every
        # real column (vessel_id, ship_name, dates_and_hours...) splits
        # correctly. The commas inside dates_and_hours (separating multiple
        # visits) stay untouched since they're no longer the delimiter.
        return dcc.send_data_frame(out.to_csv, "protected_area_vessels.csv", index=False, sep=";")