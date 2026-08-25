"""
Aegean Vessel Tracker
=====================
Entry point: overall layout + navigation (dropdown menu top-left).
Each page lives in pages/<name>.py with two functions:
    layout()                -> the page content
    register_callbacks(app) -> the callbacks specific to that page
Run: python app.py
"""

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc

from api_key import get_api_key, save_api_key, has_api_key
from shared import BG, PANEL, BDR, DIM, MAIN, ACC, SOFT

from pages import data as page_data
from pages import map as page_map
from pages import heatmap as page_heatmap
from pages import stats as page_stats
from pages import protected as page_protected
from pages import encounter as page_encounters
from pages import loitering as page_loitering
from pages import report as page_report
from pages import ports as page_ports
from pages import ais_gap as page_ais_gap

# App

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Aegean Vessel Tracker",
    suppress_callback_exceptions=True,
)
server = app.server

PAGES = {
    "data":        {"label": "Data (download)",        "layout": page_data.layout},
    "report":      {"label": "Report",                 "layout": page_report.layout},
    "map":         {"label": "Map & Trajectories",     "layout": page_map.layout},
    "heatmap":     {"label": "Heatmaps",               "layout": page_heatmap.layout},
    "stats":       {"label": "Statistics",             "layout": page_stats.layout},
    "protected":   {"label": "Protected Area",         "layout": page_protected.layout},
    "encounters":  {"label": "Encounters",             "layout": page_encounters.layout},
    "loitering":   {"label": "Loitering",              "layout": page_loitering.layout},
    "ports":       {"label": "Port visits",            "layout": page_ports.layout},
    "ais_gap":     {"label": "AIS Gaps",                "layout": page_ais_gap.layout},
}

DEFAULT_PAGE = "data"
PAGES_NEEDING_KEY = {"data", "ports", "ais_gap"}

MODAL_OVERLAY_STYLE = {
    "display": "flex", "alignItems": "center", "justifyContent": "center",
    "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
    "background": "rgba(0,0,0,0.6)", "zIndex": 9999,
}

# Navigation

def _nav_button(key, label, active):
    return html.Button(
        label,
        id={"type": "nav-link", "key": key},
        n_clicks=0,
        style={
            "border": "none",
            "background": "transparent",
            "color": ACC if active else SOFT,
            "fontWeight": "700" if active else "500",
            "fontSize": "0.9rem",
            "cursor": "pointer",
            "padding": "0.2rem 0.1rem",
            "borderBottom": f"2px solid {ACC}" if active else "2px solid transparent",
            "whiteSpace": "nowrap",
        },
    )


def build_nav(active_key):
    return html.Div([
        html.Span("Aegean Vessel Tracker", style={
            "color": MAIN, "fontWeight": "700", "fontSize": "0.95rem",
            "marginRight": "1.8rem", "whiteSpace": "nowrap", "flexShrink": "0",
        }),
        html.Div(
            [_nav_button(k, v["label"], k == active_key) for k, v in PAGES.items()],
            style={"display": "flex", "alignItems": "center", "gap": "1.2rem",
                   "flexWrap": "wrap"},
        ),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "0.6rem 1.2rem", "background": PANEL,
        "borderBottom": f"2px solid {ACC}",
        "minHeight": "52px", "boxSizing": "border-box",
    })


# Overall layout

app.layout = html.Div([
    dcc.Store(id="store-csv-path", data=None),
    dcc.Store(id="store-df", data=None),
    dcc.Store(id="nav-active", data=DEFAULT_PAGE),
    dcc.Store(id="api-key-present", data=has_api_key()),

    html.Div(
        id="api-key-modal",
        children=html.Div([
            html.H5("GFW API key required",
                    style={"color": MAIN, "marginBottom": "0.6rem"}),
            html.P("This page needs a Global Fishing Watch API key. "
                   "Enter it once — it will be saved locally for next time.",
                   style={"color": DIM, "fontSize": "0.82rem", "marginBottom": "1rem"}),
            dcc.Input(id="api-key-input", type="password",
                      placeholder="Paste your GFW API key...",
                      style={"width": "100%", "padding": "0.6rem", "background": BG,
                             "color": MAIN, "border": f"1px solid {BDR}",
                             "borderRadius": "6px", "fontFamily": "monospace",
                             "marginBottom": "0.8rem"}),
            html.Div(id="api-key-modal-status",
                     style={"fontSize": "0.75rem", "minHeight": "1rem",
                            "marginBottom": "0.8rem"}),
            html.Button("Save key", id="api-key-save", n_clicks=0,
                style={"padding": "0.5rem 1.4rem",
                       "background": f"linear-gradient(135deg,{ACC},#0d4a7a)",
                       "color": "white", "border": "none", "borderRadius": "6px",
                       "cursor": "pointer", "fontWeight": "600"}),
        ], style={"background": PANEL, "padding": "2rem", "borderRadius": "10px",
                   "width": "440px", "maxWidth": "90vw",
                   "border": f"1px solid {BDR}",
                   "boxShadow": "0 10px 40px rgba(0,0,0,0.5)"}),
        style={"display": "none"},
    ),

    html.Div(id="nav-bar-container", children=build_nav(DEFAULT_PAGE)),
    html.Div(id="page-content"),

], style={"margin": 0, "padding": 0, "background": BG, "minHeight": "100vh"})


@app.callback(
    Output("page-content", "children"),
    Output("nav-bar-container", "children"),
    Output("nav-active", "data"),
    Output("api-key-modal", "style"),
    Input({"type": "nav-link", "key": ALL}, "n_clicks"),
    State("nav-active", "data"),
    prevent_initial_call=False,
)
def render_page(_clicks, current):
    trig = ctx.triggered_id
    if isinstance(trig, dict) and trig.get("type") == "nav-link":
        page_key = trig["key"]
    else:
        page_key = current or DEFAULT_PAGE

    page = PAGES.get(page_key, PAGES[DEFAULT_PAGE])

    if page_key in PAGES_NEEDING_KEY and not has_api_key():
        modal_style = MODAL_OVERLAY_STYLE
    else:
        modal_style = {"display": "none"}

    return page["layout"](), build_nav(page_key), page_key, modal_style


@app.callback(
    Output("api-key-modal", "style", allow_duplicate=True),
    Output("api-key-modal-status", "children"),
    Output("api-key-present", "data"),
    Input("api-key-save", "n_clicks"),
    State("api-key-input", "value"),
    prevent_initial_call=True,
)
def _save_key(n, key):
    if not n:
        return no_update, no_update, no_update
    key = (key or "").strip()
    if len(key) < 10:
        return no_update, "Key looks too short.", no_update
    save_api_key(key)
    return {"display": "none"}, "", True


# Register each page's callbacks

page_data.register_callbacks(app)
page_map.register_callbacks(app)
page_heatmap.register_callbacks(app)
page_stats.register_callbacks(app)
page_protected.register_callbacks(app)
page_encounters.register_callbacks(app)
page_loitering.register_callbacks(app)
page_report.register_callbacks(app)
page_ports.register_callbacks(app)
page_ais_gap.register_callbacks(app)
if __name__ == "__main__":
    # threaded=False: the server handles one request at a time. Combined
    # with the 429 retries in gfw.py, this prevents a double-click or a
    # page refresh from sending two GFW report requests in parallel.
    app.run(debug=False, port=8050, threaded=False)