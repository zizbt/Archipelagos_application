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
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from shared import BG, PANEL, BDR, DIM, MAIN, ACC

from pages import data as page_data
from pages import map as page_map
from pages import heatmap as page_heatmap
from pages import stats as page_stats
from pages import protected as page_protected
from pages import encounter as page_encounters
from pages import loitering as page_loitering
from pages import ports as page_ports

# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Aegean Vessel Tracker",
    suppress_callback_exceptions=True,
)
server = app.server

PAGES = {
    "data":        {"label": "Data (download)",       "layout": page_data.layout},
    "map":         {"label": "Map & Trajectories",     "layout": page_map.layout},
    "heatmap":     {"label": "Heatmaps",               "layout": page_heatmap.layout},
    "stats":       {"label": "Statistics",             "layout": page_stats.layout},
    "protected":   {"label": "Protected Area",         "layout": page_protected.layout},
    "encounters":  {"label": "Encounters",             "layout": page_encounters.layout},
    "loitering":   {"label": "Loitering",              "layout": page_loitering.layout},
    "ports":       {"label": "Port visits",            "layout": page_ports.layout},
}

# ── Overall layout ────────────────────────────────────────────────────────────────

NAV_BAR = html.Div([
    html.Span("Aegean Vessel Tracker", style={
        "color": MAIN, "fontWeight": "700", "fontSize": "0.95rem",
        "marginRight": "1.5rem",
    }),
    dcc.Dropdown(
        id="nav-dropdown",
        options=[{"label": v["label"], "value": k} for k, v in PAGES.items()],
        value="data",
        clearable=False,
        searchable=False,
        style={"width": "260px", "color": "#000", "fontWeight": "600"},
    ),
], style={
    "display": "flex", "alignItems": "center",
    "padding": "0.6rem 1.2rem", "background": PANEL,
    "borderBottom": f"2px solid {ACC}",
    "height": "52px", "boxSizing": "border-box",
})

# NOTE: no persistent storage for the API key here on purpose -- this app is
# meant to be shared with multiple users, so each user must enter their own
# GFW API key each time they open the app (nothing saved in the browser).
app.layout = html.Div([
    dcc.Store(id="store-csv-path", data=None),
    dcc.Store(id="store-df", data=None),

    NAV_BAR,
    html.Div(id="page-content"),

], style={"margin": 0, "padding": 0, "background": BG, "minHeight": "100vh"})


@app.callback(
    Output("page-content", "children"),
    Input("nav-dropdown", "value"),
)
def render_page(page_key):
    page = PAGES.get(page_key, PAGES["data"])
    return page["layout"]()


# ── Register each page's callbacks ──────────────────────────────────────────────

page_data.register_callbacks(app)
page_map.register_callbacks(app)
page_heatmap.register_callbacks(app)
page_stats.register_callbacks(app)
page_protected.register_callbacks(app)
page_encounters.register_callbacks(app)
page_loitering.register_callbacks(app)
page_ports.register_callbacks(app)
if __name__ == "__main__":
    # threaded=False: the server handles one request at a time. Combined
    # with the 429 retries in gfw.py, this prevents a double-click or a
    # page refresh from sending two GFW report requests in parallel.
    app.run(debug=False, port=8050, threaded=False)
