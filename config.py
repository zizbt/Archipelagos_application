"""
config.py
=========
Centralized configuration — paths, parameters, visual constants.
"""

from pathlib import Path

# ── Folders ────────────────────────────────────────────────────────────────────

ROOT           = Path(__file__).parent
RAW_DATA       = ROOT / "data" / "raw"
PRECOMPUTED    = ROOT / "data" / "precomputed"
TRAJECTORY_DIR = PRECOMPUTED / "trajectories"
HEATMAP_DIR    = PRECOMPUTED / "heatmaps"
FILTER_DIR     = PRECOMPUTED / "filters"
STAT_DIR       = PRECOMPUTED / "statistics"
GIS_DIR        = ROOT / "data" / "gis"

for folder in [PRECOMPUTED, TRAJECTORY_DIR, HEATMAP_DIR, FILTER_DIR, STAT_DIR, GIS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ── Time parameters ───────────────────────────────────────────────────────────

YEARS = [2023, 2024, 2025, 2026]

MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

# Fixed order used everywhere (preprocess + app) for the 4-card display
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}

VESSEL_TYPES = [
    "PASSENGER", "OTHER", "CARGO", "FISHING", "BUNKER",
    "CARRIER", "SEISMIC_VESSEL", "GEAR", "DISCREPANCY", "NA",
]
ALL_TYPES    = "ALL"

MAX_HEATMAP_POINTS = 1_000_000
MAX_SCATTER_POINTS =  60_000

# ── Maritime zones ─────────────────────────────────────────────────────────────

ZONES = {
    "greece_territorial": {
        "shp":  "ZEE_and_territorial_water/Greece_6_nautic_miles_inside_EEZ/Greece_6_nautic_miles_inside_EEZ.shp",
        "fill_color": [30, 144, 255, 35],
        "line_color": [30, 144, 255, 230],
        "line_width": 160,   # thicker: narrow coastal strip, drawn bold so it stands out
        "label": "Greece territorial waters (6nm)",
    },
    "greece_eez": {
        "shp":  "ZEE_and_territorial_water/Greece_EEZ/Greece_EEZ.shp",
        "fill_color": [0, 206, 209, 15],
        "line_color": [0, 206, 209, 200],
        "line_width": 70,    # thinner: large outer boundary
        "label": "Greece EEZ",
    },
    "turkey_territorial": {
        "shp":  "ZEE_and_territorial_water/Turkey_6_nautic_miles_inside_EEZ/Turkey_6_nautic_miles_inside_EEZ.shp",
        "fill_color": [220, 50, 50, 35],
        "line_color": [220, 50, 50, 230],
        "line_width": 160,
        "label": "Turkey territorial waters (6nm)",
    },
    "turkey_eez": {
        "shp":  "ZEE_and_territorial_water/Turkey_EEZ/Turkey_EEZ.shp",
        "fill_color": [255, 140, 0, 15],
        "line_color": [255, 140, 0, 200],
        "line_width": 70,
        "label": "Turkey EEZ",
    },
    "wdpa": {
        "shp":  "WDPA/WDPA.shp",
        "fill_color": [0, 200, 100, 40],
        "line_color": [0, 200, 100, 210],
        "line_width": 100,
        "label": "Protected areas (WDPA)",
    },
    "fourni_protected": {
        "shp":  None,  # extracted from WDPA.shp -> data/gis/fourni_protected.geojson
        "fill_color": [147, 112, 219, 60],
        "line_color": [147, 112, 219, 235],
        "line_width": 130,
        # SPA (bird protection zone under the EU Birds Directive)
        "label": "Fourni protected area",
    },
}

# Centre approximatif de l'archipel de Fourni (pour zoomer la carte dessus)
FOURNI_CENTER = {"lat": 37.557, "lon": 26.472}

# ── Colors per vessel type ──────────────────────────────────────────────────────

TYPE_COLORS = {
    "PASSENGER":      [ 50, 205,  50, 210],   # green (was passenger green)
    "OTHER":          [160, 160, 160, 210],   # grey
    "CARGO":          [ 30, 144, 255, 210],   # blue
    "FISHING":        [255, 140,   0, 210],   # orange (includes trawlers)
    "BUNKER":         [220,  20,  60, 210],   # crimson (was tanker red -- fuel supply vessel)
    "CARRIER":        [186,  85, 211, 210],   # purple (was pleasure)
    "SEISMIC_VESSEL": [ 64, 224, 208, 210],   # turquoise
    "GEAR":           [218, 165,  32, 210],   # goldenrod
    "DISCREPANCY":    [100, 100, 100, 150],   # dark grey
    "NA":             [100, 100, 100, 120],   # dark grey, more transparent
}
DEFAULT_COLOR = [200, 200, 200, 150]

# ── Flags ──────────────────────────────────────────────────────────────────────

FLAG_NAMES = {
    "FRA": "France", "ATF": "Kerguelen Islands", "RIF": "French International Register",
    "TAH": "Tahiti", "NLD": "Netherlands", "DEU": "Germany", "ITA": "Italy",
    "SCO": "Scotland", "IOM": "Isle of Man", "GBR": "United Kingdom", "IRL": "Ireland",
    "DNK": "Denmark", "DIS": "Danish International Register", "GRC": "Greece",
    "PMD": "Madeira", "PRT": "Portugal", "ESP": "Spain", "CNI": "Canary Islands",
    "JEY": "Jersey", "GGY": "Guernsey", "MCO": "Monaco", "BEL": "Belgium",
    "LUX": "Luxembourg", "ISL": "Iceland", "NOR": "Norway", "NIS": "Norwegian Int. Register",
    "SWE": "Sweden", "FIN": "Finland", "AUT": "Austria", "CHE": "Switzerland",
    "FRO": "Faroe Islands", "GIB": "Gibraltar", "MLT": "Malta", "SMR": "San Marino",
    "TUR": "Turkey", "EST": "Estonia", "LVA": "Latvia", "LTU": "Lithuania",
    "POL": "Poland", "CZE": "Czech Republic", "SVK": "Slovakia", "HUN": "Hungary",
    "ROM": "Romania", "BGR": "Bulgaria", "ALB": "Albania", "UKR": "Ukraine",
    "BLR": "Belarus", "MDA": "Moldova", "RUS": "Russia", "GEO": "Georgia",
    "ARM": "Armenia", "AZE": "Azerbaijan", "KAZ": "Kazakhstan", "TKM": "Turkmenistan",
    "KGZ": "Kyrgyzstan", "SVN": "Slovenia", "HRV": "Croatia", "BIH": "Bosnia and Herzegovina",
    "MON": "Montenegro", "MAR": "Morocco", "DZA": "Algeria", "TUN": "Tunisia",
    "LBY": "Libya", "EGY": "Egypt", "SDN": "Sudan", "MRT": "Mauritania",
    "LBR": "Liberia", "CIV": "Ivory Coast", "GHA": "Ghana", "TGO": "Togo",
    "NGA": "Nigeria", "CMR": "Cameroon", "GAB": "Gabon", "AGO": "Angola",
    "KEN": "Kenya", "TZA": "Tanzania", "SYC": "Seychelles", "MOZ": "Mozambique",
    "MDG": "Madagascar", "MUS": "Mauritius", "ZAF": "South Africa", "NAM": "Namibia",
    "USA": "United States", "CAN": "Canada", "MEX": "Mexico", "BMU": "Bermuda",
    "PAN": "Panama", "CUB": "Cuba", "KNA": "Saint Kitts and Nevis", "HTI": "Haiti",
    "BHS": "Bahamas", "ATG": "Antigua and Barbuda", "CYM": "Cayman Islands",
    "JAM": "Jamaica", "BRB": "Barbados", "TTO": "Trinidad and Tobago",
    "ABW": "Aruba", "CUW": "Curaçao", "COL": "Colombia", "VEN": "Venezuela",
    "BRA": "Brazil", "CHL": "Chile", "ARG": "Argentina", "CYP": "Cyprus",
    "LBN": "Lebanon", "SYR": "Syria", "IRQ": "Iraq", "IRN": "Iran",
    "ISR": "Israel", "JOR": "Jordan", "SAU": "Saudi Arabia", "KWT": "Kuwait",
    "BHR": "Bahrain", "QAT": "Qatar", "ARE": "United Arab Emirates", "OMN": "Oman",
    "YEM": "Yemen", "PAK": "Pakistan", "IND": "India", "BGD": "Bangladesh",
    "LKA": "Sri Lanka", "MMR": "Myanmar", "THA": "Thailand", "VNM": "Vietnam",
    "IDN": "Indonesia", "MYS": "Malaysia", "SGP": "Singapore", "PHL": "Philippines",
    "CHN": "China", "KOR": "South Korea", "JPN": "Japan", "TWN": "Taiwan",
    "HKG": "Hong Kong", "AUS": "Australia", "NZL": "New Zealand",
    "COK": "Cook Islands", "MHL": "Marshall Islands", "PLW": "Palau",
    "UKN": "Unknown flag", "MDV": "Maldives", "ABD": "Abu Dhabi", "DUB": "Dubai",
}
