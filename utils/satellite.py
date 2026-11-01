"""
Free satellite imagery via NASA GIBS (Global Imagery Browse Services).
No API key required. Provides near-real-time true-color satellite tiles.

Docs: https://nasa-gibs.github.io/gibs-api-docs/

Note: this gives a real satellite *view* of the location for visual
reference. It is not the same as running our flood-segmentation model
directly on the satellite tile — that automated satellite-to-detection
pipeline is a natural next build step (the same CV model from Media
Analysis could, in principle, run on a cropped GIBS tile) but is not
wired up in this version; the tile is shown for human visual context.
"""

from datetime import date, timedelta

GIBS_TEMPLATE = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "{layer}/default/{date}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
)

TRUE_COLOR_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"


def get_satellite_tile_url(days_ago=1):
    """
    Returns a slippy-map tile URL template usable directly as a folium
    TileLayer. GIBS imagery typically lags 1 day behind real-time.
    """
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return GIBS_TEMPLATE.format(layer=TRUE_COLOR_LAYER, date=d)
