"""
Nearest safe-location finder using OpenStreetMap's free Overpass API.

Coverage in smaller towns/rural areas is often sparse for narrowly-tagged
categories (amenity=shelter is rare globally). To be actually useful in
places like rural Assam, this searches a broader, realistic set of
facility types that serve as real emergency shelters in practice —
schools, places of worship, community halls, government buildings — not
just the narrow "official shelter" tag. It also auto-expands the search
radius if nothing is found nearby, rather than silently returning empty.
"""

import math
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# (osm_key, osm_value, human label)
SAFE_POINT_TAGS = [
    ("amenity", "shelter", "Shelter"),
    ("emergency", "assembly_point", "Assembly point"),
    ("amenity", "community_centre", "Community centre"),
    ("amenity", "hospital", "Hospital"),
    ("amenity", "clinic", "Clinic"),
    ("amenity", "school", "School"),
    ("amenity", "college", "College"),
    ("amenity", "place_of_worship", "Place of worship"),
    ("amenity", "townhall", "Government building"),
    ("amenity", "police", "Police station"),
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _build_query(lat, lon, radius_m):
    clauses = []
    for key, val, _ in SAFE_POINT_TAGS:
        clauses.append(f'node["{key}"="{val}"](around:{radius_m},{lat},{lon});')
        clauses.append(f'way["{key}"="{val}"](around:{radius_m},{lat},{lon});')
    body = "\n".join(clauses)
    return f"[out:json][timeout:20];\n(\n{body}\n);\nout center;"


def find_nearby_safe_points(lat, lon, radius_m=8000, limit=6):
    """
    Returns a dict: {"results": [...], "status": "ok" | "empty" | "error", "radius_used_km": float}
    - "ok": found results
    - "empty": query succeeded but genuinely nothing tagged nearby, even after expanding radius
    - "error": the Overpass request itself failed (network/timeout/rate-limit)
    """
    radii_to_try = [radius_m, radius_m * 2.5, radius_m * 5]  # auto-expand: 8km -> 20km -> 40km

    last_error = False
    for r_m in radii_to_try:
        query = _build_query(lat, lon, r_m)
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=25)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
        except Exception:
            last_error = True
            continue

        last_error = False
        results = []
        seen = set()
        for el in elements:
            el_lat = el.get("lat") or el.get("center", {}).get("lat")
            el_lon = el.get("lon") or el.get("center", {}).get("lon")
            if el_lat is None or el_lon is None:
                continue
            tags = el.get("tags", {})
            name = tags.get("name", "Unnamed location")
            kind = tags.get("amenity") or tags.get("emergency") or "facility"
            key = (name, round(el_lat, 4), round(el_lon, 4))
            if key in seen:
                continue
            seen.add(key)
            dist = _haversine_km(lat, lon, el_lat, el_lon)
            results.append({"name": name, "type": kind, "lat": el_lat, "lon": el_lon, "distance_km": round(dist, 2)})

        if results:
            results.sort(key=lambda x: x["distance_km"])
            return {"results": results[:limit], "status": "ok", "radius_used_km": round(r_m / 1000, 1)}

    if last_error:
        return {"results": [], "status": "error", "radius_used_km": None}
    return {"results": [], "status": "empty", "radius_used_km": round(radii_to_try[-1] / 1000, 1)}
