"""
Official disaster alerts via the ReliefWeb API (UN OCHA).

ReliefWeb API v1 has been decommissioned, and since 1 Nov 2025 every request
to v2 must carry a *pre-approved* `appname`. Request one (free) at:
    https://apidoc.reliefweb.int/parameters#appname
then put it in your .env file as RELIEFWEB_APPNAME.

If no appname is configured, or ReliefWeb rejects it, this module returns an
empty list and the rest of the app keeps working (the risk score simply won't
get the "active official alert" floor).

IMD/NDMA/CWC do not currently publish open developer APIs; ReliefWeb is the
most reliable free structured substitute.
"""

import os
import requests
from dotenv import load_doatenv

try:  # python-dotenv is optional: env vars set in the shell work without it
    load_dotenv()
except ImportError:
    pass

RELIEFWEB_URL = "https://api.reliefweb.int/v2/disasters"


def get_flood_alerts(country="India", limit=5):
    appname = os.getenv("RELIEFWEB_APPNAME", "").strip()
    if not appname:
        print("[alerts] RELIEFWEB_APPNAME not set - skipping ReliefWeb lookup.")
        return []

    payload = {
        "limit": limit,
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "country.name", "value": country},
                {"field": "type.name", "value": "Flood"},
            ],
        },
        "sort": ["date.created:desc"],
        "fields": {"include": ["name", "status", "date", "url"]},
    }
    try:
        r = requests.post(RELIEFWEB_URL, params={"appname": appname}, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"[alerts] ReliefWeb request failed: {e}")
        return []

    alerts = []
    for item in data:
        fields = item.get("fields", {})
        alerts.append({
            "title": fields.get("name", "Untitled report"),
            "status": fields.get("status", "unknown"),
            "date": fields.get("date", {}).get("created", "")[:10],
            "url": fields.get("url", ""),
        })
    return alerts
