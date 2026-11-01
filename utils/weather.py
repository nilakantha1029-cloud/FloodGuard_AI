"""
Weather + flood risk data, built entirely on free, no-key-required APIs:

- Open-Meteo Forecast API   -> rainfall, precipitation probability
- Open-Meteo Flood API      -> river discharge forecast (proxy for flood risk)

Docs: https://open-meteo.com/en/docs  and  https://open-meteo.com/en/docs/flood-api
"""

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"


def geocode(place_name):
    """Free forward-geocoding via Open-Meteo's geocoding endpoint."""
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place_name, "count": 1},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        return None
    top = results[0]
    return {
        "lat": top["latitude"],
        "lon": top["longitude"],
        "name": top.get("name"),
        "country": top.get("country"),
    }


def get_weather_forecast(lat, lon):
    r = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,precipitation_probability",
            "daily": "precipitation_sum,precipitation_probability_max,rain_sum",
            "forecast_days": 5,
            "timezone": "auto",
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_flood_forecast(lat, lon):
    """River discharge forecast — a strong proxy signal for river/flash flood risk."""
    try:
        r = requests.get(
            FLOOD_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "river_discharge",
                "forecast_days": 7,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_past_week_rain(lat, lon):
    """
    Real historical daily rainfall for the last 7 days via Open-Meteo's
    free Archive API (no key required). This shows what actually
    happened, as distinct from the forward-looking forecast.
    """
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum",
                "past_days": 7,
                "forecast_days": 0,
                "timezone": "auto",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_today_tomorrow_verdict(weather_json, flood_json, has_active_alert=False):
    """
    A direct, plain-language yes/no read for TODAY and TOMORROW
    specifically, rather than making the user interpret a 5-day chart.
    Uses the first two entries of the daily forecast (index 0 = today).
    """
    daily = weather_json.get("daily", {})
    rain_sums = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    dates = daily.get("time", [])

    discharge_rising = False
    if flood_json:
        discharge = flood_json.get("daily", {}).get("river_discharge", [])
        discharge = [d for d in discharge if d is not None]
        if len(discharge) >= 2 and discharge[0] > 0:
            discharge_rising = (discharge[1] - discharge[0]) / discharge[0] > 0.03

    def verdict_for_day(idx):
        if idx >= len(rain_sums):
            return None
        rain = rain_sums[idx]
        prob = rain_prob[idx] if idx < len(rain_prob) else 0
        day_date = dates[idx] if idx < len(dates) else ""

        if has_active_alert:
            label, level = "Likely — active flooding already reported nearby", "danger"
        elif rain >= 50 or (rain >= 25 and prob >= 80):
            label, level = "Likely — heavy rain expected", "danger"
        elif rain >= 15 or (prob >= 70 and discharge_rising):
            label, level = "Possible — watch conditions closely", "warning"
        elif rain >= 5 or prob >= 50:
            label, level = "Unlikely, but some rain expected", "warning"
        else:
            label, level = "Not expected", "success"

        return {"date": day_date, "rain_mm": rain, "rain_prob": prob, "verdict": label, "level": level}

    return {"today": verdict_for_day(0), "tomorrow": verdict_for_day(1)}


def compute_risk(weather_json, flood_json, active_alerts_count=0):
    """
    Fuses rainfall forecast + river discharge trend + active official flood
    reports into a 0-100 risk score.

    IMPORTANT DESIGN NOTE (fixed after a real miss): rainfall and river-
    discharge *trend* alone only capture "is NEW flooding likely," not
    "is this place already flooded right now." A place mid-flood with
    falling (post-peak) river discharge and light forecast rain would
    otherwise score as low/moderate risk, which is misleading. To fix
    this, an active-alert signal acts as a floor: if there are current
    official flood reports for the country, the score cannot drop below
    a "High" floor, regardless of what the forward-looking weather trend
    says. This is a transparent rule-based scorer — every number here is
    inspectable, not a black box.
    """
    daily = weather_json.get("daily", {})
    rain_sums = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [])

    next_48h_rain = sum(rain_sums[:2]) if rain_sums else 0
    max_prob = max(rain_prob[:2]) if rain_prob else 0

    rain_score = min(60, (next_48h_rain / 150) * 60)
    prob_score = (max_prob / 100) * 20

    discharge_score = 0
    discharge_note = "No river-discharge station near this location."
    if flood_json:
        discharge = flood_json.get("daily", {}).get("river_discharge", [])
        discharge = [d for d in discharge if d is not None]
        if len(discharge) >= 2:
            trend = (discharge[-1] - discharge[0]) / max(discharge[0], 1)
            discharge_score = min(20, max(0, trend * 40))
            discharge_note = f"River discharge trend over forecast window: {trend*100:+.1f}%"

    forecast_score = round(min(100, rain_score + prob_score + discharge_score), 1)

    # Active-alert floor: official reports mean real flooding is happening
    # NOW, which forward-looking weather math alone can miss entirely.
    has_active_alert = active_alerts_count > 0
    total = forecast_score
    if has_active_alert:
        total = max(total, 65)  # floor into "High" band regardless of forecast trend
    total = round(min(100, total), 1)

    if total < 20:
        level = "Low"
    elif total < 45:
        level = "Moderate"
    elif total < 70:
        level = "High"
    else:
        level = "Severe"

    return {
        "risk_score": total,
        "forecast_only_score": forecast_score,
        "risk_level": level,
        "next_48h_rain_mm": round(next_48h_rain, 1),
        "max_rain_probability": max_prob,
        "discharge_note": discharge_note,
        "has_active_alert": has_active_alert,
    }
