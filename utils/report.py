"""
Builds a downloadable plain-text incident report combining every signal
the app has gathered: media analysis, location risk, official alerts,
nearest shelter, and emergency contacts.
"""

from datetime import datetime


def build_report(location_name, cv_result, weather_result, alerts, shelters, contacts, fused):
    lines = []
    lines.append("=" * 60)
    lines.append("FLOODGUARD AI — INCIDENT ASSESSMENT REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Location: {location_name or 'Not specified'}")
    lines.append("")

    lines.append("-- FINAL AI VERDICT " + "-" * 39)
    lines.append(f"Risk score: {fused['final_score']}/100")
    lines.append(f"Verdict: {fused['verdict']}")
    lines.append("Reasoning:")
    for s in fused["signals"]:
        lines.append(f"  - {s}")
    lines.append("")

    if cv_result:
        lines.append("-- MEDIA ANALYSIS " + "-" * 41)
        lines.append(f"Severity: {cv_result['severity']}")
        lines.append(f"Flooded area: {cv_result['flood_pct']}%")
        lines.append("Class breakdown:")
        for cls, pct in sorted(cv_result["class_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {cls}: {pct}%")
        if cv_result.get("demo_mode"):
            lines.append("  (NOTE: generated in demo/heuristic mode — no trained model loaded)")
        lines.append("")

    if weather_result:
        lines.append("-- WEATHER & RIVER FORECAST " + "-" * 31)
        lines.append(f"Risk level: {weather_result['risk_level']} ({weather_result['risk_score']}/100)")
        lines.append(f"Rain expected (next 48h): {weather_result['next_48h_rain_mm']} mm")
        lines.append(f"Peak rain probability: {weather_result['max_rain_probability']}%")
        lines.append(f"River discharge: {weather_result['discharge_note']}")
        lines.append("")

    if alerts:
        lines.append("-- OFFICIAL FLOOD REPORTS (ReliefWeb) " + "-" * 21)
        for a in alerts:
            lines.append(f"  - [{a['date']}] {a['title']} ({a['status']})")
        lines.append("")

    if shelters:
        lines.append("-- NEAREST SAFE LOCATIONS " + "-" * 33)
        for s in shelters:
            lines.append(f"  - {s['name']} ({s['type']}) — {s['distance_km']} km away")
        lines.append("")

    lines.append("-- EMERGENCY CONTACTS " + "-" * 37)
    for c in contacts:
        lines.append(f"  - {c['name']}: {c['number']}  ({c['note']})")
    lines.append("")

    lines.append("=" * 60)
    lines.append("Data sources: Open-Meteo (weather/river) · OpenStreetMap (shelters)")
    lines.append("· ReliefWeb/OCHA (official alerts) · NASA GIBS (satellite reference)")
    lines.append("This report is AI-generated and should be verified against official")
    lines.append("local emergency management guidance before acting on it.")
    lines.append("=" * 60)

    return "\n".join(lines)
