"""
Decision fusion layer — the "AI Decision Agent" from the project architecture.

Current implementation: a transparent, rule-based fusion function.

Roadmap (next phase, not yet implemented here):
  - Swap this for a CrewAI multi-agent crew (CV agent, weather agent, alerts
    agent, coordinator agent). Requires an LLM API key.
  - Add a RAG layer over NDMA/CWC guideline PDFs so recommendations can
    cite official procedure documents.
"""


def fuse_assessment(cv_result=None, weather_result=None, alerts=None, shelters=None):
    alerts = alerts or []
    shelters = shelters or []

    signals = []
    score = 0
    weight_total = 0

    if cv_result:
        cv_score_map = {"No flooding detected": 0, "Low": 25, "Moderate": 55, "High": 80, "Severe": 100}
        cv_score = cv_score_map.get(cv_result["severity"], 0)
        score += cv_score * 0.5
        weight_total += 0.5
        signals.append(f"Image/video analysis: {cv_result['severity']} ({cv_result['flood_pct']}% flood coverage)")

    if weather_result:
        score += weather_result["risk_score"] * 0.4
        weight_total += 0.4
        signals.append(f"Weather/river forecast risk: {weather_result['risk_level']} ({weather_result['risk_score']}/100)")

    if alerts:
        score += min(100, len(alerts) * 20) * 0.1
        weight_total += 0.1
        signals.append(f"{len(alerts)} active official flood report(s) found for this region")

    final_score = round(score / weight_total, 1) if weight_total else 0

    if final_score == 0:
        verdict = "No significant flood risk identified"
    elif final_score < 25:
        verdict = "Low risk — monitor conditions"
    elif final_score < 50:
        verdict = "Moderate risk — stay alert, avoid low-lying areas"
    elif final_score < 75:
        verdict = "High risk — prepare to move to higher ground"
    else:
        verdict = "Severe risk — evacuation to nearest safe location recommended"

    return {
        "final_score": final_score,
        "verdict": verdict,
        "signals": signals,
        "nearest_shelter": shelters[0] if shelters else None,
    }
