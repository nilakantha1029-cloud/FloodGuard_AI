import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import folium
from streamlit_folium import st_folium

from utils.model import FloodDetector, colorize_mask
from utils.weather import geocode, get_weather_forecast, get_flood_forecast, compute_risk, get_past_week_rain, get_today_tomorrow_verdict
from utils.shelters import find_nearby_safe_points
from utils.alerts import get_flood_alerts
from utils.agent import fuse_assessment
from utils.emergency_contacts import get_emergency_contacts
from utils.satellite import get_satellite_tile_url
from utils.report import build_report

st.set_page_config(page_title="FloodGuard AI", page_icon="🌊", layout="wide")

with open(os.path.join(os.path.dirname(__file__), "assets", "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    return FloodDetector(weights_path="models/best_model_pretrained.pth")


detector = load_detector()

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🌊</div>
        <div>
            <div class="sb-brand-text">FloodGuard AI</div>
            <div class="sb-brand-sub">Disaster intelligence platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_dot = "dot-demo" if detector.demo_mode else "dot-live"
    model_status = "Demo mode (heuristic)" if detector.demo_mode else "Trained model loaded"
    st.markdown(f"""
    <div class="sb-status-card">
        <div class="sb-status-row"><span class="dot {model_dot}"></span> CV model: {model_status}</div>
        <div class="sb-status-row"><span class="dot dot-live"></span> Weather API: Open-Meteo (live)</div>
        <div class="sb-status-row"><span class="dot dot-live"></span> Shelters: OpenStreetMap (live)</div>
        <div class="sb-status-row"><span class="dot dot-live"></span> Alerts: ReliefWeb / OCHA (live)</div>
        <div class="sb-status-row"><span class="dot dot-live"></span> Satellite: NASA GIBS (live)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="font-size:0.9rem;">Risk legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-status-card">
        <div class="sb-legend-item"><span class="sb-legend-swatch" style="background:#22c55e;"></span> Low risk</div>
        <div class="sb-legend-item"><span class="sb-legend-swatch" style="background:#f59e0b;"></span> Moderate risk</div>
        <div class="sb-legend-item"><span class="sb-legend-swatch" style="background:#ef4444;"></span> High risk</div>
        <div class="sb-legend-item"><span class="sb-legend-swatch" style="background:#b91c1c;"></span> Severe risk</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Built with computer vision, live weather/river data, and official disaster reporting.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">🌊 FloodGuard AI</div>
      <div class="hero-sub">Computer vision, live weather intelligence, satellite reference imagery, and
      official alerts — fused into one flood risk picture, with nearest-shelter routing and emergency contacts.</div>
      <div class="hero-badges">
        <span class="hero-badge">📷 CV flood detection</span>
        <span class="hero-badge">🗺️ Click-to-select map</span>
        <span class="hero-badge">🛰️ Live satellite layer</span>
        <span class="hero-badge">🚨 Official alerts</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if detector.demo_mode:
    st.warning(
        "⚠️ No trained model checkpoint found at `models/best_model_pretrained.pth` — "
        "running in **demo mode** with a color-heuristic fallback.",
        icon="⚠️",
    )

tab1, tab2, tab3, tab4 = st.tabs(["📷  Media Analysis", "📍  Location Risk", "🚨  Alerts & Shelters", "📄  Full Report"])

for key, default in [
    ("cv_result", None), ("weather_result", None), ("alerts", []), ("shelters", []),
    ("location_name", None), ("clicked_latlon", None), ("analyzed_latlon", None),
    ("selection_source", None), ("shelter_status", None), ("shelter_radius", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================== TAB 1 — MEDIA ANALYSIS
with tab1:
    st.markdown('<div class="section-label">📷 Upload a photo or video to check for flooding</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])

    with col_a:
        media_type = st.radio("Media type", ["Image", "Video"], horizontal=True)
        uploaded = st.file_uploader(
            "Upload file",
            type=["jpg", "jpeg", "png"] if media_type == "Image" else ["mp4", "mov", "avi"],
        )
        run_btn = st.button("🔍 Analyze media", type="primary", use_container_width=True)

    if uploaded and run_btn:
        if media_type == "Image":
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            with st.spinner("Running flood segmentation..."):
                result = detector.predict(img_bgr)
            st.session_state.cv_result = result

            overlay = colorize_mask(cv2.resize(result["mask"], (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST))
            blended = cv2.addWeighted(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0.6, overlay, 0.4, 0)

            with col_b:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

            c1, c2 = st.columns([1.3, 1])
            c1.image(blended, caption="Detected flood overlay", use_container_width=True)
            with c2:
                flood_pct = result["flood_pct"]
                verdict = "🌊 Flood detected" if flood_pct >= 15 else ("⚠️ Possible flooding" if flood_pct >= 3 else "✅ No flooding detected")
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Verdict</div>
                    <div class="metric-value" style="font-size:1.3rem;">{verdict}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Severity</div>
                    <div class="metric-value">{result['severity']}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Flooded area</div>
                    <div class="metric-value">{flood_pct}%</div>
                    <div class="flood-bar-track"><div class="flood-bar-fill" style="width:{flood_pct}%;"></div></div>
                </div>""", unsafe_allow_html=True)
                with st.expander("Full class breakdown"):
                    for cls, pct in sorted(result["class_breakdown"].items(), key=lambda x: -x[1]):
                        st.write(f"{cls}: {pct}%")

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_every = max(1, frame_count // 12)

            progress = st.progress(0, text="Analyzing frames...")
            flood_series = []
            idx = 0
            best_frame, best_pct = None, -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % sample_every == 0:
                    res = detector.predict(frame)
                    flood_series.append(res["flood_pct"])
                    if res["flood_pct"] > best_pct:
                        best_pct = res["flood_pct"]
                        best_frame = (frame, res)
                    progress.progress(min(1.0, idx / max(1, frame_count)), text="Analyzing frames...")
                idx += 1
            cap.release()
            progress.empty()

            if flood_series:
                avg_pct = round(sum(flood_series) / len(flood_series), 2)
                worst = best_frame[1]
                st.session_state.cv_result = worst
                verdict = "🌊 Flood detected" if avg_pct >= 15 else ("⚠️ Possible flooding" if avg_pct >= 3 else "✅ No flooding detected")

                st.line_chart(flood_series)
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"""<div class="metric-card"><div class="metric-label">Verdict</div><div class="metric-value" style="font-size:1.1rem;">{verdict}</div></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="metric-card"><div class="metric-label">Avg flood coverage</div><div class="metric-value">{avg_pct}%</div></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="metric-card"><div class="metric-label">Peak coverage</div><div class="metric-value">{best_pct}%</div></div>""", unsafe_allow_html=True)
                c4.markdown(f"""<div class="metric-card"><div class="metric-label">Worst-frame severity</div><div class="metric-value">{worst['severity']}</div></div>""", unsafe_allow_html=True)

                frame_bgr, _ = best_frame
                overlay = colorize_mask(cv2.resize(worst["mask"], (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST))
                blended = cv2.addWeighted(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 0.6, overlay, 0.4, 0)
                st.image(blended, caption="Most-flooded frame detected", use_container_width=True)
            else:
                st.error("Could not read frames from this video.")
            os.unlink(tfile.name)
    elif not uploaded:
        st.info("Upload an image or video, then click **Analyze media**.")

# ============================================================== TAB 2 — LOCATION RISK
with tab2:
    st.markdown('<div class="section-label">📍 Choose how to select a location</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Selection method",
        ["🔎 Search by name", "🖱️ Click on map"],
        horizontal=True,
        label_visibility="collapsed",
        key="location_mode",
    )
    show_satellite = st.checkbox("🛰️ Show satellite layer", value=False)

    # ---- Mode 1: Search (independent from map clicks — does not read map state) ----
    if mode == "🔎 Search by name":
        place = st.text_input("Search a place name, city, or address", placeholder="e.g. Secunderabad, Telangana")
        if st.button("🔎 Search this place", use_container_width=True):
            if place:
                geo = geocode(place)
                if geo:
                    st.session_state.clicked_latlon = (geo["lat"], geo["lon"])
                    st.session_state.location_name = f"{geo['name']}, {geo.get('country', '')}"
                    st.session_state.selection_source = "search"
                else:
                    st.error("Could not find that location. Try a more specific name.")
            else:
                st.warning("Type a place name first.")

        if st.session_state.clicked_latlon and st.session_state.get("selection_source") == "search":
            st.success(f"📍 Selected via search: **{st.session_state.location_name}**")

    # ---- Mode 2: Click on map (independent from search — does not read search box) ----
    else:
        center = st.session_state.clicked_latlon if st.session_state.get("selection_source") == "map" else (20.5937, 78.9629)
        zoom = 11 if st.session_state.get("selection_source") == "map" else 5

        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")
        if show_satellite:
            folium.TileLayer(tiles=get_satellite_tile_url(), attr="NASA GIBS", name="Satellite (NASA GIBS, ~1 day delay)", overlay=False).add_to(m)
            folium.LayerControl().add_to(m)

        if st.session_state.get("selection_source") == "map" and st.session_state.clicked_latlon:
            folium.Marker(st.session_state.clicked_latlon, tooltip="Selected location", icon=folium.Icon(color="cadetblue", icon="map-pin", prefix="fa")).add_to(m)

        st.caption("🖱️ Click anywhere on the map to drop a pin.")
        map_data = st_folium(m, height=440, width=None, key="main_map")

        if map_data and map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            st.session_state.clicked_latlon = (lat, lon)
            st.session_state.location_name = f"{lat:.4f}, {lon:.4f}"
            st.session_state.selection_source = "map"

        if st.session_state.clicked_latlon and st.session_state.get("selection_source") == "map":
            st.success(f"📍 Selected via map click: **{st.session_state.location_name}**")

    has_selection = st.session_state.clicked_latlon is not None
    analyze_btn = st.button("⚡ Analyze this location", type="primary", use_container_width=True, disabled=not has_selection)

    if analyze_btn and has_selection:
        lat, lon = st.session_state.clicked_latlon
        with st.spinner("Fetching weather, river, historical rainfall, satellite reference, and alert data..."):
            weather_json = get_weather_forecast(lat, lon)
            flood_json = get_flood_forecast(lat, lon)
            alerts = get_flood_alerts()
            risk = compute_risk(weather_json, flood_json, active_alerts_count=len(alerts))
            shelter_result = find_nearby_safe_points(lat, lon)
            past_week = get_past_week_rain(lat, lon)
            outlook = get_today_tomorrow_verdict(weather_json, flood_json, has_active_alert=risk["has_active_alert"])

            st.session_state.weather_result = risk
            st.session_state.shelters = shelter_result["results"]
            st.session_state.shelter_status = shelter_result["status"]
            st.session_state.shelter_radius = shelter_result["radius_used_km"]
            st.session_state.alerts = alerts
            st.session_state.analyzed_latlon = (lat, lon)
            st.session_state._weather_json = weather_json
            st.session_state._past_week = past_week
            st.session_state._outlook = outlook

    if st.session_state.weather_result and st.session_state.analyzed_latlon:
        risk = st.session_state.weather_result
        lat, lon = st.session_state.analyzed_latlon
        risk_colors = {"Low": "success", "Moderate": "warning", "High": "danger", "Severe": "danger"}
        css_class = risk_colors.get(risk["risk_level"], "warning")

        st.markdown(f"""
        <div class="risk-banner {css_class}">
            <div class="risk-banner-title">📍 {st.session_state.location_name}</div>
            <div class="risk-banner-score">{risk['risk_score']}/100 — {risk['risk_level']} flood risk</div>
        </div>
        """, unsafe_allow_html=True)

        if risk.get("has_active_alert"):
            st.warning(
                f"🚨 **Active official flood report(s) found for this country.** The forecast-only signal "
                f"(rain + river trend) alone scored {risk['forecast_only_score']}/100 — but since real "
                f"flooding is already being officially reported, the risk shown above is floored at a "
                f"minimum of 65 (High) rather than trusting forecast trend alone. Note: these alerts are "
                f"country-level, not confirmed for this exact district — check the Alerts tab for details.",
                icon="🚨",
            )

        # ---- Today / Tomorrow verdict ----
        st.markdown('<div class="section-label">📆 Today &amp; tomorrow outlook</div>', unsafe_allow_html=True)
        outlook = st.session_state.get("_outlook", {})
        oc1, oc2 = st.columns(2)
        for col, key, label in [(oc1, "today", "Today"), (oc2, "tomorrow", "Tomorrow")]:
            d = outlook.get(key)
            if d:
                banner_class = risk_colors_map = {"success": "success", "warning": "warning", "danger": "danger"}[d["level"]]
                col.markdown(f"""
                <div class="risk-banner {banner_class}" style="padding:1.1rem 1.3rem;">
                    <div class="risk-banner-title">{label} · {d['date']}</div>
                    <div class="risk-banner-score" style="font-size:1.25rem;">{d['verdict']}</div>
                    <div style="font-size:0.8rem; opacity:0.85; margin-top:6px;">{d['rain_mm']}mm rain · {d['rain_prob']}% probability</div>
                </div>
                """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card"><div class="metric-label">🌧️ Rain (next 48h)</div><div class="metric-value">{risk['next_48h_rain_mm']} mm</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><div class="metric-label">☔ Peak rain probability</div><div class="metric-value">{risk['max_rain_probability']}%</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><div class="metric-label">🌊 River discharge</div><div class="metric-value" style="font-size:13px;">{risk['discharge_note']}</div></div>""", unsafe_allow_html=True)

        # ---- Past 7 days (real historical data) vs forecast ----
        past_week = st.session_state.get("_past_week")
        st.markdown('<div class="section-label">🕐 Past 7 days — actual rainfall</div>', unsafe_allow_html=True)
        if past_week and past_week.get("daily", {}).get("time"):
            past_daily = past_week["daily"]
            st.bar_chart({"Rain (mm) — last 7 days": past_daily["precipitation_sum"]})
            total_past_week = round(sum(v for v in past_daily["precipitation_sum"] if v is not None), 1)
            st.caption(f"Total rainfall over the past 7 days: **{total_past_week} mm** (actual recorded data, not forecast)")
        else:
            st.info("Historical rainfall data unavailable for this location.")

        daily = st.session_state._weather_json.get("daily", {})
        if daily.get("time"):
            st.markdown('<div class="section-label">📊 5-day forecast — upcoming rainfall</div>', unsafe_allow_html=True)
            st.bar_chart({"Rain (mm) — forecast": daily["precipitation_sum"]})

        st.markdown('<div class="section-label">🗺️ Results overview</div>', unsafe_allow_html=True)
        result_map = folium.Map(location=(lat, lon), zoom_start=12, tiles="CartoDB dark_matter")
        if show_satellite:
            folium.TileLayer(tiles=get_satellite_tile_url(), attr="NASA GIBS", name="Satellite", overlay=False).add_to(result_map)
            folium.LayerControl().add_to(result_map)

        risk_marker_color = {"Low": "green", "Moderate": "orange", "High": "red", "Severe": "darkred"}.get(risk["risk_level"], "orange")
        folium.CircleMarker(
            (lat, lon), radius=12, color=risk_marker_color, fill=True, fill_opacity=0.5,
            tooltip=f"{risk['risk_level']} risk — {risk['risk_score']}/100",
        ).add_to(result_map)
        for s in st.session_state.shelters:
            folium.Marker(
                (s["lat"], s["lon"]), tooltip=f"{s['name']} ({s['distance_km']} km)",
                icon=folium.Icon(color="blue", icon="plus-square", prefix="fa"),
            ).add_to(result_map)
        st_folium(result_map, height=300, width=None, key="results_map")

# ============================================================== TAB 3 — ALERTS & SHELTERS
with tab3:
    st.markdown('<div class="section-label">🏥 Nearest safe locations</div>', unsafe_allow_html=True)
    if st.session_state.shelters:
        if st.session_state.shelter_radius:
            st.caption(f"Found within {st.session_state.shelter_radius} km — includes hospitals, schools, places of worship, and community halls (common real-world shelter sites), not just formally tagged 'shelter' points.")
        for s in st.session_state.shelters:
            st.markdown(f"""
            <div class="shelter-card">
                <div class="shelter-name">🏥 {s['name']}</div>
                <div class="shelter-meta">{s['type'].replace('_',' ').title()} · {s['distance_km']} km away</div>
            </div>
            """, unsafe_allow_html=True)
    elif st.session_state.shelter_status == "error":
        st.error("⚠️ Could not reach OpenStreetMap's shelter lookup service (network/timeout issue). Try analyzing this location again.")
    elif st.session_state.shelter_status == "empty":
        st.warning(
            f"No shelters, hospitals, schools, or community buildings are tagged in OpenStreetMap within "
            f"{st.session_state.shelter_radius} km of this location. This is a real gap in map data for "
            f"smaller towns, not necessarily an absence of real shelters on the ground — check with local "
            f"disaster authorities directly for this area."
        )
    else:
        st.info("Run a location check in the **Location Risk** tab to see nearby shelters.")

    st.markdown('<div class="section-label">🚨 Official flood reports</div>', unsafe_allow_html=True)
    if st.session_state.alerts:
        for a in st.session_state.alerts:
            st.markdown(f"""
            <div class="alert-card">
                <div class="alert-title">🚨 {a['title']}</div>
                <div class="alert-meta">{a['date']} · {a['status'].title()}</div>
            </div>
            """, unsafe_allow_html=True)
    elif st.session_state.location_name:
        st.info("No active flood reports found via ReliefWeb for this region right now.")
    else:
        st.info("Run a location check first to pull official flood reports.")

    st.markdown('<div class="section-label">📞 Emergency contacts (India)</div>', unsafe_allow_html=True)
    st.caption("General national helplines — always verify with local authorities during an active emergency.")
    contacts = get_emergency_contacts("India")
    cc1, cc2 = st.columns(2)
    for i, c in enumerate(contacts):
        target = cc1 if i % 2 == 0 else cc2
        target.markdown(f"""
        <div class="contact-card">
            <div>
                <div class="contact-name">{c['name']}</div>
                <div class="contact-note">{c['note']}</div>
            </div>
            <div class="contact-number">{c['number']}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================== TAB 4 — FULL REPORT
with tab4:
    st.markdown('<div class="section-label">📄 Combined AI assessment</div>', unsafe_allow_html=True)
    if not st.session_state.cv_result and not st.session_state.weather_result:
        st.info("Analyze media and/or check a location first — this tab fuses everything into one recommendation and a downloadable report.")
    else:
        contacts = get_emergency_contacts("India")
        rec = fuse_assessment(
            cv_result=st.session_state.cv_result,
            weather_result=st.session_state.weather_result,
            alerts=st.session_state.alerts,
            shelters=st.session_state.shelters,
        )
        risk_colors = {"Low": "success", "Moderate": "warning", "High": "danger", "Severe": "danger"}
        band = "Low" if rec["final_score"] < 25 else "Moderate" if rec["final_score"] < 50 else "High" if rec["final_score"] < 75 else "Severe"

        st.markdown(f"""
        <div class="risk-banner {risk_colors.get(band,'warning')}">
            <div class="risk-banner-title">Final AI assessment</div>
            <div class="risk-banner-score">{rec['final_score']}/100 — {rec['verdict']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">🧠 Reasoning trail</div>', unsafe_allow_html=True)
        for sig in rec["signals"]:
            st.markdown(f"- {sig}")

        if rec["nearest_shelter"]:
            sh = rec["nearest_shelter"]
            st.markdown(f"""
            <div class="glass-card">
                <div class="shelter-name">🏥 Nearest safe location: {sh['name']}</div>
                <div class="shelter-meta">{sh['distance_km']} km away</div>
            </div>
            """, unsafe_allow_html=True)

        report_text = build_report(
            location_name=st.session_state.location_name,
            cv_result=st.session_state.cv_result,
            weather_result=st.session_state.weather_result,
            alerts=st.session_state.alerts,
            shelters=st.session_state.shelters,
            contacts=contacts,
            fused=rec,
        )
        st.download_button("⬇ Download full report (.txt)", report_text, file_name="floodguard_report.txt", use_container_width=True)

st.markdown(
    """<div class="footer">FloodGuard AI · Weather &amp; river data: Open-Meteo · Shelters: OpenStreetMap ·
    Alerts: ReliefWeb (OCHA) · Satellite: NASA GIBS · Not a substitute for official emergency guidance</div>""",
    unsafe_allow_html=True,
)
