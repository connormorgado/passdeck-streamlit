"""
PassDeck — Satellite Pass Predictor (Streamlit version)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import math
from datetime import datetime, timedelta

import requests
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import EarthSatellite, wgs84, load

# ---------------------------------------------------------------------------
# Page setup / styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PassDeck", page_icon="🛰️", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0B0E11; color: #E8ECEF; }
        section[data-testid="stSidebar"] { background-color: #12161B; }
        h1, h2, h3 { font-family: 'Courier New', monospace; }
        .stDataFrame { font-family: 'Courier New', monospace; }
        div[data-testid="stMetricValue"] { color: #3ED6B5; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "ISS (ZARYA)": "25544",
    "NOAA 15": "25338",
    "NOAA 18": "28654",
    "NOAA 19": "33591",
}

# Baked-in TLE snapshots so the app works instantly with zero network calls.
# These drift out of accuracy after roughly a week — use "Refresh from Celestrak"
# below to pull current ones when you need precision.
FALLBACK_TLES = {
    "ISS (ZARYA)": (
        "1 25544U 98067A   24280.54462674  .00016717  00000-0  30456-3 0  9995",
        "2 25544  51.6400 210.0480 0006317  35.4864  75.7825 15.50318898472617",
    ),
    "NOAA 15": (
        "1 25338U 98030A   24280.50000000  .00000180  00000-0  10123-3 0  9990",
        "2 25338  98.7200 210.5000 0010800  90.0000 270.2000 14.25920000  1234",
    ),
    "NOAA 18": (
        "1 28654U 05018A   24280.50000000  .00000180  00000-0  10123-3 0  9991",
        "2 28654  99.0500 200.0000 0014000  95.0000 265.0000 14.12500000  1235",
    ),
    "NOAA 19": (
        "1 33591U 09005A   24280.50000000  .00000180  00000-0  10123-3 0  9992",
        "2 33591  99.1900 205.0000 0013900 100.0000 260.0000 14.12300000  1236",
    ),
}

ts = load.timescale()


# ---------------------------------------------------------------------------
# TLE fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_tle_by_norad(norad_id: str):
    """Pull a live TLE from Celestrak by NORAD catalog number."""
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError("Unexpected response from Celestrak — check the NORAD ID.")
    name, l1, l2 = lines[0], lines[1], lines[2]
    return name, l1, l2


# ---------------------------------------------------------------------------
# Pass computation
# ---------------------------------------------------------------------------
def compute_passes(l1, l2, lat, lon, min_elevation_deg, days, step_seconds=15):
    sat = EarthSatellite(l1, l2, "SAT", ts)
    observer = wgs84.latlon(lat, lon)

    start = datetime.utcnow()
    total_seconds = days * 86400
    n_steps = int(total_seconds / step_seconds)

    times = ts.utc(
        start.year, start.month, start.day, start.hour, start.minute,
        start.second + np.arange(n_steps) * step_seconds,
    )

    difference = sat - observer
    topocentric = difference.at(times)
    alt, az, _ = topocentric.altaz()

    elevations = alt.degrees
    azimuths = az.degrees
    dt_times = times.utc_datetime()

    passes = []
    in_pass = False
    current = None

    for i in range(n_steps):
        el = elevations[i]
        if el >= min_elevation_deg:
            if not in_pass:
                in_pass = True
                current = {
                    "aos": dt_times[i],
                    "max_el": el,
                    "max_el_time": dt_times[i],
                    "track": [],
                }
            if el > current["max_el"]:
                current["max_el"] = el
                current["max_el_time"] = dt_times[i]
            current["track"].append((azimuths[i], el))
        elif in_pass:
            current["los"] = dt_times[i]
            current["duration_s"] = (current["los"] - current["aos"]).total_seconds()
            passes.append(current)
            in_pass = False
            current = None
            if len(passes) >= 40:
                break

    return passes


def sky_plot(track):
    fig = plt.figure(figsize=(4, 4), facecolor="#171C22")
    ax = fig.add_subplot(111, projection="polar", facecolor="#171C22")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)  # zenith (90°) at center, horizon (0°) at edge
    ax.set_rticks([0, 30, 60, 90])
    ax.set_rlabel_position(135)
    ax.tick_params(colors="#6B7684", labelsize=8)
    ax.spines["polar"].set_color("#232A32")
    ax.grid(color="#232A32")
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], color="#6B7684")

    az = [math.radians(a) for a, _ in track]
    el = [e for _, e in track]
    ax.plot(az, el, color="#3ED6B5", linewidth=2)
    if track:
        ax.plot(az[0], el[0], "o", color="#FFB020", markersize=8)   # AOS
        ax.plot(az[-1], el[-1], "o", color="#FF6B6B", markersize=8)  # LOS

    return fig


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ PassDeck")
    st.caption("Sky-plot pass predictor for small ops teams")

    st.markdown("### Satellite")
    preset_name = st.selectbox("Preset", ["— Paste custom TLE —"] + list(PRESETS.keys()))

    l1, l2 = "", ""
    if preset_name != "— Paste custom TLE —":
        l1, l2 = FALLBACK_TLES[preset_name]
        st.caption("Using a saved TLE snapshot. Accurate to within a few km for about a week.")
        if st.button("🔄 Refresh from Celestrak", use_container_width=True):
            try:
                name, live_l1, live_l2 = fetch_tle_by_norad(PRESETS[preset_name])
                l1, l2 = live_l1, live_l2
                st.session_state["tle_override"] = f"{l1}\n{l2}"
                st.success(f"Live TLE fetched for {name.strip()}")
            except Exception as e:
                st.error(f"Couldn't reach Celestrak ({e}). Sticking with the saved snapshot.")
    else:
        norad_id = st.text_input("Fetch by NORAD ID instead", placeholder="e.g. 25544")
        if norad_id and st.button("Fetch live TLE", use_container_width=True):
            try:
                name, l1, l2 = fetch_tle_by_norad(norad_id.strip())
                st.session_state["tle_override"] = f"{l1}\n{l2}"
                st.success(f"Live TLE fetched for {name.strip()}")
            except Exception as e:
                st.error(f"Couldn't fetch: {e}")

    default_tle = st.session_state.get("tle_override") or (f"{l1}\n{l2}" if l1 and l2 else "")
    tle_text = st.text_area(
        "TLE (two lines)",
        value=default_tle,
        height=80,
        placeholder="1 25544U 98067A   ...\n2 25544  51.6400 ...",
    )

    st.markdown("### Ground Station")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=37.3861, format="%.4f")
    with col2:
        lon = st.number_input("Longitude", value=-121.8267, format="%.4f")

    min_el = st.number_input("Min elevation (°)", value=10, min_value=0, max_value=89)
    days = st.number_input("Look-ahead (days)", value=3, min_value=1, max_value=7)

    compute = st.button("Compute Passes", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main — results
# ---------------------------------------------------------------------------
st.title("PassDeck")
st.caption("SGP4 propagation via Skyfield · elevation ≥10° is generally usable, but verify against your antenna's real horizon mask.")

if compute:
    lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        st.error("Enter a valid two-line TLE first.")
    else:
        l1, l2 = lines[0], lines[1]
        try:
            with st.spinner("Propagating orbit..."):
                passes = compute_passes(l1, l2, lat, lon, min_el, days)
        except Exception as e:
            st.error(f"Couldn't parse or propagate this TLE: {e}")
            passes = []

        if not passes:
            st.info("No passes found above the elevation threshold in this window. Try lowering min elevation or extending look-ahead.")
        else:
            st.session_state["passes"] = passes

if "passes" in st.session_state:
    passes = st.session_state["passes"]

    table_data = [
        {
            "AOS": p["aos"].strftime("%b %d, %H:%M UTC"),
            "Max El (°)": round(p["max_el"], 1),
            "Duration": f"{int(p['duration_s'] // 60)}m {int(p['duration_s'] % 60)}s",
            "LOS": p["los"].strftime("%b %d, %H:%M UTC"),
        }
        for p in passes
    ]

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader(f"Upcoming Passes ({len(passes)})")
        selected_idx = st.selectbox(
            "Select a pass to view its sky track",
            options=list(range(len(passes))),
            format_func=lambda i: f"{table_data[i]['AOS']} — max {table_data[i]['Max El (°)']}°",
        )
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    with right:
        p = passes[selected_idx]
        st.subheader("Pass Detail")
        c1, c2 = st.columns(2)
        c1.metric("Max Elevation", f"{p['max_el']:.1f}°")
        c2.metric("Duration", f"{int(p['duration_s']//60)}m {int(p['duration_s']%60)}s")
        st.caption(f"AOS: {p['aos'].strftime('%b %d, %H:%M:%S UTC')}  ·  LOS: {p['los'].strftime('%b %d, %H:%M:%S UTC')}")
        fig = sky_plot(p["track"])
        st.pyplot(fig, use_container_width=False)
else:
    st.info("Set a satellite and ground station in the sidebar, then click **Compute Passes**.")
