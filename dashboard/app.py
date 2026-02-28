import streamlit as st
import time
import pandas as pd

from analytics.stats import AnalyticsManager


st.set_page_config(
    page_title="Multi-Camera Tracking Analytics",
    layout="wide",
)

st.title("Multi-Camera Pedestrian Analytics")

# --- Session state ---
if "analytics" not in st.session_state:
    st.session_state.analytics = AnalyticsManager()

analytics = st.session_state.analytics


# --- Placeholder for metrics ---
col1, col2 = st.columns(2)
unique_placeholder = col1.empty()
active_placeholder = col2.empty()

st.divider()

table_placeholder = st.empty()


def update_dashboard():
    """Refresh dashboard metrics"""
    summary = analytics.summary()

    unique_people = summary["unique_people"]
    people = summary["people"]

    active_now = sum(1 for p in people if p["dwell_time"] > 0)

    unique_placeholder.metric("Unique People", unique_people)
    active_placeholder.metric("Active Tracks", active_now)

    if people:
        df = pd.DataFrame(people)
        table_placeholder.dataframe(df, use_container_width=True)
    else:
        table_placeholder.info("No tracks yet")


# --- Simulated update loop ---
st.caption("Waiting for tracking data from main pipeline...")

while True:
    update_dashboard()
    time.sleep(1)