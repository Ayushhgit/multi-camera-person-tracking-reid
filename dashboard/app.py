import streamlit as st
import time
import pandas as pd
import cv2
import numpy as np
import sys
import os
import threading

# Add parent directory so we can import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video import MOTCamera
from utils.draw import draw_tracks, stack_cameras
from detectors.yolo_detector import YoloDetector
from trackers.deepsort_tracker import DeepSortTracker
from multicam.identity_manager import GlobalIdentityManager
from analytics.stats import AnalyticsManager

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Multi-Camera Tracking", layout="wide")
st.title("Multi-Camera Pedestrian Analytics")
st.markdown("Live detection, ReID tracking, and behavior analytics.")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Control Panel")
    start_btn = st.button("Start Pipeline", use_container_width=True, type="primary")
    stop_btn  = st.button("Stop / Reset",  use_container_width=True)
    st.divider()
    st.subheader("Settings")
    skip_frames = st.slider("Detect Every Nth Frame", 1, 10, 5,
                            help="Higher = faster video, detection less frequent")
    conf_thresh = st.slider("YOLO Confidence", 0.1, 1.0, 0.4)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# A plain Python dict shared between the UI thread and background thread.
# This avoids any st.session_state access from the background thread.
if "shared" not in st.session_state:
    st.session_state.shared = {
        "running": False,
        "latest_jpeg": None,   # store JPEG bytes, not raw numpy
        "analytics": None,
    }

SHARED = st.session_state.shared

if start_btn:
    st.session_state.pipeline_running = True
if stop_btn:
    st.session_state.pipeline_running = False
    SHARED["running"] = False

# ============================================================
# UI PLACEHOLDERS
# ============================================================
col1, col2 = st.columns(2)
metric_unique = col1.empty()
metric_active = col2.empty()
st.divider()
video_placeholder = st.empty()
st.divider()
st.subheader("Identity Analytics")
table_placeholder = st.empty()


# ============================================================
# BACKGROUND TRACKING THREAD
# ============================================================
def _encode_jpeg(rgb_frame, quality=60):
    """Convert RGB numpy frame to JPEG bytes — much smaller than PNG."""
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def tracking_worker(conf, skip_n, state):
    """Runs in a daemon thread. Reads frames, detects, tracks, writes JPEG to state dict."""
    cam1 = MOTCamera("data/mot17/train/MOT17-02/img1", max_frames=1500, size=(480, 270))
    cam2 = MOTCamera("data/mot17/train/MOT17-04/img1", max_frames=1500, size=(480, 270))

    detector = YoloDetector(conf_threshold=conf)
    tracker1 = DeepSortTracker()
    tracker2 = DeepSortTracker()
    identity_mgr = GlobalIdentityManager(similarity_threshold=0.85)
    analytics = state["analytics"]

    frame_idx = 0
    prev_gt1, prev_gt2 = [], []

    while state.get("running", False):
        ret1, f1 = cam1.read()
        ret2, f2 = cam2.read()
        if not ret1 or not ret2:
            state["running"] = False
            break

        frame_idx += 1

        # --- heavy work only every N-th frame ---
        if frame_idx % skip_n == 0:
            d1 = detector.detect(f1)
            d2 = detector.detect(f2)
            tracker1.update(d1, f1)
            tracker2.update(d2, f2)
            gt1 = identity_mgr.assign_global_ids("cam1", tracker1.tracker.tracker.tracks)
            gt2 = identity_mgr.assign_global_ids("cam2", tracker2.tracker.tracker.tracks)
            analytics.update("cam1", gt1)
            analytics.update("cam2", gt2)
            prev_gt1, prev_gt2 = gt1, gt2
        else:
            gt1, gt2 = prev_gt1, prev_gt2

        # --- draw boxes (cheap) ---
        draw1 = [(x1, y1, x2, y2, gid) for x1, y1, x2, y2, _, gid in gt1]
        draw2 = [(x1, y1, x2, y2, gid) for x1, y1, x2, y2, _, gid in gt2]
        f1 = draw_tracks(f1, draw1)
        f2 = draw_tracks(f2, draw2)

        both = stack_cameras(f1, f2)
        both_rgb = cv2.cvtColor(both, cv2.COLOR_BGR2RGB)

        # Encode as JPEG — ~10x smaller than raw numpy, makes st.image much faster
        state["latest_jpeg"] = _encode_jpeg(both_rgb, quality=70)

        # yield CPU so we don't starve the main thread
        time.sleep(0.005)


# ============================================================
# PIPELINE START / UI LOOP
# ============================================================
if st.session_state.pipeline_running:
    SHARED["running"] = True

    # Spawn thread only if one isn't already alive
    if "thread" not in st.session_state or not st.session_state.thread.is_alive():
        SHARED["analytics"] = AnalyticsManager()
        SHARED["latest_jpeg"] = None
        t = threading.Thread(
            target=tracking_worker,
            args=(conf_thresh, skip_frames, SHARED),
            daemon=True,
        )
        st.session_state.thread = t
        t.start()

    st.caption("Pipeline running — streaming frames…")

    # Fast UI polling loop — only reads from SHARED dict, never calls heavy code
    while SHARED.get("running", False):
        jpeg = SHARED.get("latest_jpeg")
        if jpeg is not None:
            video_placeholder.image(jpeg, caption="Cam 1 (Left)  |  Cam 2 (Right)")

        ana = SHARED.get("analytics")
        if ana:
            s = ana.summary()
            metric_unique.metric("Unique Pedestrians", s["unique_people"])
            people = s["people"]
            metric_active.metric("Active Tracks",
                                 sum(1 for p in people if p["dwell_time"] > 0))
            if people:
                df = pd.DataFrame(people)
                df["dwell_time"] = df["dwell_time"].round(1).astype(str) + " s"
                df["cameras"] = df["cameras"].apply(lambda c: ", ".join(c))
                table_placeholder.dataframe(
                    df.sort_values("frames", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        time.sleep(0.05)  # ~20 FPS UI refresh — fast enough, won't overwhelm browser

    st.success("✅ Pipeline finished.")
    st.session_state.pipeline_running = False
else:
    SHARED["running"] = False
    st.info("👈 Click **Start Pipeline** to begin.")