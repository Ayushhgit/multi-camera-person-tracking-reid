import streamlit as st
import time
import pandas as pd
import cv2
import sys
import os

# Add the parent directory (project root) to sys.path so it can find utils, detectors, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video import MOTCamera
from utils.draw import draw_tracks, stack_cameras
from detectors.yolo_detector import YoloDetector
from trackers.deepsort_tracker import DeepSortTracker
from multicam.identity_manager import GlobalIdentityManager
from analytics.stats import AnalyticsManager

st.set_page_config(
    page_title="Multi-Camera Tracking Analytics",
    layout="wide",
)

st.title("👁️ Multi-Camera Pedestrian Analytics")
st.markdown("Live object detection, ReID tracking, and behavior analytics across multiple cameras.")

# --- Session State Config ---
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    start_btn = st.button("▶️ Start Pipeline", use_container_width=True, type="primary")
    stop_btn = st.button("⏹️ Stop/Reset", use_container_width=True)

    st.divider()
    st.subheader("Settings")
    skip_frames = st.slider("Process Every Nth Frame", 1, 10, 3, help="Higher = Faster but less smooth")
    conf_thresh = st.slider("YOLO Confidence", 0.1, 1.0, 0.4)

if start_btn:
    st.session_state.pipeline_running = True
if stop_btn:
    st.session_state.pipeline_running = False

# --- UI Layout ---
col1, col2 = st.columns(2)
unique_placeholder = col1.empty()
active_placeholder = col2.empty()

st.divider()

# Main video display
video_placeholder = st.empty()

st.divider()
st.subheader("📊 Identity Analytics Database")
table_placeholder = st.empty()

# --- Pipeline Execution ---
if st.session_state.pipeline_running:
    
    # Initialize components only once when started
    cam1 = MOTCamera("data/mot17/train/MOT17-02/img1", max_frames=1500, size=(480, 270))
    cam2 = MOTCamera("data/mot17/train/MOT17-04/img1", max_frames=1500, size=(480, 270))
    
    detector = YoloDetector(conf_threshold=conf_thresh)
    tracker1 = DeepSortTracker()
    tracker2 = DeepSortTracker()
    
    identity_manager = GlobalIdentityManager(similarity_threshold=0.85)
    analytics = AnalyticsManager()

    frame_count = 0
    prev_gtracks1, prev_gtracks2 = [], []

    while st.session_state.pipeline_running:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if not ret1 or not ret2:
            st.warning("Reached end of video streams.")
            st.session_state.pipeline_running = False
            break

        frame_count += 1

        # Heavy detection + tracking
        if frame_count % skip_frames == 0:
            dets1 = detector.detect(frame1)
            dets2 = detector.detect(frame2)

            tracks1 = tracker1.update(dets1, frame1)
            tracks2 = tracker2.update(dets2, frame2)

            gtracks1 = identity_manager.assign_global_ids("cam1", tracker1.tracker.tracker.tracks)
            gtracks2 = identity_manager.assign_global_ids("cam2", tracker2.tracker.tracker.tracks)

            analytics.update("cam1", gtracks1)
            analytics.update("cam2", gtracks2)

            prev_gtracks1, prev_gtracks2 = gtracks1, gtracks2
        else:
            gtracks1, gtracks2 = prev_gtracks1, prev_gtracks2

        # Drawing
        draw1 = [(x1, y1, x2, y2, gid) for x1, y1, x2, y2, lid, gid in gtracks1]
        draw2 = [(x1, y1, x2, y2, gid) for x1, y1, x2, y2, lid, gid in gtracks2]

        frame1 = draw_tracks(frame1, draw1)
        frame2 = draw_tracks(frame2, draw2)
        
        # Combine frames and convert BGR (OpenCV) to RGB (Streamlit)
        both = stack_cameras(frame1, frame2)
        both_rgb = cv2.cvtColor(both, cv2.COLOR_BGR2RGB)

        # Update Live Dashboard
        video_placeholder.image(both_rgb, caption="Cam 1 (Left) | Cam 2 (Right)", use_container_width=True)

        summary = analytics.summary()
        unique_people = summary["unique_people"]
        people = summary["people"]
        active_now = sum(1 for p in people if p["dwell_time"] > 0)

        unique_placeholder.metric("Unique Pedestrians Detected", unique_people)
        active_placeholder.metric("Currently Active Tracks", active_now)

        if people:
            df = pd.DataFrame(people)
            # Format dataframe for cleaner display
            df['dwell_time'] = df['dwell_time'].round(1).astype(str) + " s"
            df['cameras'] = df['cameras'].apply(lambda x: ", ".join(x))
            
            table_placeholder.dataframe(
                df.sort_values(by="frames", ascending=False), 
                use_container_width=True,
                hide_index=True
            )
else:
    st.info("👈 Click **Start Pipeline** in the sidebar to begin processing video streams.")