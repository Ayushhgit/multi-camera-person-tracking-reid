# 👁️ Multi-Camera Pedestrian Tracking & Identity Fusion

A robust multi-camera pedestrian tracking system that maintains persistent individual identities across multiple independent video streams. 

## 🚀 Overview

This project implements an end-to-end pipeline for tracking people dynamically across different camera views. It processes each camera stream independently for object detection and local tracking, and then uses a global **Identity Manager** to fuse identities across cameras using appearance embeddings and cosine similarity (Re-Identification or ReID).

Key capabilities:
- **Real-Time Detection:** Powered by `YOLOv8-nano` for high-speed, lightweight pedestrian detection.
- **Local Tracking:** Uses `DeepSORT` for frame-by-frame object tracking and local ID assignment.
- **Cross-Camera ReID:** Maintains a persistent global identity for individuals as they move between completely different camera angles.
- **Pedestrian Analytics:** Generates actionable metrics such as unique pedestrian counts, active tracks, and dwell time.

## 🛠️ Technology Stack

- **Computer Vision:** `ultralytics` (YOLOv8), `opencv-python`, `deep-sort-realtime`
- **Data & Analytics:** `numpy`, `pandas`, `torch`
- **Dashboard:** `streamlit`
- **Environment Management:** `uv`

## 🏃‍♂️ Getting Started

### Prerequisites

Ensure you have [uv](https://docs.astral.sh/uv/) installed to manage the Python environment.

### Running the Application

To launch the live Web Dashboard:

```bash
uv run streamlit run dashboard/app.py
```

Once the dashboard opens in your browser, click **▶️ Start Pipeline** in the sidebar to begin processing the multi-camera streams.

## 📊 Analytics Dashboard

The included Streamlit dashboard provides a live, unified view of the tracking pipeline, including:
- Side-by-side video feeds with unified tracking IDs.
- Live metrics: **Unique Pedestrians Detected** and **Currently Active Tracks**.
- An active analytics table detailing global IDs, total dwell time, and camera visitation history.

## 📂 Project Structure
- `dashboard/`: Streamlit web application.
- `detectors/`: YOLOv8 detection logic.
- `trackers/`: DeepSORT tracking implementation.
- `multicam/`: Global Identity Manager for cross-camera ReID.
- `analytics/`: Analytics and statistics generation.
- `data/`: Directory for video sequences (e.g., MOT17).

---
*Built as a computer vision exploration into spatial awareness and cross-camera Re-Identification.*