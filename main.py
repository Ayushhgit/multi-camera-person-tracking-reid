import cv2

from utils.video import MOTCamera
from utils.draw import draw_tracks, stack_cameras
from detectors.yolo_detector import YoloDetector
from trackers.deepsort_tracker import DeepSortTracker
from multicam.identity_manager import GlobalIdentityManager
from analytics.stats import AnalyticsManager


# CONFIG 
CAM1_PATH = "data/mot17/train/MOT17-02/img1"
CAM2_PATH = "data/mot17/train/MOT17-04/img1"

MAX_FRAMES = 1500
WINDOW_NAME = "Multi-Camera Tracking"
SKIP_FRAMES = 3  # only run detection every Nth frame
CAM_SIZE = (480, 270)  # smaller = faster


# INIT 
cam1 = MOTCamera(CAM1_PATH, max_frames=MAX_FRAMES, size=CAM_SIZE)
cam2 = MOTCamera(CAM2_PATH, max_frames=MAX_FRAMES, size=CAM_SIZE)

detector = YoloDetector()
tracker1 = DeepSortTracker()
tracker2 = DeepSortTracker()

identity_manager = GlobalIdentityManager()
analytics = AnalyticsManager()


# LOOP 
frame_count = 0
prev_gtracks1, prev_gtracks2 = [], []

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        break

    frame_count += 1

    # Only run heavy detection + tracking every SKIP_FRAMES frames
    if frame_count % SKIP_FRAMES == 0:
        # DETECTION 
        dets1 = detector.detect(frame1)
        dets2 = detector.detect(frame2)

        # TRACKING 
        tracks1 = tracker1.update(dets1, frame1)
        tracks2 = tracker2.update(dets2, frame2)

        # MULTICAM GLOBAL IDS 
        gtracks1 = identity_manager.assign_global_ids("cam1", tracker1.tracker.tracker.tracks)
        gtracks2 = identity_manager.assign_global_ids("cam2", tracker2.tracker.tracker.tracks)

        # ANALYTICS 
        analytics.update("cam1", gtracks1)
        analytics.update("cam2", gtracks2)

        prev_gtracks1, prev_gtracks2 = gtracks1, gtracks2
    else:
        gtracks1, gtracks2 = prev_gtracks1, prev_gtracks2

    # DRAW GLOBAL IDS 
    draw1 = [(x1, y1, x2, y2, gid) for x1, y1, x2, y2, lid, gid in gtracks1]
    draw2 = [(x1, y1, x2, y2, gid) for x1, y1, x2, y2, lid, gid in gtracks2]

    frame1 = draw_tracks(frame1, draw1)
    frame2 = draw_tracks(frame2, draw2)

    # STACK CAMERAS 
    both = stack_cameras(frame1, frame2)

    # OVERLAY STATS 
    unique_people = analytics.unique_people()
    cv2.putText(
        both,
        f"Unique People: {unique_people}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # SHOW 
    cv2.imshow(WINDOW_NAME, both)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cv2.destroyAllWindows()