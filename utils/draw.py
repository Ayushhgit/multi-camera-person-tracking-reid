import cv2

def draw_detections(frame, detections, class_names, colors=(0,255,0)):
    """
    Draw bounding boxes from detector.
    detections: list of (x1,y1,x2,y2,conf)
    """
    for x1,y1,x2,y2,conf in detections:
        cv2.rectangle(frame, (x1,y1),(x2,y2), colors, 2)
    return frame
    
def draw_tracks(frame, tracks, color=(255, 0, 0)):
    """
    Draw tracking IDs.
    tracks: list of (x1,y1,x2,y2,track_id)
    """
    for x1, y1, x2, y2, tid in tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return frame


def stack_cameras(frame1, frame2):
    """Side-by-side camera view"""
    return cv2.hconcat([frame1, frame2])