from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """
    DeepSORT tracker wrapper (latest API compatible)
    Input detections:
        [(x1, y1, x2, y2, conf), ...]
    Output tracks:
        [(x1, y1, x2, y2, track_id), ...]
    """

    def __init__(self, max_age: int = 30, n_init: int = 3, max_iou_distance: float = 0.7, embedder: str = "mobilenet"):
        """
        Args:
            max_age: frames to keep lost tracks
            n_init: frames before confirming track
            max_iou_distance: association threshold
            embedder: appearance model ("mobilenet" = CPU friendly)
        """

        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance, embedder=embedder)

    def update(self, detections, frame):
        """
        Update tracker with detections.
        Args:
            detections: [(x1,y1,x2,y2,conf), ...]
            frame: BGR image
        Returns:
            tracks: [(x1,y1,x2,y2,track_id), ...]
        """

        # Convert to DeepSORT format: ([l,t,w,h], conf, class)
        ds_detections = []
        for x1, y1, x2, y2, conf in detections:
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, "person"))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            track_id = track.track_id

            results.append((l, t, r, b, track_id))

        return results