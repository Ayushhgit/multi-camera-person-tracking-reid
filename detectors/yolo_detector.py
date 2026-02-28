from ultralytics import YOLO
import numpy as np 

class YOLODetector:
    """
    YOLOv8 person detector wrapper.

    Returns detections in format:
    [(x1, y1, x2, y2, confidence), ...]
    """

    def __init__(self, model_path: str="yolo8n.pt", conf_threshold:float = 0.4, device: str = "cpu"):
        self.model = YOLO(model_path) 
        self.conf = conf_threshold
        self.device = device

    def detect(self, frame):
        """
        Run YOLO detection on frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            detections: list of (x1,y1,x2,y2,conf)
        """
        if frame is None:
            return []

        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls = int(box.cls[0])

            # YOLO class 0 = person
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append((x1, y1, x2, y2, conf))

        return detections