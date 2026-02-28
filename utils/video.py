import os
import cv2

class MOTCamers:
    """
    Loads MOT17 sequence as a virtual camera stream.
    Reads frames sequentially from img1 folder.
    """

    def __init__(self, img_dir: str, max_frames: int = 500, size=(640, 360)):
        self.img_dir = img_dir
        self.size = size

        self.frames = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        )

        self.max_frames = min(max_frames, len(self.frames))
        self.idx = 0

    def read(self):
        """Read next frame"""
        if self.idx >= self.max_frames:
            return False, None

        img_path = os.path.join(self.img_dir, self.frames[self.idx])
        frame = cv2.imread(img_path)

        if frame is None:
            return False, None

        frame = cv2.resize(frame, self.size)
        self.idx += 1
        return True, frame 

    def reset(self):
        """Restart sequence"""
        self.idx = 0
