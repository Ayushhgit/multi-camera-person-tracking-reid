import time

class AnalyticsManager:
    """
    Maintains per-person analytics across cameras.
    Tracks:
        - first_seen time
        - last_seen time
        - frames seen
        - cameras visited
        - trajectory points
    """

    def __init__(self):
        self.people = {}
        self.total_frames = 0

    def update(self, camera_id, tracks):
        """
        Update analytics with tracks from one camera.
        Args:
            camera_id: str/int
            tracks: [(x1,y1,x2,y2,local_id,global_id)]
        """

        self.total_frames += 1
        now = time.time()

        for x1, y1, x2, y2, local_id, global_id in tracks:

            if global_id not in self.people:
                self.people[global_id] = {
                    "first_seen": now,
                    "last_seen": now,
                    "frames": 1,
                    "cameras": set([camera_id]),
                    "trajectory": [],
                }
            else:
                p = self.people[global_id]
                p["last_seen"] = now
                p["frames"] += 1
                p["cameras"].add(camera_id)

            # trajectory = bbox center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self.people[global_id]["trajectory"].append((cx, cy))

    def unique_people(self):
        """Total unique global identities"""
        return len(self.people)

    def dwell_time(self, global_id):
        """Seconds person was visible"""
        p = self.people.get(global_id)
        if not p:
            return 0.0
        return p["last_seen"] - p["first_seen"]

    def camera_count(self, global_id):
        """How many cameras person appeared in"""
        p = self.people.get(global_id)
        if not p:
            return 0
        return len(p["cameras"])

    def summary(self):
        """
        Returns analytics summary dict.
        """
        data = {
            "unique_people": self.unique_people(),
            "people": [],
        }

        for gid, p in self.people.items():
            data["people"].append(
                {
                    "global_id": gid,
                    "dwell_time": p["last_seen"] - p["first_seen"],
                    "frames": p["frames"],
                    "cameras": list(p["cameras"]),
                }
            )

        return data