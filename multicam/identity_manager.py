import numpy as np


def cosine_similarity(a, b):
    """Cosine similarity between two feature vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


class GlobalIdentityManager:
    """
    Fuses identities across multiple cameras using appearance embeddings.
    Maintains mapping:
        (camera_id, local_track_id) → global_id
    """

    def __init__(self, similarity_threshold=0.7):
        self.sim_threshold = similarity_threshold

        # gallery: global_id -> embedding vector
        self.gallery = {}

        # mapping: (cam_id, local_id) -> global_id
        self.local_to_global = {}

        self.next_global_id = 1

    def _match_embedding(self, embedding):
        """
        Find best matching global identity for embedding.
        """
        best_gid = None
        best_sim = 0.0

        for gid, g_emb in self.gallery.items():
            sim = cosine_similarity(embedding, g_emb)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_sim >= self.sim_threshold:
            return best_gid

        return None

    def assign_global_ids(self, camera_id, tracks):
        """
        Assign global IDs to tracks from one camera.
        Args:
            camera_id: int or str
            tracks: list of DeepSORT track objects
        Returns:
            results: [(x1,y1,x2,y2,local_id,global_id)]
        """

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            local_id = track.track_id

            # Already mapped
            key = (camera_id, local_id)
            if key in self.local_to_global:
                global_id = self.local_to_global[key]
            else:
                # Get appearance embedding from DeepSORT
                if track.features is None or len(track.features) == 0:
                    continue

                emb = track.features[-1]

                matched_gid = self._match_embedding(emb)

                if matched_gid is not None:
                    global_id = matched_gid
                else:
                    global_id = self.next_global_id
                    self.gallery[global_id] = emb
                    self.next_global_id += 1

                self.local_to_global[key] = global_id

            l, t, r, b = map(int, track.to_ltrb())

            results.append((l, t, r, b, local_id, global_id))

        return results