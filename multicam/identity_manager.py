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

    MAX_GALLERY_SIZE = 50  # max embeddings stored per person

    def __init__(self, similarity_threshold=0.85):
        self.sim_threshold = similarity_threshold

        # gallery: global_id -> list of embedding vectors
        self.gallery = {}

        # mapping: (cam_id, local_id) -> global_id
        self.local_to_global = {}

        self.next_global_id = 1

    def _get_mean_embedding(self, gid):
        """Average embedding for a global identity."""
        embeddings = self.gallery[gid]
        return np.mean(embeddings, axis=0)

    def _match_embedding(self, embedding):
        """
        Find best matching global identity for embedding.
        Compares against the mean embedding of each gallery entry.
        """
        best_gid = None
        best_sim = 0.0

        for gid in self.gallery:
            mean_emb = self._get_mean_embedding(gid)
            sim = cosine_similarity(embedding, mean_emb)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        if best_sim >= self.sim_threshold:
            return best_gid

        return None

    def _add_to_gallery(self, gid, embedding):
        """Add embedding to gallery, keeping at most MAX_GALLERY_SIZE."""
        if gid not in self.gallery:
            self.gallery[gid] = [embedding]
        else:
            self.gallery[gid].append(embedding)
            if len(self.gallery[gid]) > self.MAX_GALLERY_SIZE:
                self.gallery[gid].pop(0)

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

                # Update gallery with latest embedding for better matching
                if track.features and len(track.features) > 0:
                    self._add_to_gallery(global_id, track.features[-1])
            else:
                # Get appearance embedding from DeepSORT
                if track.features is None or len(track.features) == 0:
                    continue

                emb = track.features[-1]

                matched_gid = self._match_embedding(emb)

                if matched_gid is not None:
                    global_id = matched_gid
                    self._add_to_gallery(global_id, emb)
                else:
                    global_id = self.next_global_id
                    self._add_to_gallery(global_id, emb)
                    self.next_global_id += 1

                self.local_to_global[key] = global_id

            l, t, r, b = map(int, track.to_ltrb())

            results.append((l, t, r, b, local_id, global_id))

        return results