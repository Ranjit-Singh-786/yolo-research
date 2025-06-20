from utils import iou
class SimpleTracker:
    def __init__(self, iou_threshold=0.75):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}  # id: bbox

    def update(self, detections):
        updated_tracks = {}
        used_detections = set()
        matched_ids = set()

        for track_id, track_bbox in self.tracks.items():
            best_iou = 0
            best_match = -1
            for i, det_bbox in enumerate(detections):
                if i in used_detections:
                    continue
                current_iou = iou(track_bbox, det_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match = i

            if best_iou >= self.iou_threshold:
                updated_tracks[track_id] = detections[best_match]
                used_detections.add(best_match)
                matched_ids.add(track_id)

        # Assign new IDs to unmatched detections
        for i, det_bbox in enumerate(detections):
            if i not in used_detections:
                updated_tracks[self.next_id] = det_bbox
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks
