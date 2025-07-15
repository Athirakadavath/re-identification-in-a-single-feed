import numpy as np
from collections import defaultdict

def iou(boxA, boxB):
    # Compute Intersection over Union between two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class PlayerTracker:
    def __init__(self, iou_threshold=0.3, max_lost=20):
        self.next_id = 1
        self.tracks = dict()
        self.lost = dict()  # lost counter
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, frame, detections):
        updated_tracks = dict()
        # Assign detections to existing tracks based on IoU
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det = None
            for det in detections:
                iou_score = iou(track['bbox'], det[:4])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = det
            if best_iou > self.iou_threshold:
                updated_tracks[track_id] = {
                    'bbox': best_det[:4],
                    'conf': best_det[4],
                    'lost': 0
                }
                detections.remove(best_det)
            else:
                # Not matched, increase lost count
                track['lost'] += 1
                if track['lost'] < self.max_lost:
                    updated_tracks[track_id] = track
        # Add new detections as new tracks
        for det in detections:
            updated_tracks[self.next_id] = {
                'bbox': det[:4],
                'conf': det[4],
                'lost': 0
            }
            self.next_id += 1
        # Remove tracks too lost
        self.tracks = {tid: t for tid, t in updated_tracks.items() if t['lost'] < self.max_lost}
        # Prepare output
        output = []
        for tid, t in self.tracks.items():
            output.append({'id': tid, 'bbox': t['bbox'], 'conf': t['conf']})
        return output