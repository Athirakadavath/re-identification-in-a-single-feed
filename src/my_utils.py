import cv2

def draw_boxes(frame, tracked):
    for det in tracked:
        x1, y1, x2, y2 = map(int, det['bbox'])
        player_id = det['id']
        conf = det['conf']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f"Player {player_id}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)