import argparse
import cv2
import os
from tracker import PlayerTracker
from my_utils import draw_boxes
from ultralytics import YOLO

def main(args):
    # Load YOLO model
    model = YOLO(args.weights)
    tracker = PlayerTracker()

    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.mp4', fourcc, out_fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLO Inference
        results = model(frame)
        # Ultralytics YOLO results output format:
        # results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls
        players = []
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())
            if cls == 0:  # Assuming class 0 is 'player'
                players.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        # Update tracker
        tracked = tracker.update(frame, players)
        # Draw bounding boxes and IDs
        draw_boxes(frame, tracked)
        out.write(frame)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames...", end='\r')
    cap.release()
    out.release()
    print("Done! Output saved as output.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/15sec_input_720p.mp4", help="Path to input video")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to YOLO model weights")
    args = parser.parse_args()
    main(args)