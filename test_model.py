import argparse
import os
import threading
import time

import cv2
import torch
from ultralytics import YOLO


class LiveStreamCapture:
    """Threaded video capture to ensure the OpenCV buffer is cleared."""
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified YOLO Boxing Tracker.")
    parser.add_argument(
        "--weights",
        default="runs/detect/train-6/weights/best.pt", 
        help="Path to your custom YOLO weights.",
    )
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--width", type=int, default=640, help="Camera width.")
    parser.add_argument("--height", type=int, default=480, help="Camera height.")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold.")
    parser.add_argument("--cooldown", type=float, default=0.4, help="Cooldown between punches.")
    args = parser.parse_args()

    # --- SETUP ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights).to(device)

    raw_cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap = LiveStreamCapture(raw_cap)

    # --- STATE ---
    total_punches = 0
    last_punch_time = 0
    prev_time = time.time()

    print("[INFO] Tracker running. Press 'q' to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                continue

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            results = model(
                frame, 
                stream=True, 
                device=device,
                half=(device != "cpu"), 
                imgsz=args.width,
                verbose=False 
            )
            
            punch_detected = False
            for r in results:
                # OPTIONAL: Uncomment the line below if you want to see the boxes
                # frame = r.plot() 
                
                for box in r.boxes:
                    if int(box.cls[0]) == 0 and box.conf[0] > args.conf:
                        punch_detected = True

            # Logic
            if punch_detected and (curr_time - last_punch_time) > args.cooldown:
                total_punches += 1
                last_punch_time = curr_time

            # --- MINIMAL UI ---
            # Clean black bar or simple text in the corner
            cv2.putText(frame, f"PUNCHES: {total_punches}", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {int(fps)}", (20, args.height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Boxing Tracker", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()