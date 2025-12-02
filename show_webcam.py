#!/usr/bin/env python3
import cv2
import argparse

def main(device_index: int = 0):
    # open camera (use CAP_DSHOW on Windows if available to reduce latency)
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Cannot open camera {device_index}")
        return

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Webcam", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):  # ESC or 'q' to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Display webcam video with OpenCV")
    p.add_argument("-d", "--device", type=int, default=0, help="camera device index")
    args = p.parse_args()
    main(args.device)