import cv2

def simulate_webcam_from_video(video_path):
    """
    Simulates a webcam feed by playing a video file.

    Args:
        video_path (str): The path to the video file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break

        cv2.imshow('Simulated Webcam Feed', frame)

        # Press 'q' to exit the simulated feed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'your_video.mp4' with the path to your video file
    video_file = 'your_video.mp4'
    simulate_webcam_from_video(video_file)