import cv2
import os

# Output directory
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

def frame_processing(file_path, target_fps=2):
    cap = cv2.VideoCapture(file_path) 

    # Get the original FPS of the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # fallback default if reading FPS fails

    frame_interval = int(video_fps / target_fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f'{output_dir}/frame_{saved_count:04d}.jpg'
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
