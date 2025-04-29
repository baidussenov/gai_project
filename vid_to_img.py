import cv2
import os

def video_to_images(video_path, output_folder, prefix='frame', image_format='png'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build filename and save frame as PNG
        filename = os.path.join(output_folder, f"{prefix}_{frame_count:05d}.{image_format}")
        cv2.imwrite(filename, frame)
        frame_count += 1

    cap.release()
    print(f"Finished extracting {frame_count} frames to '{output_folder}'.")

# Example usage
video_path = 'your_video.mp4'
output_folder = 'video_frames'
video_to_images(video_path, output_folder)
