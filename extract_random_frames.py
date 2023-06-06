import os
import cv2
import random

def extract_random_frames(video_path, output_folder, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    selected_frames = sorted(random.sample(range(total_frames), num_frames))

    for frame_idx, target_frame in enumerate(selected_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if ret:
            output_file = os.path.join(output_folder, f"frame2_{frame_idx+1}.png")
            cv2.imwrite(output_file, frame)
        else:
            print(f"Error reading frame {target_frame}")

    cap.release()

video_path = "../Video/pexels-kelly-15100724-1920x1080-24fps.mp4"
output_folder = "Data/HighResolution"

extract_random_frames(video_path, output_folder)