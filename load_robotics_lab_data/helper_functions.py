import cv2 
import os
import tarfile
from pathlib import Path

def mp4_to_jpg(video_path):
    video_name = Path(video_path).stem  
    output_dir = Path('frames') / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        filename = f'frames/{video_name}/{count:04d}.jpg'
        cv2.imwrite(filename, image)
        success, image = vidcap.read()
        count += 1

def jpg_to_zip(video_name):
    frame_dir = Path('frames') / video_name
    with tarfile.open(f'../data/{video_name}.tar.gz', 'w:gz') as tar:
        for filename in os.listdir(frame_dir):
            tar.add(os.path.join(frame_dir, filename), arcname=filename)