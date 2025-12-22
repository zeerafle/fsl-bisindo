import json
import multiprocessing
import os
import urllib

import mediapipe as mp
import mediapipe_extract
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

METADATA_URL = "https://raw.githubusercontent.com/AceKinnn/WL-BISINDO/refs/heads/main/data_structuring/SI_split_metadata.json"


def load_metadata(url=METADATA_URL):
    with urllib.request.urlopen(url) as response:
        data = response.read()
        metadata = json.loads(data)
    return metadata


def process_video(video_info):
    video_path, output_path = video_info

    if os.path.exists(output_path) or os.path.exists(video_path):
        return

    try:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=2
        )

        frames = mediapipe_extract.load_frames_from_video(video_path)

        if len(frames) == 0:
            return

        keypoints, confs = mediapipe_extract.get_holistic_keypoints(
            frames, holistic=holistic
        )
        np.save(output_path, keypoints)
    except Exception as e:
        # Print error but don't stop the whole process
        print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    DATA_PATH = os.getenv("WLBISINDO_DATA_PATH", "./data/WL-BISINDO")
    RGB_DATA_PATH = os.path.join(DATA_PATH, "rgb")
    OUTPUT_DIR = os.path.join(DATA_PATH, "keypoints")

    metadata = load_metadata()
    tasks = []
    for item in metadata:
        for instance in item["instances"]:
            video_id = instance["video_id"]
            video_path = os.path.join(DATA_PATH, video_id + ".mp4")
            output_path = os.path.join(OUTPUT_DIR, video_id + ".npy")
            tasks.append((video_path, output_path))

    # Run in parallel
    n_cores = multiprocessing.cpu_count()
    print(f"Processing {len(tasks)} videos using {n_cores} cores...")

    # Use joblib to parallelize
    Parallel(n_jobs=n_cores, backend="loky")(
        delayed(process_video)(task) for task in tqdm(tasks)
    )
