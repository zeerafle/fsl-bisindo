import kagglehub
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

# Download latest version
path = kagglehub.dataset_download("glennleonali/wl-bisindo")

data_path = os.getenv("WLBISINDO_DATA_PATH", "./data/WL-BISINDO")
video_path = os.path.join(data_path, "rgb")

shutil.move(f"{path}/", data_path)
os.rename(f"{data_path}/1", video_path)

print("Path to dataset files:", video_path)
