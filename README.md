# FSL Bisindo

Few-shot learning implementation for sign language recognition on WL-BISINDO dataset.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/zeerafle/fsl-bisindo.git
   cd fsl-bisindo
   ```

2. Have uv installed and run

   ```bash
   uv sync
   ```

3. Create a `.env` by copying the `.env.example`:

    ```bash
    cp .env.example .env
    ```

4. Set the `WLBISINDO_DATA_PATH` variable in the `.env` file to point to your WL-BISINDO dataset directory. Optionally, set your Weights & Biases API key in the `.env` file for experiment tracking.

5. Download WL-BISINDO dataset and extract the keypoints.

   ```bash
   uv run tools/download_bisindo.py
   uv run tools/extract_keypoints.py
   ```

   The video files will be available at `./data/WL-BISINDO/rgb` and the extracted keypoints will be at `./data/WL-BISINDO/keypoints`.

