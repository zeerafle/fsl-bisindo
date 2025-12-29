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

5. Download WL-BISINDO extracted keypoints from github release.

   ```bash
   uv run tools/download_release_keypoints.py
   ```

   Optionally, you can download the raw video files and extract the keypoints yourself by running:

   ```bash
   uv run tools/download_bisindo.py
   uv run tools/extract_keypoints.py
   ```

   The video files will be available at `./data/WL-BISINDO/rgb` and the extracted keypoints will be at `./data/WL-BISINDO/keypoints`.

6. Download pretrained weights for the backbones.

   ```bash
   uv run tools/download_weights.py
   ```

   It automatically modifies the config files that comes with the weights to point to the correct dataset path and some adjustment for running feature extraction. The modified config files are saved in `./configs/backbones/`.

## Few-Shot Training (Recommended)

Train ProtoNet with a pretrained encoder baked into the model. The encoder can be frozen (default) or fine-tuned.

### Basic Training

```bash
# Train with default settings (AUTSL encoder, frozen)
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml

# Train with a different encoder (CSL or LSA64)
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
  --encoder configs/fewshot/encoders/csl_slgcn.yaml

uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
  --encoder configs/fewshot/encoders/lsa64_slgcn.yaml
```

### Fine-tuning the Encoder

```bash
# Unfreeze encoder and train with lower learning rate
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
  --unfreeze_encoder --lr 0.0001
```

### Custom Few-Shot Settings

```bash
# Different N-way K-shot settings
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
  --n_way 5 --k_shot 1 --n_query 15
```

### Run Encoder Comparison

Compare all three encoders automatically:

```bash
# Full comparison
uv run tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml

# Quick test with reduced epochs
uv run tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml \
  --n_epochs 5 --no_wandb

# Fine-tune all encoders
uv run tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml \
  --unfreeze_encoder --lr 0.0001
```

### Configuration

The main configuration is in `configs/fewshot/train_protonet.yaml`:

- **encoder.backbone_cfg**: Path to encoder config (autsl, csl, or lsa64)
- **protonet.freeze_encoder**: Whether to freeze encoder weights
- **protonet.distance**: Distance metric ("euclidean" or "cosine")
- **fewshot.n_way**: Number of classes per episode
- **fewshot.k_shot**: Support samples per class
- **fewshot.n_query**: Query samples per class
- **split**: Base/val/test class assignments

## Feature Extraction (Legacy Method)

If you prefer to pre-extract features and train separately:

```bash
uv run tools/extract_features.py \
  --backbone_cfg configs/backbones/autsl_slgcn.yaml \
  --data_cfg configs/data/wlbisindo_keypoints.yaml \
  --out_dir data/features \
  --wandb_project wl-bisindo-fsl \
  --wandb_group feature_extraction_v1
```

Feature extraction process doesn't need GPU to run, though running with CPU-only machine will take around 3 minutes each.

## Project Structure

```
fsl-bisindo/
├── configs/
│   ├── fewshot/              # Few-shot training configs
│   │   ├── train_protonet.yaml
│   │   └── encoders/         # Encoder-specific configs
│   │       ├── autsl_slgcn.yaml
│   │       ├── csl_slgcn.yaml
│   │       └── lsa64_slgcn.yaml
│   ├── backbones/            # Legacy backbone configs
│   └── data/                 # Data configs
├── src/fsl_bisindo/
│   ├── data/                 # Dataset and samplers
│   │   ├── wlbisindo_dataset.py
│   │   └── fsl_sampler.py
│   ├── engine/               # Training engine
│   │   └── trainer.py
│   └── models/               # Model implementations
│       ├── protonet.py       # ProtoNet with integrated encoder
│       └── load_pretrained.py
├── tools/                    # CLI scripts
│   ├── train_fewshot.py
│   ├── run_encoder_comparison.py
│   └── extract_features.py
└── checkpoints/              # Pretrained weights
```
