from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from fsl_bisindo.engine.extractor import extract_features, pad_or_truncate_frames
from fsl_bisindo.models.load_pretrained import build_backbone_from_cfg
from fsl_bisindo.utils.wandb_utils import log_artifact, wandb_init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_cfg", type=str, required=True)
    p.add_argument("--data_cfg", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--wandb_project", type=str, required=True)
    p.add_argument("--wandb_group", type=str, required=True)
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing transform (matches notebook feature_extraction.ipynb)
# ---------------------------------------------------------------------------
def make_preprocess_transform(window_size: int | None = None):
    """
    Create a transform that:
    1. Converts 543 keypoints -> 27 (Body+Hands selection)
    2. Normalizes using OpenHands presets
    3. Returns [C, T, V] tensor

    This matches the notebook's preprocess_keypoints() function.
    """
    from openhands.datasets.pose_transforms import (
        CenterAndScaleNormalize,
        PoseSelect,
    )

    pose_select = PoseSelect(preset="mediapipe_holistic_minimal_27")
    normalizer = CenterAndScaleNormalize(
        reference_points_preset="shoulder_mediapipe_holistic_minimal_27"
    )

    def transform(arr: np.ndarray) -> torch.Tensor:
        """
        Args:
            arr: numpy array [T, 543, C] from MediaPipe

        Returns:
            torch.Tensor [C, T, V] where V=27
        """
        # Ensure 3 channels (X, Y, Z)
        if arr.shape[-1] == 2:
            arr = np.concatenate([arr, np.zeros((*arr.shape[:-1], 1))], axis=-1)

        # 543 -> 75: Select Body(33) + LeftHand(21) + RightHand(21)
        body = arr[:, :33, :]
        lh = arr[:, 501:522, :]
        rh = arr[:, 522:543, :]
        arr_75 = np.concatenate([body, lh, rh], axis=1)  # [T, 75, 3]

        # Convert to tensor [C, T, V] - only use X, Y (2 channels)
        tensor = torch.from_numpy(arr_75).float().permute(2, 0, 1)[:2]  # [2, T, 75]
        sample = {"frames": tensor}

        # Apply OpenHands transforms: 75 -> 27 keypoints
        sample = pose_select(sample)
        sample = normalizer(sample)

        frames = sample["frames"]  # [C, T, 27]

        # Pad/truncate if window_size specified
        if window_size is not None:
            frames = pad_or_truncate_frames(frames, window_size)

        return frames

    return transform


# ---------------------------------------------------------------------------
# Custom collate for variable-length sequences
# ---------------------------------------------------------------------------
def varlen_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate that handles variable-length frame tensors.

    Pads all frames to the max length in the batch, then stacks.
    """
    # Find max temporal length
    max_t = max(item["frames"].shape[1] for item in batch)  # frames are [C, T, V]

    # Pad and stack frames
    padded_frames = []
    for item in batch:
        frames = item["frames"]  # [C, T, V]
        frames = pad_or_truncate_frames(frames, max_t)
        padded_frames.append(frames)

    collated: dict[str, Any] = {
        "frames": torch.stack(padded_frames, dim=0),  # [B, C, T, V]
    }

    # Collate other fields
    for key in batch[0].keys():
        if key == "frames":
            continue
        vals = [item[key] for item in batch]
        if torch.is_tensor(vals[0]):
            collated[key] = torch.stack(vals, dim=0)
        else:
            collated[key] = vals

    return collated


def build_wlbisindo_loader(
    data_cfg: DictConfig | ListConfig,
    batch_size: int,
    num_workers: int,
    window_size: int | None = None,
) -> DataLoader:
    """
    Build DataLoader for WL-BISINDO keypoints.

    Applies preprocessing transform and handles variable-length sequences.
    """
    from fsl_bisindo.data.wlbisindo_dataset import WLBisindoKeypointsDataset

    transform = make_preprocess_transform(window_size=window_size)

    ds = WLBisindoKeypointsDataset(
        keypoints_root=data_cfg.paths.keypoints_root,
        transform=transform,
        allow_unparsed=False,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=varlen_collate_fn,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # load W&B API key from .env
    args = parse_args()
    backbone_cfg = OmegaConf.load(args.backbone_cfg)  # type: ignore
    data_cfg = OmegaConf.load(args.data_cfg)  # type: ignore
    # split_cfg = load_yaml(args.split_cfg)  # type: ignore

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a single merged cfg for logging
    cfg = {
        "backbone": backbone_cfg,
        "data": data_cfg,
        "extract": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
        },
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "job_type": "extract",
            "name": f"extract/{backbone_cfg.name}",
            "tags": ["extract", "slgcn", backbone_cfg.name],
        },
    }

    run = wandb_init(cfg, job_type="extract")

    # 1) model (encoder)
    encoder = build_backbone_from_cfg(
        cfg["backbone"], device=args.device, return_encoder=True
    )

    # 2) data loader (WL-BISINDO keypoints)
    window_size = OmegaConf.select(backbone_cfg, "model.window_size", default=None)
    loader = build_wlbisindo_loader(
        data_cfg, args.batch_size, args.num_workers, window_size=window_size
    )

    # 3) extract and save
    out_path = out_dir / f"features_{backbone_cfg.name}.pt"

    result = extract_features(
        encoder=encoder,
        dataloader=loader,
        out_path=out_path,
        device=args.device,
        expected_t=window_size,
        extra_meta={
            "backbone_name": backbone_cfg.name,
        },
    )

    # 4) log artifact (features)
    art_name = f"features-{backbone_cfg.name}"
    logged = log_artifact(
        run,
        file_or_dir=result.out_path,
        name=art_name,
        type="features",
        metadata={
            "backbone": backbone_cfg.name,
            "feature_dim": result.feature_dim,
            "num_samples": result.num_samples,
        },
        aliases=["latest"],
    )

    # Optional: also log quick scalars for searchability
    run.log(
        {
            "extract/num_samples": result.num_samples,
            "extract/feature_dim": result.feature_dim,
            "extract/skipped": result.skipped,
            "artifacts/features_ref": logged.qualified,
        }
    )

    run.finish()
