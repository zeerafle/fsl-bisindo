from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass
class FeatureExtractionOutput:
    out_path: str
    num_samples: int
    feature_dim: int
    skipped: int
    meta: dict[str, Any]


def _to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors to device."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def pad_or_truncate_frames(frames: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    Pad or truncate temporal dimension to target_t.

    Args:
        frames: Tensor [C, T, V] or [B, C, T, V]
        target_t: target temporal length

    Returns:
        Tensor with T == target_t
    """
    ndim = frames.dim()
    if ndim == 3:
        # [C, T, V]
        c, t, v = frames.shape
        t_dim = 1
    elif ndim == 4:
        # [B, C, T, V]
        b, c, t, v = frames.shape
        t_dim = 2
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {ndim}D")

    if t == target_t:
        return frames
    if t > target_t:
        # Truncate
        return frames.narrow(t_dim, 0, target_t)

    # Pad
    pad_size = list(frames.shape)
    pad_size[t_dim] = target_t - t
    pad_tensor = torch.zeros(pad_size, dtype=frames.dtype, device=frames.device)
    return torch.cat([frames, pad_tensor], dim=t_dim)


def default_batch_to_model_input(batch: dict[str, Any]) -> torch.Tensor:
    """
    Convert batch["frames"] to model input [B, C, T, V].

    Handles:
      - Single tensor [C, T, V] -> unsqueeze to [1, C, T, V]
      - Batched tensor [B, C, T, V] -> pass through
      - List of tensors -> stack (requires same shape)
    """
    x = batch["frames"]

    if isinstance(x, (list, tuple)):
        # List of [C, T, V] tensors (from custom collate)
        return torch.stack(x, dim=0)

    if not torch.is_tensor(x):
        raise TypeError(f"batch['frames'] must be Tensor or list, got {type(x)}")

    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x

    raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")


def _collect_field(batch: dict, singular: str, plural: str) -> list:
    """Extract a field from batch, handling both singular and plural keys."""
    if plural in batch:
        val = batch[plural]
    elif singular in batch:
        val = batch[singular]
    else:
        return []

    # Convert to list
    if torch.is_tensor(val):
        return val.detach().cpu().tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


@torch.inference_mode()
def extract_features(
    encoder: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    out_path: str | Path,
    device: str | torch.device = "cuda",
    *,
    batch_to_model_input=default_batch_to_model_input,
    feature_postprocess=None,
    expected_t: int | None = None,
    save_dtype: torch.dtype = torch.float32,
    strict_shapes: bool = False,
    progress: bool = True,
    extra_meta: dict[str, Any] | None = None,
) -> FeatureExtractionOutput:
    """
    Generic feature extraction.

    Args:
        encoder: Model that maps [B,C,T,V] -> [B,D]
        dataloader: Yields dict batches with "frames" key
        out_path: Where to save the .pt file
        device: Device to run inference on
        batch_to_model_input: Function to convert batch -> model input tensor
        feature_postprocess: Optional function to postprocess features
        expected_t: If set, pad/truncate temporal dim to this length
        save_dtype: dtype for saved features
        strict_shapes: If True, raise on encoder errors; else skip sample
        progress: Show tqdm progress bar
        extra_meta: Additional metadata to save

    Returns:
        FeatureExtractionOutput with paths and statistics
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    encoder = encoder.to(device).eval()

    # Accumulators
    all_feats: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_video_ids: list[str] = []
    all_paths: list[str] = []
    extras: dict[str, list[Any]] = {}

    skipped = 0
    n_batches = 0

    # Optional tqdm
    it: Iterable = dataloader
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(dataloader, desc="Extracting features")
        except ImportError:
            pass

    # Known metadata fields to handle specially
    KNOWN_FIELDS = {
        "frames",
        "label",
        "labels",
        "video_id",
        "video_ids",
        "path",
        "paths",
    }

    for batch in it:
        n_batches += 1
        batch = _to_device(batch, device)

        # Build model input [B, C, T, V]
        x = batch_to_model_input(batch)
        if expected_t is not None:
            x = pad_or_truncate_frames(x, expected_t)

        # Forward pass
        try:
            z = encoder(x)
        except Exception:
            if strict_shapes:
                raise
            skipped += x.shape[0]
            continue

        # Flatten to [B, D] and move to CPU
        z = z.reshape(z.size(0), -1)
        if feature_postprocess:
            z = feature_postprocess(z)
        all_feats.append(z.detach().cpu().to(save_dtype))

        # Collect standard metadata fields
        all_labels.extend([int(v) for v in _collect_field(batch, "label", "labels")])
        all_video_ids.extend(
            [str(v) for v in _collect_field(batch, "video_id", "video_ids")]
        )
        all_paths.extend([str(v) for v in _collect_field(batch, "path", "paths")])

        # Collect extra fields (non-tensor, non-standard)
        for k, v in batch.items():
            if k in KNOWN_FIELDS or torch.is_tensor(v):
                continue
            vals = [v] if isinstance(v, (str, int, float)) else list(v)
            extras.setdefault(k, []).extend(vals)

    # Build output
    if all_feats:
        features = torch.cat(all_feats, dim=0)
        num_samples, feature_dim = features.shape
    else:
        features = torch.empty((0, 0), dtype=save_dtype)
        num_samples = feature_dim = 0

    blob: dict[str, Any] = {
        "features": features,
        "meta": {
            "num_batches": n_batches,
            "skipped": skipped,
            "device": str(device),
            **(extra_meta or {}),
        },
    }

    # Add optional fields only if non-empty
    if all_labels:
        blob["labels"] = torch.tensor(all_labels, dtype=torch.long)
    if all_video_ids:
        blob["video_ids"] = all_video_ids
    if all_paths:
        blob["paths"] = all_paths
    if extras:
        blob["extras"] = extras

    torch.save(blob, str(out_path))

    return FeatureExtractionOutput(
        out_path=str(out_path),
        num_samples=num_samples,
        feature_dim=feature_dim,
        skipped=skipped,
        meta=blob["meta"],
    )
