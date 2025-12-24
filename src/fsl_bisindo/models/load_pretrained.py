from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

try:
    from openhands.apis.inference import InferenceModel
except Exception as e:  # keep import-time failures readable
    InferenceModel = None


# -------------------------
# Public API
# -------------------------


def build_backbone_from_cfg(
    backbone_cfg: dict[str, Any] | DictConfig,
    *,
    device: str | torch.device = "cpu",
    return_encoder: bool = True,
    freeze: bool = True,
) -> torch.nn.Module:
    """
    Build a backbone (or encoder) using a backbone config.
    If return_encoder=True, returns model.encoder (the feature extractor).

    Args:
        backbone_cfg: Configuration dict with model architecture and checkpoint path
        device: Device to load model on
        return_encoder: If True, return only the encoder; else return full model
        freeze: If True, freeze all parameters and set to eval mode

    Returns:
        The encoder or full model module
    """
    family = str(_get(backbone_cfg, "model.encoder.type", default="decoupled-gcn"))

    if family == "decoupled-gcn":
        model = _build_openhands_slgcn(
            cfg=backbone_cfg,
            device=device,
            return_encoder=return_encoder,
        )
    else:
        raise ValueError(f"Unsupported backbone family: {family}")

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    else:
        # When not frozen, ensure all parameters are trainable
        for p in model.parameters():
            p.requires_grad = True
        model.train()

    return model


# -------------------------
# OpenHands SL-GCN loader
# -------------------------


def _build_openhands_slgcn(
    *,
    cfg: dict[str, Any] | DictConfig,
    device: str | torch.device,
    return_encoder: bool,
) -> torch.nn.Module:
    """
    Loads an OpenHands model using its config + checkpoint and returns either
    the full model or the encoder.

    OpenHands uses:
      cfg = OmegaConf.load(...)
      model = InferenceModel(cfg=cfg)
      model.init_from_checkpoint_if_available()
    for inference/testing. [web:167]
    """
    if InferenceModel is None:
        raise ImportError(
            "OpenHands is not available. Install openhands to use family='openhands_slgcn'."
        )

    inf = InferenceModel(cfg=cfg)
    inf.init_from_checkpoint_if_available()

    # Be robust to small internal differences: some versions expose .model.encoder
    # (as in your notebook), others might expose encoder directly.
    core = getattr(inf, "model", inf)

    if return_encoder:
        enc = getattr(core, "encoder", None)
        if enc is None:
            raise AttributeError(
                "Could not find encoder on loaded OpenHands model (expected `.model.encoder`)."
            )
        enc = enc.to(device)
        # Don't set eval() here - let the caller control train/eval mode
        return enc

    core = core.to(device)
    # Don't set eval() here - let the caller control train/eval mode
    return core


# -------------------------
# Checkpoint resolution
# -------------------------


def _resolve_checkpoint_to_local(checkpoint_path: str | Path) -> str:
    """
    Supports:
      - Local paths: /.../model.ckpt or .pth
      - W&B artifact URI: wandb-artifact://entity/project/name:alias (download)
        (Artifacts are W&B's recommended way to version datasets/models/features.) [web:67]
    """
    p = str(checkpoint_path)

    if p.startswith("wandb-artifact://"):
        return _download_wandb_artifact(p)

    return p


def _download_wandb_artifact(uri: str) -> str:
    """
    Download a W&B artifact to a local directory and return a resolved checkpoint file path.

    URI format (convention used here):
      wandb-artifact://<artifact_ref>

    Where <artifact_ref> is the normal W&B artifact ref string, e.g.:
      entity/project/artifact_name:latest
    """
    # Keep wandb import local so core loading still works without wandb.
    import wandb

    ref = uri.replace("wandb-artifact://", "", 1).strip()
    api = wandb.Api()
    art = api.artifact(ref)
    local_dir = art.download()

    # Heuristic: if artifact is a directory, pick the first common checkpoint extension.
    # Prefer explicit config usage where checkpoint_path points to a single file.
    cand = _find_first_checkpoint_file(local_dir)
    if cand is None:
        raise FileNotFoundError(
            f"Downloaded artifact to {local_dir}, but no checkpoint file found. "
            "Expected one of: .ckpt, .pth, .pt"
        )
    return cand


def _find_first_checkpoint_file(root_dir: str | Path) -> str | None:
    root_dir = Path(root_dir)
    for ext in (".ckpt", ".pth", ".pt"):
        hits = sorted(root_dir.rglob(f"*{ext}"))
        if hits:
            return str(hits[0])
    return None


# -------------------------
# Small config helpers
# -------------------------


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "__getitem__") and key in cfg:
        return cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


def _require(cfg: Any, key: str) -> Any:
    v = _get(cfg, key, default=None)
    if v is None:
        raise KeyError(f"Missing required backbone config key: '{key}'")
    return v
