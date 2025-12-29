"""
Prototypical Networks with integrated encoder for end-to-end few-shot learning.

This module extends easyfsl's PrototypicalNetworks to work with OpenHands
SL-GCN encoders for sign language recognition.

The encoder can be frozen (default) or trainable for fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from easyfsl.methods import PrototypicalNetworks


class SignLanguageProtoNet(PrototypicalNetworks):
    """
    Prototypical Networks adapted for sign language pose sequences.

    This extends easyfsl's PrototypicalNetworks with:
    - Support for SL-GCN encoders from OpenHands
    - Encoder freeze/unfreeze for training flexibility
    - Proper handling of pose sequence inputs [B, C, T, V]

    Args:
        backbone: The feature extractor (e.g., SL-GCN encoder)
        use_softmax: Whether to return softmax probabilities
        freeze_encoder: If True, freeze encoder parameters

    Note:
        easyfsl's PrototypicalNetworks expects backbone to output [B, D] features.
        SL-GCN encoders output [B, D] directly after their temporal pooling.
    """

    def __init__(
        self,
        backbone: nn.Module,
        use_softmax: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__(backbone=backbone, use_softmax=use_softmax)

        if freeze_encoder:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters (no gradients)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters (allow gradients)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """Override train() to keep frozen encoder in eval mode."""
        super().train(mode)
        # If encoder is frozen, keep it in eval mode
        if hasattr(self, "backbone"):
            if not any(p.requires_grad for p in self.backbone.parameters()):
                self.backbone.eval()
        return self


def build_protonet_from_cfg(
    backbone_cfg_path: str,
    *,
    device: str | torch.device = "cpu",
    freeze_encoder: bool = True,
    use_softmax: bool = False,
) -> SignLanguageProtoNet:
    """
    Build a ProtoNet model from a backbone config file.

    Args:
        backbone_cfg_path: Path to backbone config (e.g., configs/backbones/autsl_slgcn.yaml)
        device: Device to load model on
        freeze_encoder: If True, freeze encoder parameters
        use_softmax: If True, return softmax probabilities

    Returns:
        SignLanguageProtoNet model
    """
    from omegaconf import OmegaConf

    from fsl_bisindo.models.load_pretrained import build_backbone_from_cfg

    # Load backbone config
    backbone_cfg = OmegaConf.load(backbone_cfg_path)

    # Build encoder (don't freeze here, SignLanguageProtoNet handles it)
    encoder = build_backbone_from_cfg(
        backbone_cfg,
        device=device,
        return_encoder=True,
        freeze=False,  # SignLanguageProtoNet will handle freezing
    )

    # Build ProtoNet
    model = SignLanguageProtoNet(
        backbone=encoder,
        use_softmax=use_softmax,
        freeze_encoder=freeze_encoder,
    )

    return model.to(device)


# Available encoder configs mapping
ENCODER_CONFIGS = {
    "autsl": "configs/backbones/autsl_slgcn.yaml",
    "csl": "configs/backbones/csl_slgcn.yaml",
    "lsa64": "configs/backbones/lsa64_slgcn.yaml",
}


def get_encoder_config_path(encoder_name: str) -> str:
    """Get the config path for a named encoder."""
    if encoder_name not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Available: {list(ENCODER_CONFIGS.keys())}"
        )
    return ENCODER_CONFIGS[encoder_name]
