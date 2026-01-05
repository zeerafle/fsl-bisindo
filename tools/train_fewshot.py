"""
Few-shot learning training script for ProtoNet on WL-BISINDO.

Uses easyfsl library for episodic training with TaskSampler.

Pipeline: download data -> extract keypoints -> train few-shot (with baked-in encoder)

Usage:
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml

    # Override encoder (use existing backbone configs):
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --encoder configs/backbones/csl_slgcn.yaml

    # Override few-shot settings:
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --n_way 5 --k_shot 5 --n_query 15

    # Fine-tune encoder (unfreeze):
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --unfreeze_encoder --lr 0.0001

    # Evaluate frozen encoder baseline (zero-shot, no training):
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --eval_only

    # Evaluate different encoders as baseline:
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --encoder configs/backbones/csl_slgcn.yaml --eval_only

    # Evaluate with loaded checkpoint (local or W&B):
    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --eval_only --resume_checkpoint path/to/model.pt --n_test_episodes 1000

    python tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
        --eval_only --resume_artifact "entity/project/artifact:v0"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from easyfsl.samplers import TaskSampler
from omegaconf import OmegaConf
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsl_bisindo.data.wlbisindo_dataset import (
    WLBisindoFewShotDataset,
    WLBisindoKeypointsDataset,
)
from fsl_bisindo.models.protonet import SignLanguageProtoNet, build_protonet_from_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ProtoNet for few-shot sign language recognition"
    )

    # Config
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )

    # Encoder override
    p.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Override encoder config path (e.g., configs/backbones/autsl_slgcn.yaml)",
    )

    # Few-shot settings override
    p.add_argument("--n_way", type=int, default=None, help="Override N-way")
    p.add_argument("--k_shot", type=int, default=None, help="Override K-shot")
    p.add_argument("--n_query", type=int, default=None, help="Override N-query")

    # Training overrides
    p.add_argument("--n_epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        help="Unfreeze encoder for fine-tuning",
    )

    # Hardware
    p.add_argument("--device", type=str, default=None, help="Override device")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Output
    p.add_argument("--save_dir", type=str, default=None, help="Override save directory")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")

    # Logging
    p.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Override W&B mode",
    )
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    # Evaluation only mode (for frozen encoder baseline)
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, only evaluate on test set (for frozen encoder baseline)",
    )

    # Checkpoint loading for evaluation
    p.add_argument(
        "--resume_artifact",
        type=str,
        default=None,
        help="W&B artifact reference to load checkpoint from (e.g., entity/project/protonet-best:v0)",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Local checkpoint path to load (e.g., experiments/protonet/best_model.pt)",
    )
    p.add_argument(
        "--n_test_episodes",
        type=int,
        default=None,
        help="Override number of test episodes",
    )

    return p.parse_args()


def make_preprocess_transform(window_size: int | None = None):
    """
    Create a transform that:
    1. Converts 543 keypoints -> 27 (Body+Hands selection)
    2. Normalizes using OpenHands presets
    3. Returns [C, T, V] tensor
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


def pad_or_truncate_frames(frames: torch.Tensor, target_t: int) -> torch.Tensor:
    """Pad or truncate temporal dimension to target_t."""
    c, t, v = frames.shape
    if t == target_t:
        return frames
    if t > target_t:
        return frames[:, :target_t, :]
    # Pad
    pad = torch.zeros((c, target_t - t, v), dtype=frames.dtype)
    return torch.cat([frames, pad], dim=1)


class FilteredFewShotDataset(WLBisindoFewShotDataset):
    """
    WLBisindoFewShotDataset filtered to specific classes.

    This is needed because easyfsl's TaskSampler samples from the dataset's
    get_labels() return. For train/val/test splits with different classes,
    we need separate filtered datasets.
    """

    def __init__(
        self,
        base_dataset: WLBisindoKeypointsDataset,
        allowed_classes: list[int],
    ):
        super().__init__(base_dataset)

        # Build index mapping for allowed classes
        self.allowed_classes = set(allowed_classes)
        self.filtered_indices = [
            i
            for i, item in enumerate(base_dataset.items)
            if item.label in self.allowed_classes
        ]

        # Create label remapping (for consistency with TaskSampler)
        self._labels = [base_dataset.items[i].label for i in self.filtered_indices]

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        original_idx = self.filtered_indices[idx]
        return super().__getitem__(original_idx)

    def get_labels(self) -> list[int]:
        return self._labels


def build_datasets_and_loaders(
    cfg: dict[str, Any],
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders using easyfsl's TaskSampler.
    """
    # Data config
    data_cfg = cfg.get("data", {})
    keypoints_root = data_cfg.get("keypoints_root", "data/WL-BISINDO/keypoints")
    window_size = data_cfg.get("window_size", 64)

    # Few-shot config
    fsl_cfg = cfg.get("fewshot", {})
    n_way = fsl_cfg.get("n_way", 5)
    k_shot = fsl_cfg.get("k_shot", 5)
    n_query = fsl_cfg.get("n_query", 10)

    # Split config
    split_cfg = cfg.get("split", {})
    base_classes = split_cfg.get("base_classes", [])
    val_classes = split_cfg.get("val_classes", [])
    test_classes = split_cfg.get("test_classes", [])

    # Training config
    train_cfg = cfg.get("training", {})
    n_train_tasks = train_cfg.get("n_train_episodes", 1000)
    n_val_tasks = train_cfg.get("n_val_episodes", 100)
    n_test_tasks = train_cfg.get("n_test_episodes", 600)

    # Build transform
    transform = make_preprocess_transform(window_size=window_size)

    # Build base dataset
    base_dataset = WLBisindoKeypointsDataset(
        keypoints_root=keypoints_root,
        transform=transform,
        allow_unparsed=False,
    )

    print(f"Total samples in dataset: {len(base_dataset)}")

    # Create filtered datasets for each split
    train_dataset = FilteredFewShotDataset(base_dataset, base_classes)
    val_dataset = FilteredFewShotDataset(base_dataset, val_classes)
    test_dataset = FilteredFewShotDataset(base_dataset, test_classes)

    print(f"Train split: {len(train_dataset)} samples, {len(base_classes)} classes")
    print(f"Val split: {len(val_dataset)} samples, {len(val_classes)} classes")
    print(f"Test split: {len(test_dataset)} samples, {len(test_classes)} classes")

    # Create TaskSamplers (from easyfsl)
    train_sampler = TaskSampler(
        train_dataset,
        n_way=n_way,
        n_shot=k_shot,
        n_query=n_query,
        n_tasks=n_train_tasks,
    )
    val_sampler = TaskSampler(
        val_dataset,
        n_way=n_way,
        n_shot=k_shot,
        n_query=n_query,
        n_tasks=n_val_tasks,
    )
    test_sampler = TaskSampler(
        test_dataset,
        n_way=n_way,
        n_shot=k_shot,
        n_query=n_query,
        n_tasks=n_test_tasks,
    )

    # Create DataLoaders with episodic_collate_fn

    dataloader_config = {
        "persistent_workers": True if num_workers > 0 else False,  # keep workers alive
        "prefetch_factor": 2
        if num_workers > 0
        else None,  # prefetch 2 batches per worker
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
        **dataloader_config,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
        **dataloader_config,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    return train_loader, val_loader, test_loader


def build_model(
    cfg: dict[str, Any], device: str | torch.device
) -> SignLanguageProtoNet:
    """Build ProtoNet model from config."""
    # Load encoder config path
    encoder_cfg = cfg.get("encoder", {})
    backbone_cfg_path = encoder_cfg.get("backbone_cfg")

    if backbone_cfg_path is None:
        raise ValueError("Must specify encoder.backbone_cfg in config")

    # ProtoNet config
    protonet_cfg = cfg.get("protonet", {})
    freeze_encoder = protonet_cfg.get("freeze_encoder", True)
    use_softmax = protonet_cfg.get("use_softmax", False)

    # Build using our factory function
    model = build_protonet_from_cfg(
        backbone_cfg_path=backbone_cfg_path,
        device=device,
        freeze_encoder=freeze_encoder,
        use_softmax=use_softmax,
    )

    return model


def build_optimizer_and_scheduler(
    model: SignLanguageProtoNet,
    cfg: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Build optimizer and scheduler from config."""
    optim_cfg = cfg.get("optim", {})
    sched_cfg = cfg.get("scheduler", {})
    train_cfg = cfg.get("training", {})

    # Optimizer
    optim_name = optim_cfg.get("name", "Adam")
    lr = optim_cfg.get("lr", 0.0001)
    weight_decay = optim_cfg.get("weight_decay", 0.0)

    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        print("Warning: No trainable parameters! Encoder might be frozen.")
        # Create a dummy optimizer that won't update anything
        params = [torch.nn.Parameter(torch.zeros(1))]

    if optim_name == "Adam":
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "SGD":
        momentum = optim_cfg.get("momentum", 0.9)
        optimizer = SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    # Scheduler
    sched_name = sched_cfg.get("name", None)
    scheduler = None

    if sched_name == "CosineAnnealingLR":
        T_max = sched_cfg.get("T_max", train_cfg.get("n_epochs", 100))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif sched_name == "StepLR":
        step_size = sched_cfg.get("step_size", 30)
        gamma = sched_cfg.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, scheduler


def train_one_epoch(
    model: SignLanguageProtoNet,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """
    Train for one epoch using episodic training.

    Each batch from train_loader is a few-shot task with:
        (support_images, support_labels, query_images, query_labels, class_ids)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_queries = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        support_images, support_labels, query_images, query_labels, _ = batch

        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        optimizer.zero_grad()

        # Process support set (computes prototypes)
        model.process_support_set(support_images, support_labels)

        # Get predictions for query set
        logits = model(query_images)

        # Compute loss
        loss = F.cross_entropy(logits, query_labels)

        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == query_labels).sum().item()
        total_queries += query_labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_queries

    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(
    model: SignLanguageProtoNet,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> dict[str, float]:
    """
    Evaluate model on few-shot tasks.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_queries = 0
    episode_accuracies = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            support_images, support_labels, query_images, query_labels, _ = batch

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            # Process support set
            model.process_support_set(support_images, support_labels)

            # Get predictions
            logits = model(query_images)

            # Compute loss
            loss = F.cross_entropy(logits, query_labels)

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct = (preds == query_labels).sum().item()
            total_correct += correct
            total_queries += query_labels.size(0)

            # Track per-episode accuracy
            episode_accuracies.append(correct / query_labels.size(0))

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_queries

    # Compute 95% confidence interval
    accs = np.array(episode_accuracies)
    mean_acc = accs.mean()
    std_acc = accs.std()
    ci = 1.96 * std_acc / np.sqrt(len(accs))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "mean_accuracy": mean_acc,
        "ci_95": ci,
        "n_episodes": len(episode_accuracies),
    }


def main():
    from dotenv import load_dotenv

    load_dotenv()

    args = parse_args()

    # Load config
    cfg_omg = OmegaConf.load(args.config)
    cfg: dict[str, Any] = OmegaConf.to_container(cfg_omg, resolve=True)  # type: ignore

    # Apply overrides
    if args.encoder is not None:
        cfg["encoder"]["backbone_cfg"] = args.encoder

    if args.n_way is not None:
        cfg["fewshot"]["n_way"] = args.n_way
    if args.k_shot is not None:
        cfg["fewshot"]["k_shot"] = args.k_shot
    if args.n_query is not None:
        cfg["fewshot"]["n_query"] = args.n_query

    if args.n_epochs is not None:
        cfg["training"]["n_epochs"] = args.n_epochs
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.unfreeze_encoder:
        cfg["protonet"]["freeze_encoder"] = False
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.n_test_episodes is not None:
        cfg["training"]["n_test_episodes"] = args.n_test_episodes

    if args.device is not None:
        cfg["device"] = args.device
    if args.save_dir is not None:
        cfg["checkpoint"]["save_dir"] = args.save_dir

    # W&B setup
    wandb_cfg = cfg.get("wandb", {})
    if args.no_wandb:
        wandb_cfg["mode"] = "disabled"
    elif args.wandb_mode is not None:
        wandb_cfg["mode"] = args.wandb_mode

    # Initialize W&B
    run = None
    try:
        if wandb_cfg.get("mode") != "disabled":
            from fsl_bisindo.utils.wandb_utils import wandb_init

            run = wandb_init(cfg, job_type="train")
    except ImportError:
        print("W&B not available, continuing without logging")

    # Set seed
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get device
    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"Using device: {device}")

    # Build model
    print("Building model...")
    model = build_model(cfg, device)

    freeze_encoder = cfg.get("protonet", {}).get("freeze_encoder", True)
    print(f"Encoder frozen: {freeze_encoder}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Build data loaders
    print("Building data loaders...")
    train_loader, val_loader, test_loader = build_datasets_and_loaders(
        cfg, num_workers=args.num_workers
    )

    print(f"Train tasks per epoch: {len(train_loader)}")
    print(f"Val tasks: {len(val_loader)}")
    print(f"Test tasks: {len(test_loader)}")

    # Training config
    train_cfg = cfg.get("training", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    n_epochs = train_cfg.get("n_epochs", 100)
    eval_interval = train_cfg.get("eval_interval", 5)
    early_stopping_patience = train_cfg.get("early_stopping_patience", None)
    save_dir = checkpoint_cfg.get("save_dir")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation-only mode (for frozen encoder baseline or loaded checkpoint)
    if args.eval_only:
        print("\n" + "=" * 60)

        # Load checkpoint if specified
        checkpoint_path = None
        if args.resume_artifact:
            print(f"Loading checkpoint from W&B artifact: {args.resume_artifact}")
            import wandb

            api = wandb.Api()
            artifact = api.artifact(args.resume_artifact, type="model")
            artifact_dir = artifact.download()
            checkpoint_path = Path(artifact_dir) / "best_model.pt"
            print(f"  Downloaded to: {checkpoint_path}")
        elif args.resume_checkpoint:
            checkpoint_path = Path(args.resume_checkpoint)
            print(f"Loading checkpoint from local path: {checkpoint_path}")

        if checkpoint_path:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("  Checkpoint loaded successfully!")
            print("Evaluation-only mode (with loaded checkpoint)")
        else:
            print("Evaluation-only mode (frozen encoder baseline)")

        print("Skipping training, directly evaluating on test set...")
        print("=" * 60)

        # Evaluate on test set
        print("\nEvaluating on test set (novel classes)...")
        test_metrics = evaluate(model, test_loader, device, desc="Testing")

        print(f"\nTest Results ({test_metrics['n_episodes']} episodes):")
        print(
            f"  Mean accuracy: {test_metrics['mean_accuracy']:.4f} ± {test_metrics['ci_95']:.4f} (95% CI)"
        )
        print(f"  Loss: {test_metrics['loss']:.4f}")

        # Also evaluate on val set for reference
        print("\nEvaluating on validation set...")
        val_metrics = evaluate(model, val_loader, device, desc="Validating")
        print(f"\nValidation Results ({val_metrics['n_episodes']} episodes):")
        print(
            f"  Mean accuracy: {val_metrics['mean_accuracy']:.4f} ± {val_metrics['ci_95']:.4f} (95% CI)"
        )

        # Log to W&B
        if run:
            run.log(
                {
                    "test/loss": test_metrics["loss"],
                    "test/accuracy": test_metrics["mean_accuracy"],
                    "test/ci_95": test_metrics["ci_95"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["mean_accuracy"],
                    "val/ci_95": val_metrics["ci_95"],
                }
            )
            run.summary["test_acc_mean"] = test_metrics["mean_accuracy"]
            run.summary["test_acc_ci"] = test_metrics["ci_95"]
            run.summary["val_acc_mean"] = val_metrics["mean_accuracy"]
            run.summary["val_acc_ci"] = val_metrics["ci_95"]
            run.finish()

        return {"val": val_metrics, "test": test_metrics}

    # Build optimizer and scheduler (only needed for training)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    if early_stopping_patience:
        print(f"Early stopping enabled with patience={early_stopping_patience}")
    print("=" * 60)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        print(
            f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
        )

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Evaluate
        if epoch % eval_interval == 0 or epoch == n_epochs:
            val_metrics = evaluate(model, val_loader, device, desc="Validating")
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['mean_accuracy']:.4f} ± {val_metrics['ci_95']:.4f}"
            )

            # Log to W&B
            if run:
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["mean_accuracy"],
                        "val/ci_95": val_metrics["ci_95"],
                    }
                )

            # Save best model
            if val_metrics["mean_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["mean_accuracy"]
                epochs_without_improvement = 0
                if save_dir:
                    torch.save(model.state_dict(), save_dir / "best_model.pt")
                    print(f"  Saved best model (val_acc={best_val_acc:.4f})")
            else:
                epochs_without_improvement += 1
                if (
                    early_stopping_patience
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    print(
                        f"\nEarly stopping triggered after {epochs_without_improvement} evaluations without improvement"
                    )
                    break

    history["best_val_acc"] = best_val_acc

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 60)

    # Load best model for final evaluation
    if save_dir and (save_dir / "best_model.pt").exists():
        print("\nLoading best model weights for final evaluation...")
        model.load_state_dict(
            torch.load(save_dir / "best_model.pt", map_location=device)
        )

    # Final evaluation on test set
    print("\nEvaluating on test set (novel classes)...")
    test_metrics = evaluate(model, test_loader, device, desc="Testing")

    print(f"\nTest Results ({test_metrics['n_episodes']} episodes):")
    print(
        f"  Mean accuracy: {test_metrics['mean_accuracy']:.4f} ± {test_metrics['ci_95']:.4f} (95% CI)"
    )
    print(f"  Loss: {test_metrics['loss']:.4f}")

    # Log final results and save model artifact
    if run:
        run.log(
            {
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["mean_accuracy"],
                "test/ci_95": test_metrics["ci_95"],
            }
        )
        run.summary["test_acc_mean"] = test_metrics["mean_accuracy"]
        run.summary["test_acc_ci"] = test_metrics["ci_95"]
        run.summary["best_val_acc"] = best_val_acc

        # Upload best model as W&B artifact
        if save_dir:
            best_model_path = save_dir / "best_model.pt"
            if best_model_path.exists():
                from fsl_bisindo.utils.wandb_utils import log_artifact

                try:
                    artifact_info = log_artifact(
                        run,
                        save_dir / "best_model.pt",
                        name="protonet-best",
                        type="model",
                        metadata={
                            "best_val_acc": best_val_acc,
                            "test_acc_mean": test_metrics["mean_accuracy"],
                            "test_acc_ci": test_metrics["ci_95"],
                            "encoder": cfg.get("encoder", {}).get("backbone_cfg"),
                            "n_way": cfg.get("fewshot", {}).get("n_way"),
                            "k_shot": cfg.get("fewshot", {}).get("k_shot"),
                        },
                        aliases=["best", "latest"],
                    )
                    print(f"  Artifact uploaded: {artifact_info.qualified}")
                except Exception as e:
                    print(f"Warning: failed to upload best model artifact to W&B: {e}")
            else:
                print("\nBest model file not found; skipping W&B artifact upload.")

        run.finish()

    return history, test_metrics


if __name__ == "__main__":
    main()
