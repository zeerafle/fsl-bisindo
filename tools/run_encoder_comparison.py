"""
Run few-shot experiments with multiple encoders for comparison.

This script runs ProtoNet training with different pretrained encoders
(AUTSL, CSL, LSA64) and compares results.

Usage:
    python tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml

    # With specific encoders:
    python tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml \
        --encoders autsl csl

    # Fine tuning mode (unfreeze encoder):
    python tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml \
        --unfreeze_encoder --lr 0.0001

    # Quick test:
    python tools/run_encoder_comparison.py --config configs/fewshot/train_protonet.yaml \
        --n_epochs 5 --n_train_episodes 100
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Available encoders (using existing backbone configs)
ENCODER_CONFIGS = {
    "autsl": "configs/backbones/autsl_slgcn.yaml",
    "csl": "configs/backbones/csl_slgcn.yaml",
    "lsa64": "configs/backbones/lsa64_slgcn.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run few-shot experiments with multiple encoders"
    )

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Base training config YAML",
    )

    p.add_argument(
        "--encoders",
        nargs="+",
        default=list(ENCODER_CONFIGS.keys()),
        choices=list(ENCODER_CONFIGS.keys()),
        help="Encoders to compare (default: all)",
    )

    p.add_argument("--n_epochs", type=int, default=None, help="Override epochs")
    p.add_argument(
        "--n_train_episodes", type=int, default=None, help="Override train episodes"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Add fine-tuning support
    p.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        help="Unfreeze encoder for fine-tuning (applies to all encoders)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (recommended when unfreezing encoder)",
    )

    p.add_argument("--device", type=str, default="cuda", help="Device")
    p.add_argument(
        "--save_dir",
        type=str,
        default="experiments/comparison",
        help="Output directory",
    )

    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    return p.parse_args()


def run_single_experiment(
    config_path: str,
    encoder_name: str,
    encoder_cfg_path: str,
    save_dir: Path,
    overrides: dict[str, Any],
    use_wandb: bool = True,
    unfreeze_encoder: bool = False,
) -> dict[str, Any]:
    """Run a single experiment with specified encoder."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "tools/train_fewshot.py",
        "--config",
        config_path,
        "--encoder",
        encoder_cfg_path,
        "--save_dir",
        str(save_dir / encoder_name),
    ]

    # Add overrides
    if overrides.get("n_epochs"):
        cmd.extend(["--n_epochs", str(overrides["n_epochs"])])
    if overrides.get("seed"):
        cmd.extend(["--seed", str(overrides["seed"])])
    if overrides.get("device"):
        cmd.extend(["--device", overrides["device"]])
    if overrides.get("lr"):
        cmd.extend(["--lr", str(overrides["lr"])])
    if unfreeze_encoder:
        cmd.append("--unfreeze_encoder")
    if not use_wandb:
        cmd.append("--no_wandb")

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {encoder_name}")
    print(f"Mode: {'Fine-tuning' if unfreeze_encoder else 'Frozen encoder'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    # Run experiment
    result = subprocess.run(cmd, capture_output=False)

    return {
        "encoder": encoder_name,
        "encoder_cfg": encoder_cfg_path,
        "unfreeze_encoder": unfreeze_encoder,
        "return_code": result.returncode,
        "save_dir": str(save_dir / encoder_name),
    }


def main():
    args = parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_finetuned" if args.unfreeze_encoder else "_frozen"
    save_dir = Path(args.save_dir) / f"{timestamp}{mode_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running encoder comparison experiment")
    print(
        f"Mode: {'Fine-tuning (encoder unfrozen)' if args.unfreeze_encoder else 'Frozen encoder'}"
    )
    print(f"Encoders: {args.encoders}")
    print(f"Output directory: {save_dir}")

    if args.unfreeze_encoder and args.lr is None:
        print("\n⚠️  WARNING: Fine-tuning enabled but no learning rate specified.")
        print("   Consider using --lr with a smaller value (e.g., 0.0001 or 0.00001)")

    # Collect overrides
    overrides: dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
    }
    if args.n_epochs is not None:
        overrides["n_epochs"] = args.n_epochs
    if args.lr is not None:
        overrides["lr"] = args.lr

    # Run experiments
    results = []
    for encoder_name in args.encoders:
        encoder_cfg = ENCODER_CONFIGS[encoder_name]
        result = run_single_experiment(
            config_path=args.config,
            encoder_name=encoder_name,
            encoder_cfg_path=encoder_cfg,
            save_dir=save_dir,
            overrides=overrides,
            use_wandb=not args.no_wandb,
            unfreeze_encoder=args.unfreeze_encoder,
        )
        results.append(result)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": args.config,
        "encoders": args.encoders,
        "unfreeze_encoder": args.unfreeze_encoder,
        "overrides": overrides,
        "results": results,
    }

    summary_path = save_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Comparison complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'=' * 60}")

    # Print summary table
    print("\nResults Summary:")
    print("-" * 40)
    for r in results:
        status = "✓" if r["return_code"] == 0 else "✗"
        print(f"  {status} {r['encoder']}: return_code={r['return_code']}")


if __name__ == "__main__":
    main()
