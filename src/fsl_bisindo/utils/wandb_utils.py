from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

# -----------------------
# Init helpers
# -----------------------


def wandb_init(
    cfg: dict[str, Any],
    *,
    job_type: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
) -> wandb.Run:
    """
    Initialize a W&B run using conventions for research repos.

    W&B grouping:
      - pass group + job_type to wandb.init to organize runs into an experiment. [web:6]
    """
    wcfg = dict(cfg.get("wandb", {}))

    project = wcfg.get("project", None)
    entity = wcfg.get("entity", None)

    # Allow overrides from function args
    group = group or wcfg.get("group", None)
    job_type = job_type or wcfg.get("job_type", None)
    name = name or wcfg.get("name", None)
    tags = tags or wcfg.get("tags", None)

    # Modes:
    # - online (default), offline, disabled (no logging)
    mode = mode or wcfg.get("mode", None)  # can also be controlled via env vars

    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "group": group,
        "job_type": job_type,
        "name": name,
        "tags": tags,
        "config": cfg,  # logs full experiment config for filtering/comparison
    }

    # Remove None keys to avoid wandb warnings
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    if mode is not None:
        init_kwargs["mode"] = mode

    # If you run multiple runs in one Python process (e.g., run_matrix),
    # reinit prevents overwriting/merging surprises.
    # Use "finish_previous" to finish all active runs before starting a new one.
    init_kwargs.setdefault("reinit", "finish_previous")

    run = wandb.init(**init_kwargs)
    return run


def wandb_set_run_notes(run: wandb.Run, notes: str) -> None:
    run.notes = notes


def wandb_log_code_snapshot(
    run: wandb.Run,
    *,
    root: str | Path = ".",
) -> None:
    """
    Optional: tracks current code state in the run.
    W&B supports 'log_code' to snapshot the repo code for reproducibility.
    """
    try:
        run.log_code(root=str(root))
    except Exception:
        # keep this best-effort (some environments block it)
        pass


# -----------------------
# Artifact helpers
# -----------------------


@dataclass
class LoggedArtifact:
    name: str
    type: str
    version: str  # e.g. "v0"
    qualified: str  # e.g. "entity/project/name:v0"


def log_artifact(
    run: wandb.Run,
    file_or_dir: str | Path,
    *,
    name: str,
    type: str,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> LoggedArtifact:
    """
    Log a file or directory as a W&B Artifact.

    W&B artifact workflow:
      - create wandb.Artifact(...)
      - add_file/add_dir
      - run.log_artifact(...) [web:67]
    """
    path = Path(file_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")

    art = wandb.Artifact(name=name, type=type, metadata=metadata or {})

    if path.is_dir():
        art.add_dir(str(path))
    else:
        art.add_file(str(path))

    handle = run.log_artifact(art, aliases=aliases)
    handle.wait()

    # 'handle' contains the logged artifact reference
    # handle.name is artifact name; handle.version is like "v0"
    qualified = f"{handle.entity}/{handle.project}/{handle.name}:{handle.version}"
    return LoggedArtifact(
        name=handle.name, type=type, version=handle.version, qualified=qualified
    )


def use_artifact(
    run: wandb.Run,
    artifact_ref: str,
    *,
    type: str | None = None,
) -> wandb.Artifact:
    """
    Fetch an artifact by ref, e.g.:
      "entity/project/features-autsl-split123:latest"
    """
    art = run.use_artifact(artifact_ref, type=type)
    return art


def download_artifact(
    artifact: wandb.Artifact,
    *,
    root: str | Path = "artifacts",
) -> str:
    """
    Download artifact contents locally and return local directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return artifact.download(root=str(root))


def artifact_file(
    artifact: wandb.Artifact,
    filename: str,
    *,
    root: str | Path = "artifacts",
) -> str:
    """
    Download exactly one file from an artifact (when you know its name).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Download the specific file and return its local path
    return artifact.get_entry(filename).download(root=str(root))


# -----------------------
# Small convenience utilities
# -----------------------


def maybe_disable_wandb_from_cfg(cfg: dict[str, Any]) -> None:
    """
    Convenience: allow turning off W&B without code changes:
      cfg["wandb"]["mode"] = "disabled"
    """
    wcfg = cfg.get("wandb", {})
    mode = wcfg.get("mode", None)
    if mode in {"offline", "disabled"}:
        os.environ["WANDB_MODE"] = mode


def dump_json_artifact_sidecar(
    out_path: str | Path,
    obj: dict[str, Any],
) -> str:
    """
    Writes a small JSON sidecar (useful for split definitions, indices, meta).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return str(out_path)
