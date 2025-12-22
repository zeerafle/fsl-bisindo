from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

_FILENAME_RE = re.compile(
    r"^signer(?P<signer>\d+)_label(?P<label>\d+)_sample(?P<sample>\d+)$"
)


@dataclass(frozen=True)
class WLBisindoItem:
    path: str
    video_id: str  # stem
    signer_id: int
    label: int
    sample_id: int


def parse_wlbisindo_filename(stem: str) -> tuple[int, int, int] | None:
    """
    Parse: signer0_label0_sample1  -> (signer_id, label, sample_id)
    """
    m = _FILENAME_RE.match(stem)
    if m is None:
        return None
    return int(m["signer"]), int(m["label"]), int(m["sample"])


class WLBisindoKeypointsDataset(Dataset):
    """
    Loads WL-BISINDO keypoints saved as .npy files.

    Each item returns a dict (so DataLoader can collate it):
      {
        "frames": FloatTensor [C, T, V]  (after transform; or [T,V,C] if return_numpy=True)
        "label": LongTensor scalar
        "signer_id": LongTensor scalar
        "video_id": str
        "path": str
      }

    Notes:
    - Keep heavy preprocessing outside the Dataset if possible; but lightweight transforms
      are fine (e.g., reordering/selecting joints, normalization). [web:94]
    """

    def __init__(
        self,
        keypoints_root: str | Path,
        *,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
        file_list: Sequence[str | Path] | None = None,
        allow_unparsed: bool = False,
        return_numpy: bool = False,
        sort_files: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(keypoints_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Keypoints root not found: {self.root}")

        self.transform = transform
        self.return_numpy = return_numpy

        # Build file list
        if file_list is None:
            files = list(self.root.glob("*.npy"))
        else:
            files = [Path(p) for p in file_list]

        if sort_files:
            files = sorted(files, key=lambda p: p.name)

        items: list[WLBisindoItem] = []
        skipped = 0

        for p in files:
            stem = p.stem
            parsed = parse_wlbisindo_filename(stem)

            if parsed is None:
                if allow_unparsed:
                    # Fallback: signer/label/sample unknown
                    items.append(
                        WLBisindoItem(
                            path=str(p),
                            video_id=stem,
                            signer_id=-1,
                            label=-1,
                            sample_id=-1,
                        )
                    )
                else:
                    skipped += 1
                continue

            signer_id, label, sample_id = parsed
            items.append(
                WLBisindoItem(
                    path=str(p),
                    video_id=stem,
                    signer_id=signer_id,
                    label=label,
                    sample_id=sample_id,
                )
            )

        if len(items) == 0:
            raise RuntimeError(
                f"No usable .npy files found under {self.root}. "
                f"skipped_unparsed={skipped}, allow_unparsed={allow_unparsed}"
            )

        self.items = items
        self.skipped_unparsed = skipped

    def __len__(self) -> int:
        return len(self.items)

    def _load_npy(self, path: str) -> np.ndarray:
        # Expect something like (T, 543, C) from MediaPipe exports, but keep it generic.
        arr = np.load(path)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"np.load did not return ndarray for: {path}")
        return arr

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        arr = self._load_npy(item.path)  # e.g., (T, 543, 3)

        if self.return_numpy:
            frames_out = arr
        else:
            if self.transform is None:
                # default: convert to torch as (T,V,C) float32
                frames_out = torch.from_numpy(arr).to(torch.float32)
            else:
                # transform is expected to return torch.Tensor (recommended (C,T,V))
                frames_out = self.transform(arr)

        return {
            "frames": frames_out,
            "label": torch.tensor(item.label, dtype=torch.long),
            "signer_id": torch.tensor(item.signer_id, dtype=torch.long),
            "video_id": item.video_id,
            "path": item.path,
            "sample_id": torch.tensor(item.sample_id, dtype=torch.long),
        }
