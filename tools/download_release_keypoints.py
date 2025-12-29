"""
Download and extract WL-BISINDO keypoints from a GitHub Release asset.

Usage (example):
  python tools/download_release_keypoints.py \
      --owner your-org \
      --repo your-repo \
      --tag v0.1.0 \
      --asset keypoints-wl-bisindo.tar.gz \
      --out-dir data/WL-BISINDO/keypoints

Notes:
- Uses the GitHub API; set GITHUB_TOKEN (recommended to avoid rate limits) or pass --token.
- Supports .tar.gz/.tgz and .zip; other extensions will just be downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


def _request_json(url: str, token: Optional[str]) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest: Path, token: Optional[str]) -> None:
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers)) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def download_and_extract(
    tag: str,
    asset_name: str,
    out_dir: Path,
    token: Optional[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    release_url = (
        f"https://api.github.com/repos/zeerafle/fsl-bisindo/releases/tags/{tag}"
    )
    release = _request_json(release_url, token)

    assets = release.get("assets", [])
    match = next((a for a in assets if a.get("name") == asset_name), None)
    if match is None:
        raise FileNotFoundError(f"Asset '{asset_name}' not found in release {tag}")

    archive_path = out_dir / asset_name
    _download_file(match["url"], archive_path, token)

    suffix = archive_path.suffix.lower()
    if archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(out_dir)
    elif suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)

    return archive_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download and extract keypoints from a GitHub Release asset"
    )
    p.add_argument("--tag", default="v0.1.0", help="Release tag (e.g., v0.1.0)")
    p.add_argument(
        "--asset", default="keypoints-wl-bisindo.tar.gz", help="Asset filename to fetch"
    )
    p.add_argument(
        "--out-dir",
        default="data/WL-BISINDO/keypoints",
        help="Destination directory for extraction",
    )
    p.add_argument(
        "--token",
        default=None,
        help="GitHub token (falls back to GITHUB_TOKEN env var)",
    )
    args = p.parse_args()

    token = args.token or os.getenv("GITHUB_TOKEN")
    archive_path = download_and_extract(
        tag=args.tag,
        asset_name=args.asset,
        out_dir=Path(args.out_dir),
        token=token,
    )

    print(f"Downloaded to {archive_path}")
    print(f"Extracted into {args.out_dir}")


if __name__ == "__main__":
    main()
