"""
download_weights.py — Fetch CH4Net model weights from cloud storage.

The model weight files are too large for plain git (each ~52 MB). They must
be downloaded separately before running inference or training scripts.

Usage
-----
    python download_weights.py                # download production weight only
    python download_weights.py --all          # download all experiment weights
    python download_weights.py --check        # verify checksums of existing files

Configuration
-------------
Set WEIGHTS_BASE_URL below to the base URL of wherever the weights are hosted,
e.g. a Google Drive folder shared as direct-download links, a Zenodo record, or
an S3 bucket.  Each entry in WEIGHT_FILES maps a filename to its individual
download URL (override) or None to fall back to WEIGHTS_BASE_URL/<filename>.

If you are using Google Drive, convert the share link:
    Share link:   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    Download URL: https://drive.google.com/uc?export=download&id=<FILE_ID>
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

# Base URL for weight files.  Replace with your actual hosting URL.
# Example (Google Drive folder or Zenodo record base):
WEIGHTS_BASE_URL = "https://YOUR_HOSTING_URL_HERE"

# Map each weight filename to (direct_download_url, sha256_hex).
# Set direct_download_url to None to construct it as WEIGHTS_BASE_URL/<filename>.
# Set sha256_hex to None to skip checksum verification for that file.
WEIGHT_FILES = {
    "european_model_v8.pth": {
        "url":        None,   # replace with direct URL if needed
        "sha256":     None,   # replace with sha256 hex string after uploading
        "production": True,   # this is the model used for all paper results
        "description": "CH4Net v8 European fine-tune (div_factor=1, 13.5M params)",
    },
    "european_model_v10.pth": {
        "url":        None,
        "sha256":     None,
        "production": False,
        "description": "CH4Net v10 experiment checkpoint",
    },
    "european_model.pth": {
        "url":        None,
        "sha256":     None,
        "production": False,
        "description": "CH4Net European fine-tune v1 (earlier checkpoint)",
    },
    "european_model_imagelevel.pth": {
        "url":        None,
        "sha256":     None,
        "production": False,
        "description": "Image-level classifier experiment checkpoint",
    },
    "synthetic_only_v1.pth": {
        "url":        None,
        "sha256":     None,
        "production": False,
        "description": "Synthetic-data-only ablation checkpoint",
    },
    "best_model.pth": {
        "url":        None,
        "sha256":     None,
        "production": False,
        "description": "Best validation-loss checkpoint from final training run",
    },
    "ch4net_div8_retrained.pth": {
        "url":        None,
        "sha256":     None,
        "production": False,
        "description": "div_factor=8 retrain experiment (smaller model)",
    },
}

WEIGHTS_DIR = Path(__file__).parent / "weights"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple download progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  ({downloaded >> 20} / {total_size >> 20} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()
    else:
        sys.stdout.write(f"\r  downloaded {downloaded >> 20} MB...")
        sys.stdout.flush()


def _resolve_url(name: str, info: dict) -> str:
    """Return the download URL for a weight file."""
    if info.get("url"):
        return info["url"]
    if WEIGHTS_BASE_URL == "https://YOUR_HOSTING_URL_HERE":
        raise RuntimeError(
            "WEIGHTS_BASE_URL has not been set in download_weights.py.\n"
            "Edit the script and set WEIGHTS_BASE_URL to the base URL of "
            "your cloud storage location, or set individual 'url' entries "
            "in WEIGHT_FILES."
        )
    return f"{WEIGHTS_BASE_URL.rstrip('/')}/{name}"


def download_weight(name: str, info: dict, force: bool = False) -> bool:
    """
    Download a single weight file.

    Parameters
    ----------
    name : str
        Filename (key in WEIGHT_FILES).
    info : dict
        Metadata dict from WEIGHT_FILES.
    force : bool
        Re-download even if the file already exists.

    Return
    ------
    bool
        True if the file is present and verified after this call.
    """
    dest = WEIGHTS_DIR / name
    if dest.exists() and not force:
        print(f"  {name}: already present — skipping (use --force to re-download)")
        if info.get("sha256"):
            actual = _sha256(dest)
            if actual != info["sha256"]:
                print(f"    WARNING: checksum mismatch  expected={info['sha256'][:12]}…  got={actual[:12]}…")
                return False
        return True

    url = _resolve_url(name, info)
    print(f"  Downloading {name}  ({info.get('description', '')})")
    print(f"    from: {url}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    except Exception as exc:
        print(f"\n    ERROR: download failed — {exc}")
        if dest.exists():
            dest.unlink()
        return False

    if info.get("sha256"):
        actual = _sha256(dest)
        if actual != info["sha256"]:
            print(f"    ERROR: checksum mismatch  expected={info['sha256'][:12]}…  got={actual[:12]}…")
            dest.unlink()
            return False
        print(f"    checksum OK")

    print(f"    saved to {dest}")
    return True


def check_weights() -> None:
    """Print status of all weight files (present/missing, checksum pass/fail)."""
    print(f"\nWeight files in {WEIGHTS_DIR}/\n")
    print(f"  {'File':<40} {'Size':>8}  {'Checksum':>12}  {'Notes'}")
    print("  " + "-" * 80)
    for name, info in WEIGHT_FILES.items():
        path = WEIGHTS_DIR / name
        tag = "[PRODUCTION]" if info.get("production") else ""
        if path.exists():
            size_mb = path.stat().st_size / (1 << 20)
            if info.get("sha256"):
                actual = _sha256(path)
                cs = "OK" if actual == info["sha256"] else "MISMATCH"
            else:
                cs = "not set"
            print(f"  {name:<40} {size_mb:>6.0f}MB  {cs:>12}  {tag}")
        else:
            print(f"  {name:<40} {'MISSING':>8}  {'':>12}  {tag}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main entry point for the weight downloader.

    Parameters
    ----------
    None (reads sys.argv via argparse)

    Return
    ------
    None
    """
    parser = argparse.ArgumentParser(
        description="Download CH4Net model weights from cloud storage."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all weight files (default: production weight only)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check which weights are present and verify checksums; do not download"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist"
    )
    parser.add_argument(
        "--file", metavar="NAME",
        help="Download a specific weight file by name"
    )
    args = parser.parse_args()

    if args.check:
        check_weights()
        return

    if args.file:
        if args.file not in WEIGHT_FILES:
            print(f"Unknown weight file: {args.file}")
            print(f"Available: {list(WEIGHT_FILES)}")
            sys.exit(1)
        targets = {args.file: WEIGHT_FILES[args.file]}
    elif args.all:
        targets = WEIGHT_FILES
    else:
        targets = {k: v for k, v in WEIGHT_FILES.items() if v.get("production")}

    print(f"\nDownloading {'all' if args.all else 'production'} CH4Net weights to {WEIGHTS_DIR}/\n")
    ok, failed = 0, []
    for name, info in targets.items():
        if download_weight(name, info, force=args.force):
            ok += 1
        else:
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Downloaded: {ok}   Failed: {len(failed)}")
    if failed:
        print(f"Failed files: {failed}")
        sys.exit(1)
    print("Done. The production model (european_model_v8.pth) is ready for inference.")


if __name__ == "__main__":
    main()
