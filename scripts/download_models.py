#!/usr/bin/env python3
"""Explicit, reviewable Hugging Face model downloader with integrity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from pi0fast_wm_rl.utils.config import load_yaml, project_root


def tree_sha256(root: Path) -> str:
    """Hash relative paths and file contents in deterministic order."""
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.name != ".integrity.json"
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def selected_models(config: dict[str, Any], name: str) -> list[tuple[str, dict[str, Any]]]:
    names = ["pi0fast", "fast_tokenizer"] if name == "all" else [name]
    return [(item, config[item]) for item in names]


def verify(name: str, model: dict[str, Any], target: Path) -> bool:
    """Verify a configured checksum or the checksum recorded after download."""
    if not target.is_dir() or not any(path.is_file() for path in target.rglob("*")):
        print(f"{name}: missing or empty: {target}")
        return False
    actual = tree_sha256(target)
    expected = str(model.get("sha256", "")).strip()
    integrity = target / ".integrity.json"
    if not expected and integrity.is_file():
        expected = str(json.loads(integrity.read_text(encoding="utf-8")).get("tree_sha256", ""))
    if expected and actual != expected:
        print(f"{name}: checksum mismatch (expected {expected}, got {actual})")
        return False
    print(f"{name}: integrity OK; tree sha256={actual}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi0fast", "fast_tokenizer", "all"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation")
    args = parser.parse_args(argv)
    config = load_yaml(project_root() / "configs/pretrained_models.yaml")
    choices = selected_models(config, args.model)
    print("Planned model operation:")
    for name, model in choices:
        model_id = model["model_id"]
        revision = model.get("revision", "main")
        print(f"- {name}: {model_id} @ {revision} -> {model['local_path']}")
    if args.verify:
        return int(
            not all(
                verify(name, model, project_root() / model["local_path"]) for name, model in choices
            )
        )
    if args.dry_run:
        print("Dry run only: snapshot files would be downloaded; no network write was performed.")
        return 0
    if not args.yes:
        answer = input("Download these model files? Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Cancelled; nothing was downloaded.")
            return 1
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface-hub is missing. Install with: pip install -e '.[models]'", file=sys.stderr
        )
        return 2
    token = os.environ.get("HF_TOKEN") or None
    for name, model in choices:
        target = project_root() / model["local_path"]
        existing_files = (
            [path for path in target.rglob("*") if path.is_file() and path.name != ".gitkeep"]
            if target.exists()
            else []
        )
        if existing_files:
            print(f"Refusing to merge into non-empty target: {target}", file=sys.stderr)
            return 2
        target.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=model["model_id"],
                revision=model.get("revision", "main"),
                local_dir=target,
                token=token,
            )
        except Exception as exc:
            print(f"Download failed for {name}: {exc}", file=sys.stderr)
            return 2
        checksum = tree_sha256(target)
        integrity = {
            "model_id": model["model_id"],
            "revision": model.get("revision", "main"),
            "tree_sha256": checksum,
        }
        (target / ".integrity.json").write_text(
            json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
        )
        if not verify(name, model, target):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
