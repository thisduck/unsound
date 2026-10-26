#!/usr/bin/env python3
"""Build Tauri's update feed from the signed archive and draft release metadata."""
import argparse
import base64
import json
from pathlib import Path
import re
from urllib.parse import quote


def manifest(version, repository, archive, release):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("expected a stable release version")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid GitHub repository")
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError("the updater archive is missing or empty")
    signature = Path(str(archive) + ".sig").read_text().strip()
    if not signature or not base64.b64decode(signature, validate=True).startswith(b"untrusted comment:"):
        raise ValueError("invalid updater signature file")
    if release["tag_name"] != f"v{version}":
        raise ValueError("release tag does not match the bundled version")
    return {
        "version": version,
        "notes": release.get("body") or "",
        "pub_date": release["created_at"],
        "platforms": {"darwin-aarch64": {
            "signature": signature,
            "url": f"https://github.com/{repository}/releases/download/v{version}/{quote(archive.name)}",
        }},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("repository")
    parser.add_argument("release_json", type=Path)
    parser.add_argument("--archive", type=Path, default=Path("src-tauri/target/release/bundle/macos/unsound.app.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("latest.json"))
    args = parser.parse_args()
    value = manifest(args.version, args.repository, args.archive, json.loads(args.release_json.read_text()))
    args.output.write_text(json.dumps(value, indent=2) + "\n")
