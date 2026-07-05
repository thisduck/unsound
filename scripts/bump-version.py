#!/usr/bin/env python3
"""Set the app version across tauri.conf.json, package.json, Cargo.toml and
Cargo.lock. Used by the release workflow: bump-version.py 0.2.0"""

import json
import re
import sys

version = sys.argv[1]
if not re.fullmatch(r"\d+\.\d+\.\d+", version):
    sys.exit(f"not a valid version: {version!r} (expected e.g. 0.2.0)")


def bump_json(path: str) -> None:
    with open(path) as f:
        data = json.load(f)
    data["version"] = version
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def bump_cargo_toml(path: str) -> None:
    with open(path) as f:
        text = f.read()
    # First version line belongs to [package] at the top of the file.
    text = re.sub(r'^version = ".*"$', f'version = "{version}"', text, count=1, flags=re.M)
    with open(path, "w") as f:
        f.write(text)


def bump_cargo_lock(path: str) -> None:
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == 'name = "unsound"':
            lines[i + 1] = f'version = "{version}"\n'
            break
    else:
        sys.exit("unsound package not found in Cargo.lock")
    with open(path, "w") as f:
        f.writelines(lines)


bump_json("src-tauri/tauri.conf.json")
bump_json("package.json")
bump_cargo_toml("src-tauri/Cargo.toml")
bump_cargo_lock("src-tauri/Cargo.lock")
print(f"version set to {version}")
