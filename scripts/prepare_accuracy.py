"""Extract frozen pilot frames from the already provisioned local Apollo film.

This script has no downloader. Provision sources with prepare_validation.py
separately. Every asset must match the frozen annotation manifest digest.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    video = args.assets / "sources/apollo11.mov"
    if digest(video) != "f48960f089ade191ca56a99fa854359a161f5757a569584087b8ec89feaf9148":
        raise ValueError("Apollo film digest mismatch")
    print(subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
    for frame in json.loads(args.manifest.read_text())["images"]:
        target = (args.assets / frame["path"]).resolve()
        if not target.is_relative_to(args.assets.resolve()):
            raise ValueError("asset path escapes selected directory")
        if not target.exists() and frame["id"].startswith("apollo-"):
            timestamp = int(frame["id"].split("-")[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    str(target),
                ],
                check=True,
            )
        if digest(target) != frame["sha256"]:
            raise ValueError(f"asset digest mismatch: {frame['id']}; do not replace truth silently")
        print(f"Verified {frame['id']}", flush=True)


if __name__ == "__main__":
    main()
