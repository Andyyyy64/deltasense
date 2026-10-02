"""Explicitly provision developer validation data/weights. Requires network and ffmpeg.

This developer script is never called by DeltaSense. All downloads are verified
against the digests used in local validation, then video frames are extracted.
See docs/validation.md for credits, rights, and capture limitations.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path
from urllib.request import urlopen

DOWNLOADS = [
    (
        "models/yolo26n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    ),
    (
        "models/yolo26n-seg.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
        "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f",
    ),
    (
        "sources/astronaut.png",
        "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png",
        "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    ),
    (
        "sources/coffee.png",
        "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/coffee.png",
        "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    ),
    (
        "sources/chelsea.png",
        "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/chelsea.png",
        "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    ),
    (
        "sources/apollo11.mov",
        "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11f1093740.mov",
        "f48960f089ade191ca56a99fa854359a161f5757a569584087b8ec89feaf9148",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, url, digest in DOWNLOADS:
        destination = args.output / name
        if destination.is_file():
            data = destination.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"validation asset digest mismatch: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        print(f"Verified {name}", flush=True)
    for timestamp, name in [
        (30, "apollo-030.png"),
        (120, "apollo-120.png"),
        (121, "apollo-121.png"),
    ]:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                str(args.output / "sources/apollo11.mov"),
                "-frames:v",
                "1",
                str(args.output / "sources" / name),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
