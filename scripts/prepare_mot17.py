"""Explicitly provision a bounded MOT17 subset from its official ZIP archive.

HTTP byte ranges retrieve only selected entries. The 5.5 GB archive is never
downloaded in full. ZIP CRC and SHA-256 are recorded; cached files are verified.
The sampling plan must be fixed before inference. Not called by the library.
"""

import argparse
import hashlib
import io
import json
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get_range(url, start, length, size, etag):
    if length == 0:
        return b""
    if start < 0 or length < 0 or start + length > size or length > 16_000_000:
        raise ValueError("invalid or oversized archive range")
    request = Request(
        url, headers={"Range": f"bytes={start}-{start + length - 1}", "If-Match": etag}
    )
    with urlopen(request, timeout=30) as response:
        expected = f"bytes {start}-{start + length - 1}/{size}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("server did not honor the bounded byte range")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("truncated/oversized archive byte range")
    return data


class RemoteZip(io.RawIOBase):
    def __init__(self, plan):
        self.plan, self.position = plan, 0
        with urlopen(Request(plan["archive_url"], method="HEAD"), timeout=30) as response:
            if int(response.headers["Content-Length"]) != plan["archive_size"]:
                raise ValueError("archive size changed")
            if response.headers["ETag"] != plan["archive_etag"]:
                raise ValueError("archive ETag changed")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.plan["archive_size"] + offset
        else:
            raise ValueError("invalid seek")
        return self.position

    def read(self, length=-1):
        size = self.plan["archive_size"]
        length = min(length if length >= 0 else size - self.position, size - self.position)
        data = get_range(
            self.plan["archive_url"], self.position, length, size, self.plan["archive_etag"]
        )
        self.position += len(data)
        return data


def select_entries(plan):
    names = []
    for sequence, length in plan["sequences"].items():
        anchors = [
            1 + (length - 61) * i // (plan["anchor_count_per_sequence"] - 1)
            for i in range(plan["anchor_count_per_sequence"])
        ]
        frames = sorted({a + gap for a in anchors for gap in [0, *plan["gaps"]]})
        prefix = f"MOT17/train/{sequence}"
        names.extend([f"{prefix}/gt/gt.txt", f"{prefix}/seqinfo.ini"])
        names.extend(f"{prefix}/img1/{f:06}.jpg" for f in frames)
    return names


def extract_entry(info, plan):
    start = info.header_offset
    header = get_range(plan["archive_url"], start, 30, plan["archive_size"], plan["archive_etag"])
    if header[:4] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local header")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    body = get_range(
        plan["archive_url"],
        start + 30,
        name_len + extra_len + info.compress_size,
        plan["archive_size"],
        plan["archive_etag"],
    )
    if body[:name_len].decode("utf-8") != info.filename:
        raise ValueError("ZIP member name mismatch")
    compressed = body[name_len + extra_len :]
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(compressed, info.file_size + 1)
        if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError("invalid compressed ZIP payload")
    else:
        raise ValueError("unsupported ZIP compression")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError("ZIP size/CRC mismatch")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    wanted = select_entries(plan)
    with zipfile.ZipFile(RemoteZip(plan)) as archive:
        infos = {name: archive.getinfo(name) for name in wanted}
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "assets.json"
    locked = json.loads(lock_path.read_text())["assets"] if lock_path.exists() else {}

    def provision(name):
        info = infos[name]
        target = (args.output / name).resolve()
        if not target.is_relative_to(args.output.resolve()):
            raise ValueError("ZIP path escapes output")
        data = target.read_bytes() if target.is_file() else extract_entry(info, plan)
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError(f"cached file size/CRC mismatch: {name}")
        fingerprint = sha(data)
        if name in locked and locked[name]["sha256"] != fingerprint:
            raise ValueError(f"cached/downloaded SHA-256 mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return name, {
            "sha256": fingerprint,
            "bytes": len(data),
            "crc32": info.CRC,
            "archive_offset": info.header_offset,
            "compressed_bytes": info.compress_size,
        }

    assets = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, fingerprint in pool.map(provision, wanted):
            assets[name] = fingerprint
            print(f"Verified {name}", flush=True)
    lock_path.write_text(
        json.dumps(
            {
                "plan_sha256": sha(args.plan.read_bytes()),
                "archive_url": plan["archive_url"],
                "archive_etag": plan["archive_etag"],
                "assets": assets,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
