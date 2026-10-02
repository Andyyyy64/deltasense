"""Read-only verification of a local candidate's wheel, sdist and source identity."""

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [root / "README.md", root / "CHANGELOG.md", root / "pyproject.toml"]
    for folder in ("src", "tests", "scripts", "docs", "evaluation", ".github/workflows"):
        sources.extend(
            p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    source_bytes = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(sources)}
    source_hashes = {name: digest(data) for name, data in source_bytes.items()}
    report = {
        "source_sha256": source_hashes,
        "source_tree_sha256": digest(json.dumps(source_hashes, sort_keys=True).encode()),
        "artifacts": {},
    }
    wheels = list(args.dist.glob("*.whl"))
    sdists = list(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    for path in (*wheels, *sdists):
        contents = path.read_bytes()
        report["artifacts"][path.name] = {"sha256": digest(contents), "bytes": len(contents)}
    with tarfile.open(sdists[0], "r:gz") as archive:
        members = archive.getmembers()
        if any(m.issym() or m.islnk() or ".." in PurePosixPath(m.name).parts for m in members):
            raise ValueError("unexpected link or traversal path in sdist")
        files = {
            PurePosixPath(m.name)
            .relative_to(PurePosixPath(m.name).parts[0])
            .as_posix(): archive.extractfile(m).read()
            for m in members
            if m.isfile()
        }
    for name, contents in source_bytes.items():
        if files.get(name) != contents:
            raise ValueError(f"sdist differs from candidate source: {name}")
    forbidden = {".pt", ".png", ".jpg", ".jpeg", ".mov", ".mp4", ".npy", ".npz"}
    if any(PurePosixPath(name).suffix.lower() in forbidden for name in files):
        raise ValueError("model weights or runtime media bundled in sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_files = {name: archive.read(name) for name in archive.namelist()}
    expected = {
        name.removeprefix("src/"): data
        for name, data in source_bytes.items()
        if name.startswith("src/deltasense/")
    }
    actual = {name: data for name, data in wheel_files.items() if ".dist-info/" not in name}
    if expected != actual:
        raise ValueError("wheel package differs from candidate source")
    metadata = BytesParser().parsebytes(wheel_files["deltasense-0.1.0.dist-info/METADATA"])
    if metadata["Name"] != "deltasense" or metadata["Version"] != "0.1.0":
        raise ValueError("unexpected wheel package identity")
    report.update(
        sdist_sources_match=True,
        wheel_sources_match=True,
        no_bundled_weights_or_runtime_media=True,
        requires_python=metadata["Requires-Python"],
        requires_dist=metadata.get_all("Requires-Dist"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified wheel, sdist and {len(source_bytes)} source files", flush=True)


if __name__ == "__main__":
    main()
