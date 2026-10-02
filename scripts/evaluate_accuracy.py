"""Run the frozen, explicitly scoped pilot benchmark using local weights only."""

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

from accuracy_metrics import aggregate, score_pair, validate_manifest
from PIL import Image, ImageDraw

import deltasense
from deltasense import DeltaSense


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-isolated-network", action="store_true")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--review-every", type=int, default=1)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    args = parser.parse_args()
    if args.plot_every < 1 or args.review_every < 1:
        raise ValueError("plot/review intervals must be positive")
    if args.output.exists():
        raise ValueError("choose a new output directory; do not overwrite evaluation evidence")
    manifest_bytes = args.manifest.read_bytes()
    data = json.loads(manifest_bytes)
    frames = validate_manifest(data)
    settings = dict(data["settings"])
    if args.imgsz is not None:
        settings["imgsz"] = args.imgsz
    if args.conf is not None:
        settings["conf"] = args.conf
    routes = Path("/proc/net/route").read_text().strip().splitlines()
    if args.require_isolated_network and len(routes) > 1:
        raise ValueError("network namespace has IPv4 routes")
    network_events = []

    def audit(event, values):
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.sendto",
        }:
            network_events.append(event)
            raise RuntimeError("evaluation attempted networking")

    sys.addaudithook(audit)
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_bytes(manifest_bytes)
    files = {}
    for index, (name, frame) in enumerate(frames.items()):
        path = (args.assets / frame["path"]).resolve()
        if not path.is_relative_to(args.assets.resolve()) or digest(path) != frame["sha256"]:
            raise ValueError(f"asset path/digest mismatch: {name}")
        with Image.open(path) as original:
            if list(original.size) != frame["size"]:
                raise ValueError(f"asset dimensions mismatch: {name}")
            review = original.convert("RGB")
        draw = ImageDraw.Draw(review)
        for obj in frame.get("ignore_objects", []):
            draw.rectangle(obj["bbox_xyxy"], outline="orange", width=2)
        for obj in frame["objects"]:
            draw.rectangle(obj["bbox_xyxy"], outline="lime", width=2)
            draw.text(tuple(obj["bbox_xyxy"][:2]), obj["object_id"], fill="lime")
        if not frame["objects"]:
            draw.text((5, 5), "GT: no target object", fill="yellow")
        if index % args.review_every == 0:
            review.save(args.output / f"truth-{name}.png")
        files[name] = path
    report = {
        "annotation_status": data["annotation_status"],
        "independent_human_review": data["independent_human_review"],
        "dataset_authors_human_annotations": data.get("dataset_authors_human_annotations", False),
        "source_statistics": data.get("source_statistics"),
        "manifest_sha256": digest(args.manifest),
        "sampling": data["sampling"],
        "evaluation": data["evaluation"],
        "settings": settings,
        "manifest_baseline_settings": data["settings"],
        "deltasense_import": str(Path(deltasense.__file__).resolve()),
        "versions": {
            n: version(n) for n in ("deltasense", "ultralytics", "torch", "numpy", "Pillow")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": {
            p.name: digest(p) for p in sorted(Path(deltasense.__file__).parent.glob("*.py"))
        },
        "evaluation_script_sha256": {
            name: digest(Path(__file__).parent / name)
            for name in ("evaluate_accuracy.py", "accuracy_metrics.py")
        },
        "network_isolation_required": args.require_isolated_network,
        "models": {},
    }
    for task, model in [("detect", "yolo26n.pt"), ("segment", "yolo26n-seg.pt")]:
        sense = DeltaSense(args.models / model, **settings)
        observations = []
        folder = args.output / task
        folder.mkdir()
        inference_seconds = []
        for index, pair in enumerate(data["pairs"]):
            started = time.perf_counter()
            result = sense.compare(files[pair["before"]], files[pair["after"]])
            inference_seconds.append(time.perf_counter() - started)
            doc = result.to_dict()
            (folder / f"{pair['id']}.json").write_text(result.to_json(indent=2) + "\n")
            if index % args.plot_every == 0:
                result.plot().save(folder / f"{pair['id']}.png")
            observations.append((pair, doc))
            if (index + 1) % 12 == 0:
                print(f"{task}: {index + 1}/{len(data['pairs'])} comparisons", flush=True)
        scored = {}
        for threshold in data["evaluation"]["sensitivity_ious"]:
            rows, unique = [], {}
            for pair, doc in observations:
                a, b = frames[pair["before"]], frames[pair["after"]]
                row = {
                    "pair_id": pair["id"],
                    "stratum": pair["stratum"],
                    "scene": a["scene"],
                    **score_pair(doc, a, b, threshold),
                }
                rows.append(row)
                unique.setdefault(a["id"], row["detection_before"])
                unique.setdefault(b["id"], row["detection_after"])
            scored[str(threshold)] = {
                "summary": aggregate(rows, unique),
                "by_stratum": {
                    name: aggregate([r for r in rows if r["stratum"] == name], {})
                    for name in sorted({r["stratum"] for r in rows})
                },
                "by_scene": {
                    scene: aggregate(
                        [r for r in rows if r["scene"] == scene],
                        {
                            name: metric
                            for name, metric in unique.items()
                            if frames[name]["scene"] == scene
                        },
                    )
                    for scene in sorted({r["scene"] for r in rows})
                },
                "cases": rows,
            }
        report["models"][task] = {
            "checkpoint_sha256": digest(args.models / model),
            "provenance": observations[0][1]["provenance"],
            "scores_by_localization_iou": scored,
            "compare_elapsed_seconds_includes_first_model_load": inference_seconds,
        }
        print(task, json.dumps(scored["0.5"]["summary"]), flush=True)
    if any(digest(files[name]) != frames[name]["sha256"] for name in files):
        raise RuntimeError("source image changed during benchmark")
    if digest(args.manifest) != report["manifest_sha256"]:
        raise RuntimeError("annotations changed during benchmark")
    if network_events:
        raise RuntimeError("network operation detected")
    report.update(
        source_images_unchanged=True, annotations_unchanged=True, network_events=network_events
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
