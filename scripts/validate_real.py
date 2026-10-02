"""Explicit offline real-model validation; never downloads weights or images.

Run preparation separately, then run this inside an OS network namespace.
Writes evidence only to the explicitly selected output directory.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import deltasense
from deltasense import DeltaSense, InferenceError, InputError


def variants(source, directory):
    with Image.open(source) as original:
        original = original.convert("RGB")
    width, height = original.size
    examples = {"identical": original.copy()}
    for factor in (0.4, 0.7, 1.3, 1.8):
        examples[f"brightness_{factor}"] = ImageEnhance.Brightness(original).enhance(factor)
    for radius in (1, 3, 8):
        examples[f"blur_{radius}"] = original.filter(ImageFilter.GaussianBlur(radius))
    for fraction in (0.2, 0.5, 0.8):
        occluded = original.copy()
        start = int(width * (1 - fraction) / 2)
        ImageDraw.Draw(occluded).rectangle((start, 0, width - start, height), fill=(128, 128, 128))
        examples[f"occlusion_{fraction}"] = occluded
    for dx, dy in [(3, 2), (20, 15), (-20, -15)]:
        translated = Image.new("RGB", original.size, "white")
        translated.paste(original, (dx, dy))
        examples[f"image_translation_{dx}_{dy}"] = translated
    for factor in (0.8, 1.2):
        resized = original.resize((round(width * factor), round(height * factor)))
        scaled = Image.new("RGB", original.size, "white")
        scaled.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        examples[f"image_scale_{factor}"] = scaled
    examples["blank"] = Image.new("RGB", original.size, "white")
    examples["mirror"] = original.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    outputs = []
    for name, image in examples.items():
        path = directory / f"{name}.png"
        image.save(path)
        outputs.append((name, path))
    for quality in (30, 90):
        path = directory / f"jpeg_{quality}.jpg"
        original.save(path, quality=quality)
        outputs.append((f"jpeg_{quality}", path))
    path = directory / "exif_orientation.jpg"
    exif = Image.Exif()
    exif[274] = 6
    original.transpose(Image.Transpose.ROTATE_90).save(path, quality=95, exif=exif)
    outputs.append(("exif_orientation", path))
    return outputs


def check_summary(doc):
    before = doc["before"]["detections"]
    after = doc["after"]["detections"]
    counts_a = Counter(d["class_id"] for d in before)
    counts_b = Counter(d["class_id"] for d in after)
    for row in doc["prediction_counts"]:
        cls = row["class_id"]
        assert row["before"] == counts_a[cls] and row["after"] == counts_b[cls]
        assert row["delta"] == counts_b[cls] - counts_a[cls]
    assert len(doc["matches"]) + len(doc["unmatched_before"]) == len(before)
    assert len(doc["matches"]) + len(doc["unmatched_after"]) == len(after)
    assert len({m["before_id"] for m in doc["matches"]}) == len(doc["matches"])
    assert len({m["after_id"] for m in doc["matches"]}) == len(doc["matches"])
    for match in doc["matches"]:
        a, b = before[match["before_id"]], after[match["after_id"]]
        assert a["class_id"] == b["class_id"]
        aa, bb = np.asarray(a["bbox_xyxy"]), np.asarray(b["bbox_xyxy"])
        areas = [float(np.prod(box[2:] - box[:2])) for box in (aa, bb)]
        assert np.allclose(areas, [match["bbox_area"]["before"], match["bbox_area"]["after"]])
        displacement = (bb[:2] + bb[2:]) / 2 - (aa[:2] + aa[2:]) / 2
        assert np.allclose(displacement, [match["displacement"]["dx"], match["displacement"]["dy"]])
        if "mask_area" in match and match["mask_area"]["unavailable_reason"] is None:
            assert match["mask_area"]["before"] == a["mask_area"]
            assert match["mask_area"]["after"] == b["mask_area"]
    assert doc["physical_change_verified"] is False
    assert doc["unchecked_conditions"] == [
        "alignment",
        "illumination",
        "occlusion",
        "physical_identity",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-isolated-network", action="store_true")
    args = parser.parse_args()
    routes = Path("/proc/net/route").read_text().strip().splitlines()
    # Linux can return either an empty file or just the header when there are no routes.
    if args.require_isolated_network and len(routes) > 1:
        raise RuntimeError("expected an isolated Linux network namespace with no IPv4 routes")
    network_events = []

    def audit_network(event, values):
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.sendto",
        }:
            network_events.append(event)
            raise RuntimeError("unexpected network operation during local inference")

    sys.addaudithook(audit_network)
    args.output.mkdir(parents=True, exist_ok=True)
    originals = sorted(args.sources.glob("*.png"))
    initial_digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    inputs = []
    for scene in ("astronaut", "coffee", "chelsea"):
        source = args.sources / f"{scene}.png"
        directory = args.output / "inputs" / scene
        directory.mkdir(parents=True, exist_ok=True)
        inputs.extend((scene, name, source, path) for name, path in variants(source, directory))
    inputs.extend(
        [
            (
                "apollo",
                "unchanged_recapture",
                args.sources / "apollo-120.png",
                args.sources / "apollo-121.png",
            ),
            (
                "apollo",
                "visible_subject_change",
                args.sources / "apollo-030.png",
                args.sources / "apollo-120.png",
            ),
        ]
    )
    report = {
        "deltasense_import": str(Path(deltasense.__file__).resolve()),
        "library_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(deltasense.__file__).parent.glob("*.py"))
        },
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ["deltasense", "ultralytics", "torch", "numpy", "Pillow"]
        },
        "source_sha256": initial_digests,
        "network_isolation_required": args.require_isolated_network,
        "ipv4_route_count": max(0, len(routes) - 1),
        "cases": [],
        "checks": [],
        "limits": [
            "Small demonstration, not a model-accuracy benchmark.",
            "Static-photo variants are controlled digital edits, not independent camera captures.",
            "Apollo frames are distinct historical captures with fixed framing; "
            "no visible localized subject change is apparent in the recapture pair. "
            "This is not ground truth of physical invariance.",
            "Camera alignment, illumination, occlusion and physical identity remain unchecked.",
        ],
    }
    for task, filename in [("detect", "yolo26n.pt"), ("segment", "yolo26n-seg.pt")]:
        sense = DeltaSense(args.models / filename)
        for scene, name, before, after in inputs:
            started = time.perf_counter()
            result = sense.compare(before, after)
            doc = json.loads(result.to_json())
            check_summary(doc)
            # Identical pixels are not a promise of bit-identical batch inference.
            # Preserve observed numerical differences; A01's exact arithmetic is
            # tested with independently specified identical observation fixtures.
            target = args.output / f"{task}-{scene}-{name}"
            target.with_name(target.name + ".json").write_text(result.to_json() + "\n")
            result.plot().save(target.with_name(target.name + ".png"))
            reverse = json.loads(sense.compare(after, before).to_json())
            check_summary(reverse)
            assert [c["delta"] for c in doc["prediction_counts"]] == [
                -c["delta"] for c in reverse["prediction_counts"]
            ]
            assert {(m["before_id"], m["after_id"]) for m in doc["matches"]} == {
                (m["after_id"], m["before_id"]) for m in reverse["matches"]
            }
            report["cases"].append(
                {
                    "task": task,
                    "scene": scene,
                    "case": name,
                    "prediction_counts": doc["prediction_counts"],
                    "accepted_pairs": len(doc["matches"]),
                    "ambiguous_edges": len(doc["ambiguous_candidates"]),
                    "elapsed_seconds_forward_plot_and_reverse": time.perf_counter() - started,
                    "checkpoint_sha256": doc["provenance"]["checkpoint_sha256"],
                }
            )
        source = args.sources / "coffee.png"
        for label, settings in [
            ("confidence_1", {"conf": 1.0}),
            ("max_det_1", {"max_det": 1}),
            ("class_filter", {"classes": [41]}),
            ("non_stride_imgsz", {"imgsz": 641}),
        ]:
            doc = json.loads(
                DeltaSense(args.models / filename, **settings).compare(source, source).to_json()
            )
            check_summary(doc)
            for side in ["before", "after"]:
                if label == "confidence_1":
                    assert doc[side]["detections"] == []
                elif label == "max_det_1":
                    assert len(doc[side]["detections"]) <= 1
                elif label == "class_filter":
                    assert all(d["class_id"] == 41 for d in doc[side]["detections"])
            if label == "non_stride_imgsz":
                assert doc["provenance"]["inference_settings"]["imgsz"] == [672, 672]
            report["checks"].append({"task": task, "check": label, "passed": True})
        corrupt = args.output / "corrupt.png"
        corrupt.write_bytes(b"\x89PNG\r\n\x1a\ncorrupt")
        try:
            sense.compare(corrupt, source)
        except InputError:
            report["checks"].append(
                {"task": task, "check": "corrupt_after_upstream_import", "passed": True}
            )
        else:
            raise AssertionError("corrupt input returned success")
        invalid_device = DeltaSense(args.models / filename, device="not-a-device")
        try:
            invalid_device.compare(source, source)
        except InferenceError:
            report["checks"].append({"task": task, "check": "no_device_fallback", "passed": True})
        else:
            raise AssertionError("unsupported device silently fell back")
        print(
            f"{task}: {len(inputs)} forward/reverse pairs and settings/error checks passed",
            flush=True,
        )
    assert initial_digests == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals
    }
    report["source_images_unchanged"] = True
    assert network_events == [], (
        "a dependency attempted networking, even if it suppressed the error"
    )
    report["network_events"] = network_events
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
