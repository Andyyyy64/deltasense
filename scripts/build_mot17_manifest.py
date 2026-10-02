"""Convert pinned MOT17 annotations into the local accuracy manifest, offline.

Uses active pedestrian (native class 1) boxes/IDs. Distractors 2/7/8/12 and
inactive pedestrian annotations are explicit ignores. This is a diagnostic
two-image evaluation, not MOTChallenge's official tracking evaluation.
"""

import argparse
import configparser
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from accuracy_metrics import validate_manifest


def convert_box(row, width, height):
    x, y, w, h = map(float, row[2:6])
    # MOT's published convention is 1-based. DeltaSense uses 0-based image coordinates.
    box = [max(0, x - 1), max(0, y - 1), min(width, x - 1 + w), min(height, y - 1 + h)]
    return box if box[0] < box[2] and box[1] < box[3] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new manifest path; frozen labels must not be overwritten")
    plan = json.loads(args.plan.read_text())
    lock = json.loads(args.lock.read_text())
    if lock["plan_sha256"] != hashlib.sha256(args.plan.read_bytes()).hexdigest():
        raise ValueError("sampling plan differs from the asset lock")
    for name, expected in lock["assets"].items():
        path = (args.assets / name).resolve()
        if not path.is_relative_to(args.assets.resolve()):
            raise ValueError("asset path escapes directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"asset digest mismatch: {name}")
    images, pairs, statistics = [], [], {}
    for sequence, length in plan["sequences"].items():
        prefix = f"MOT17/train/{sequence}"
        config = configparser.ConfigParser()
        config.read(args.assets / prefix / "seqinfo.ini")
        section = config["Sequence"]
        if section.getint("seqLength") != length:
            raise ValueError("sequence length differs from frozen plan")
        width, height = section.getint("imWidth"), section.getint("imHeight")
        annotations = {}
        with (args.assets / prefix / "gt/gt.txt").open() as stream:
            for row in csv.reader(stream):
                if len(row) != 9:
                    raise ValueError("expected nine MOT17 ground-truth columns")
                annotations.setdefault(int(row[0]), []).append(row)
        anchors = [
            1 + (length - 61) * i // (plan["anchor_count_per_sequence"] - 1)
            for i in range(plan["anchor_count_per_sequence"])
        ]
        selected = sorted({a + gap for a in anchors for gap in [0, *plan["gaps"]]})
        classes, excluded = Counter(), Counter()
        for number in selected:
            path = f"{prefix}/img1/{number:06}.jpg"
            objects, ignores = [], []
            for row in annotations.get(number, []):
                native_class, active = int(row[7]), int(row[6])
                classes[native_class] += 1
                box = convert_box(row, width, height)
                if box is None:
                    excluded["box_outside_frame"] += 1
                    continue
                visibility = float(row[8])
                if not 0 <= visibility <= 1:
                    raise ValueError("invalid native visibility")
                obj = {
                    "object_id": f"track-{int(row[1])}",
                    "class_id": 0,
                    "bbox_xyxy": box,
                    "visibility": visibility,
                    "native_class_id": native_class,
                    "native_active": active,
                    "native_bbox_xywh": list(map(float, row[2:6])),
                }
                if native_class == 1 and active != 0:
                    objects.append(obj)
                elif native_class in {2, 7, 8, 12} or (native_class == 1 and active == 0):
                    ignores.append({**obj, "reason": "native_distractor_or_inactive_pedestrian"})
                else:
                    excluded[f"non_person_native_class_{native_class}"] += 1
            images.append(
                {
                    "id": f"{sequence}-{number:06}",
                    "scene": sequence,
                    "path": path,
                    "sha256": lock["assets"][path]["sha256"],
                    "size": [width, height],
                    "scope_class_ids": [0],
                    "objects": objects,
                    "ignore_objects": ignores,
                    "frame_number": number,
                    "fps": section.getint("frameRate"),
                }
            )
        for anchor in anchors:
            for gap in plan["gaps"]:
                pairs.append(
                    {
                        "id": f"{sequence}-{anchor:06}-{anchor + gap:06}",
                        "before": f"{sequence}-{anchor:06}",
                        "after": f"{sequence}-{anchor + gap:06}",
                        "scene": sequence,
                        "stratum": f"gap_{gap:02}_frames",
                        "frame_gap": gap,
                    }
                )
        statistics[sequence] = {
            "selected_frames": len(selected),
            "native_classes": dict(classes),
            "excluded_rows": dict(excluded),
            "ground_truth_sha256": lock["assets"][f"{prefix}/gt/gt.txt"]["sha256"],
        }
    data = {
        "schema_version": "1",
        "annotation_status": "published_mot17_human_annotations",
        "independent_human_review": False,
        "dataset_authors_human_annotations": True,
        "annotation_method": (
            "MOT17 official human annotations. No new human review of conversion. "
            "Active pedestrian boxes/IDs; 1-based to 0-based conversion and clipping to frame. "
            "Native distractors/inactive pedestrians are explicit ignores. "
            "Amodal annotated boxes, not segmentation or physical truth."
        ),
        "frozen_before_new_inference": True,
        "sampling": plan["sampling"],
        "sources": ["https://motchallenge.net/data/MOT17/", "https://arxiv.org/html/2010.07548v2"],
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "asset_lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        "source_statistics": statistics,
        "settings": plan["model_settings"],
        "evaluation": {
            "localization_iou": 0.5,
            "sensitivity_ious": [0.3, 0.5, 0.7],
            "ignore_policy": (
                "Match valid targets first; remaining predictions match ignored boxes "
                "one-to-one at IoU >=0.5. Remove ignored predictions from detection/link "
                "metrics, but preserve raw counts. Custom diagnostic policy, "
                "not official MOT scoring."
            ),
            "count_change_rule": (
                "Any nonzero raw person-prediction delta, compared with active annotated "
                "pedestrian count delta. Raw model also includes seated people and other "
                "dataset distractors; not a calibrated physical-change false-alarm rate."
            ),
            "geometry": (
                "Conditional bbox-center displacement error on GT-supported identities, "
                "image pixels only; no mask or physical-unit truth."
            ),
        },
        "images": images,
        "pairs": pairs,
    }
    validate_manifest(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps({"images": len(images), "pairs": len(pairs), "statistics": statistics}, indent=2)
    )


if __name__ == "__main__":
    main()
