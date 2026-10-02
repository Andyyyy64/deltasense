"""Explore bbox matching thresholds on immutable saved predictions, offline.

Reconstructs bbox-only observations. Segmentation masks are not reconstructed
or scored. The original 0.2 candidate graph must agree before any sweep.
"""

import argparse
import hashlib
import json
from pathlib import Path

from accuracy_metrics import aggregate, score_pair

from deltasense.comparison import ComparisonSettings, compare_observations
from deltasense.observations import Detection, ImageIdentity, Observation


def reconstruct(side):
    identity = side["image"]
    return Observation(
        ImageIdentity(identity["sha256"], identity["width"], identity["height"]),
        "detect",
        tuple(
            Detection(d["class_id"], d["class_name"], d["confidence"], tuple(d["bbox_xyxy"]))
            for d in side["detections"]
        ),
    )


def rematch(doc, threshold):
    return {
        **doc,
        **compare_observations(
            reconstruct(doc["before"]), reconstruct(doc["after"]), ComparisonSettings(threshold)
        ),
    }


def verify_baseline(doc, rebuilt):
    keys = ("before_id", "after_id", "iou")
    if [[m[k] for k in keys] for m in doc["matches"]] != [
        [m[k] for k in keys] for m in rebuilt["matches"]
    ]:
        raise ValueError("saved and reconstructed baseline correspondences differ")
    for name in (
        "ambiguous_candidates",
        "unmatched_before",
        "unmatched_after",
        "prediction_counts",
    ):
        if doc[name] != rebuilt[name]:
            raise ValueError(f"saved and reconstructed baseline {name} differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("sweep evidence must not be overwritten")
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    data = json.loads(args.manifest.read_text())
    frames = {f["id"]: f for f in data["images"]}
    report = {
        "manifest_sha256": manifest_sha,
        "method": (
            "BBox-only reconstruction; original candidate graph verified; no inference, "
            "no mask evaluation. Exploratory reuse of the same labeled training subset; "
            "library defaults unchanged."
        ),
        "runs": {},
    }
    for run in args.runs:
        original = json.loads((run / "report.json").read_text())
        if original["manifest_sha256"] != manifest_sha:
            raise ValueError("run uses different ground truth")
        models = {}
        for task in original["models"]:
            documents, fingerprints = [], {}
            for pair in data["pairs"]:
                path = run / task / f"{pair['id']}.json"
                doc = json.loads(path.read_text())
                fingerprints[pair["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
                verify_baseline(doc, rematch(doc, original["settings"]["match_iou"]))
                documents.append((pair, doc))
            trials = {}
            for threshold in (0.2, 0.3, 0.5, 0.7):
                rows, unique = [], {}
                for pair, doc in documents:
                    a, b = frames[pair["before"]], frames[pair["after"]]
                    row = {
                        "pair_id": pair["id"],
                        "stratum": pair["stratum"],
                        "scene": a["scene"],
                        **score_pair(rematch(doc, threshold), a, b, 0.5),
                    }
                    rows.append(row)
                    unique.setdefault(a["id"], row["detection_before"])
                    unique.setdefault(b["id"], row["detection_after"])
                trials[str(threshold)] = {
                    "summary": aggregate(rows, unique),
                    "by_gap": {
                        name: aggregate([r for r in rows if r["stratum"] == name], {})
                        for name in sorted({r["stratum"] for r in rows})
                    },
                    "cases": rows,
                }
                print(
                    run.name,
                    task,
                    threshold,
                    trials[str(threshold)]["summary"]["correspondence"],
                    flush=True,
                )
            for pair, _ in documents:
                if (
                    hashlib.sha256((run / task / f"{pair['id']}.json").read_bytes()).hexdigest()
                    != fingerprints[pair["id"]]
                ):
                    raise ValueError("source prediction changed during sweep")
            models[task] = {"trials": trials, "prediction_sha256": fingerprints}
        report["runs"][run.name] = {"inference_settings": original["settings"], "models": models}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
