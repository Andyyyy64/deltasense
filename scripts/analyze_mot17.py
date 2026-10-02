"""Attribute correspondence failures in immutable saved MOT17 predictions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from accuracy_metrics import assign_to_truth


def attribution(doc, before, after):
    a, _ = assign_to_truth(doc["before"]["detections"], before["objects"], [0], 0.5)
    b, _ = assign_to_truth(doc["after"]["detections"], after["objects"], [0], 0.5)
    reverse_a, reverse_b = {v: k for k, v in a.items()}, {v: k for k, v in b.items()}
    expected = {g["object_id"] for g in before["objects"]} & {
        g["object_id"] for g in after["objects"]
    }
    accepted = {(m["before_id"], m["after_id"]) for m in doc["matches"] if m["class_id"] == 0}
    reasons_a = {u["id"]: u["reason"] for u in doc["unmatched_before"]}
    reasons_b = {u["id"]: u["reason"] for u in doc["unmatched_after"]}
    categories = Counter()
    for identity in expected:
        if identity not in reverse_a or identity not in reverse_b:
            categories["recognition_or_localization_missing"] += 1
            continue
        i, j = reverse_a[identity], reverse_b[identity]
        if (i, j) in accepted:
            categories["correct"] += 1
        elif "ambiguous_correspondence" in (reasons_a.get(i), reasons_b.get(j)):
            categories["competing_candidates"] += 1
        elif any(x == i or y == j for x, y in accepted):
            categories["accepted_different_identity"] += 1
        else:
            categories["no_accepted_candidate"] += 1
    return dict(categories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("analysis evidence must not be overwritten")
    data = json.loads(args.manifest.read_text())
    frames = {f["id"]: f for f in data["images"]}
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    analyses = {}
    for run in args.runs:
        report = json.loads((run / "report.json").read_text())
        if report["manifest_sha256"] != manifest_sha:
            raise ValueError("run uses different ground truth")
        models = {}
        for task, model in report["models"].items():
            total, scenes, gaps, cases = Counter(), {}, {}, []
            for pair in data["pairs"]:
                doc = json.loads((run / task / f"{pair['id']}.json").read_text())
                reasons = attribution(doc, frames[pair["before"]], frames[pair["after"]])
                total.update(reasons)
                scenes.setdefault(pair["scene"], Counter()).update(reasons)
                gaps.setdefault(pair["stratum"], Counter()).update(reasons)
                cases.append({"pair_id": pair["id"], "categories": reasons})
            summary = model["scores_by_localization_iou"]["0.5"]["summary"]
            if total["correct"] != summary["correspondence"]["tp"]:
                raise ValueError("attribution differs from scoring")
            if (
                sum(total.values())
                != summary["correspondence"]["tp"] + summary["correspondence"]["fn"]
            ):
                raise ValueError("attribution did not partition expected identities")
            models[task] = {
                "summary": summary,
                "attribution": dict(total),
                "by_scene": {k: dict(v) for k, v in scenes.items()},
                "by_gap": {k: dict(v) for k, v in gaps.items()},
                "cases": cases,
            }
        analyses[run.name] = {
            "settings": report["settings"],
            "models": models,
            "report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"manifest_sha256": manifest_sha, "runs": analyses}, indent=2) + "\n"
    )
    for name, run in analyses.items():
        for task, model in run["models"].items():
            print(name, task, model["attribution"], flush=True)


if __name__ == "__main__":
    main()
