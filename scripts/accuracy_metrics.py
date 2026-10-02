"""Developer-only accuracy scoring against explicitly scoped visual annotations.

No production matching code is reused here. Count-change signals are an
evaluation proxy, not a physical-change claim by DeltaSense.
"""

import math
from collections import Counter


def overlap_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def pr_counts(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": fraction(tp, tp + fp),
        "recall": fraction(tp, tp + fn),
    }


def maximum_assignment(scores):
    """Maximize eligible pair count, then total IoU, with dummy unmatched columns.

    Rectangular Hungarian algorithm; zero means ineligible. Deterministic ties.
    Separate evaluation implementation, no production matching dependency.
    """
    n = len(scores)
    if not n:
        return {}
    targets = len(scores[0])
    m = targets + n
    boost = n + 1
    costs = [[-(boost + s) if s > 0 else 0.0 for s in row] + [0.0] * n for row in scores]
    u, v, owners, previous = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        owners[0], column = i, 0
        minimum, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[column] = True
            row = owners[column]
            delta, next_column = math.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cost = costs[row - 1][j - 1] - u[row] - v[j]
                if cost < minimum[j]:
                    minimum[j], previous[j] = cost, column
                if minimum[j] < delta:
                    delta, next_column = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owners[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if owners[column] == 0:
                break
        while column:
            owners[column] = owners[previous[column]]
            column = previous[column]
    return {
        owners[j] - 1: j - 1
        for j in range(1, targets + 1)
        if owners[j] and scores[owners[j] - 1][j - 1] > 0
    }


def assign_to_truth(detections, truth, scope, threshold):
    """Same-class maximum-cardinality / maximum-IoU assignment to labeled boxes."""
    selected = [d for d in detections if d["class_id"] in scope]
    scores = []
    for d in selected:
        row = []
        for g in truth:
            iou = overlap_iou(d["bbox_xyxy"], g["bbox_xyxy"])
            row.append(iou if d["class_id"] == g["class_id"] and iou >= threshold else 0.0)
        scores.append(row)
    pairs = maximum_assignment(scores)
    assignments = {selected[p]["id"]: truth[g]["object_id"] for p, g in pairs.items()}
    return assignments, pr_counts(len(pairs), len(selected) - len(pairs), len(truth) - len(pairs))


def score_observation(detections, frame, threshold):
    scope = frame["scope_class_ids"]
    assignments, metrics = assign_to_truth(detections, frame["objects"], scope, threshold)
    remaining = [d for d in detections if d["id"] not in assignments]
    ignored, _ = assign_to_truth(remaining, frame.get("ignore_objects", []), scope, 0.5)
    metrics = pr_counts(metrics["tp"], metrics["fp"] - len(ignored), metrics["fn"])
    metrics["ignored_predictions"] = len(ignored)
    by_visibility = {}
    for obj in frame["objects"]:
        visibility = obj.get("visibility")
        group = (
            "unknown"
            if visibility is None
            else "low"
            if visibility < 0.3
            else "medium"
            if visibility < 0.7
            else "high"
        )
        bucket = by_visibility.setdefault(group, {"tp": 0, "fn": 0})
        bucket["tp" if obj["object_id"] in assignments.values() else "fn"] += 1
    metrics["by_visibility"] = by_visibility
    return assignments, metrics, set(ignored)


def score_pair(doc, before, after, threshold):
    scope = before["scope_class_ids"]
    assignments_a, detection_a, ignored_a = score_observation(
        doc["before"]["detections"], before, threshold
    )
    assignments_b, detection_b, ignored_b = score_observation(
        doc["after"]["detections"], after, threshold
    )
    truth_a = {g["object_id"] for g in before["objects"]}
    truth_b = {g["object_id"] for g in after["objects"]}
    expected_links = truth_a & truth_b
    correct_links = set()
    false_links, geometry = [], []
    ignored_links = 0
    for m in doc["matches"]:
        if m["class_id"] not in scope:
            continue
        if m["before_id"] in ignored_a or m["after_id"] in ignored_b:
            ignored_links += 1
            continue
        a, b = assignments_a.get(m["before_id"]), assignments_b.get(m["after_id"])
        if a is not None and a == b:
            correct_links.add(a)
            if "displacement" in m:
                ga = next(g for g in before["objects"] if g["object_id"] == a)["bbox_xyxy"]
                gb = next(g for g in after["objects"] if g["object_id"] == b)["bbox_xyxy"]
                expected_dx = (gb[0] + gb[2] - ga[0] - ga[2]) / 2
                expected_dy = (gb[1] + gb[3] - ga[1] - ga[3]) / 2
                geometry.append(
                    {
                        "object_id": a,
                        "expected_dx": expected_dx,
                        "expected_dy": expected_dy,
                        "predicted_dx": m["displacement"]["dx"],
                        "predicted_dy": m["displacement"]["dy"],
                        "displacement_error_pixels": math.hypot(
                            m["displacement"]["dx"] - expected_dx,
                            m["displacement"]["dy"] - expected_dy,
                        ),
                    }
                )
        else:
            false_links.append({"before_id": m["before_id"], "after_id": m["after_id"]})
    localized_both = expected_links & set(assignments_a.values()) & set(assignments_b.values())
    truth_counts = [Counter(o["class_id"] for o in frame["objects"]) for frame in (before, after)]
    predicted = [
        Counter(d["class_id"] for d in doc[side]["detections"]) for side in ("before", "after")
    ]
    counts = [
        {
            "class_id": cls,
            "truth_before": truth_counts[0][cls],
            "truth_after": truth_counts[1][cls],
            "truth_delta": truth_counts[1][cls] - truth_counts[0][cls],
            "predicted_before": predicted[0][cls],
            "predicted_after": predicted[1][cls],
            "predicted_delta": predicted[1][cls] - predicted[0][cls],
        }
        for cls in scope
    ]
    true_signal = any(c["truth_delta"] != 0 for c in counts)
    predicted_signal = any(c["predicted_delta"] != 0 for c in counts)
    return {
        "detection_before": detection_a,
        "detection_after": detection_b,
        "correspondence": {
            **pr_counts(len(correct_links), len(false_links), len(expected_links - correct_links)),
            "expected_links": len(expected_links),
            "localized_at_both_ends": len(localized_both),
            "conditional_recall": fraction(len(correct_links), len(localized_both)),
            "false_links": false_links,
            "ignored_links": ignored_links,
        },
        "counts": counts,
        "signed_count_vector_correct": all(
            c["truth_delta"] == c["predicted_delta"] for c in counts
        ),
        "count_change_proxy": {"truth_signal": true_signal, "predicted_signal": predicted_signal},
        "ignored_prediction_ids": {"before": sorted(ignored_a), "after": sorted(ignored_b)},
        "conditional_geometry": geometry,
        "out_of_scope_predictions": {
            side: [d for d in doc[side]["detections"] if d["class_id"] not in scope]
            for side in ("before", "after")
        },
    }


def aggregate(rows, unique_frame_metrics):
    detection = pr_counts(
        *(sum(row[k] for row in unique_frame_metrics.values()) for k in ("tp", "fp", "fn"))
    )
    links = pr_counts(*(sum(r["correspondence"][k] for r in rows) for k in ("tp", "fp", "fn")))
    localized = sum(r["correspondence"]["localized_at_both_ends"] for r in rows)
    links.update(
        localized_at_both_ends=localized,
        conditional_recall=fraction(links["tp"], localized),
    )
    detection["ignored_predictions"] = sum(
        r.get("ignored_predictions", 0) for r in unique_frame_metrics.values()
    )
    visibility = {}
    for r in unique_frame_metrics.values():
        for name, bucket in r.get("by_visibility", {}).items():
            totals = visibility.setdefault(name, {"tp": 0, "fn": 0})
            for k in totals:
                totals[k] += bucket[k]
    detection["by_visibility"] = {
        name: {**bucket, "recall": fraction(bucket["tp"], bucket["tp"] + bucket["fn"])}
        for name, bucket in visibility.items()
    }
    links["ignored_links"] = sum(r["correspondence"].get("ignored_links", 0) for r in rows)
    confusion = Counter()
    for r in rows:
        truth = r["count_change_proxy"]["truth_signal"]
        pred = r["count_change_proxy"]["predicted_signal"]
        confusion["tp" if truth and pred else "fn" if truth else "fp" if pred else "tn"] += 1
    change = pr_counts(confusion["tp"], confusion["fp"], confusion["fn"])
    change.update(
        tn=confusion["tn"],
        false_alarm_rate=fraction(confusion["fp"], confusion["fp"] + confusion["tn"]),
        miss_rate=fraction(confusion["fn"], confusion["fn"] + confusion["tp"]),
    )
    errors = [abs(c["predicted_delta"] - c["truth_delta"]) for r in rows for c in r["counts"]]
    geometry_errors = [
        g["displacement_error_pixels"] for r in rows for g in r.get("conditional_geometry", [])
    ]
    return {
        "pair_count": len(rows),
        "unique_frame_count": len(unique_frame_metrics),
        "detection_on_first_observation_of_each_unique_frame": detection,
        "correspondence": links,
        "count_change_proxy": change,
        "conditional_displacement": {
            "correct_links_with_geometry": len(geometry_errors),
            "mean_error_pixels": fraction(sum(geometry_errors), len(geometry_errors)),
            "maximum_error_pixels": max(geometry_errors, default=None),
        },
        "signed_count_vector": {
            "correct_pairs": sum(r["signed_count_vector_correct"] for r in rows),
            "total_pairs": len(rows),
            "accuracy": fraction(sum(r["signed_count_vector_correct"] for r in rows), len(rows)),
            "class_pair_count": len(errors),
            "delta_mae": fraction(sum(errors), len(errors)),
        },
    }


def validate_manifest(data):
    """Reject incomplete/contradictory ground truth rather than infer labels."""
    images = {}
    identities = {}
    for frame in data["images"]:
        if frame["id"] in images:
            raise ValueError("duplicate image ID")
        images[frame["id"]] = frame
        width, height = frame["size"]
        if not all(type(v) is int and v > 0 for v in (width, height)):
            raise ValueError("invalid dimensions")
        scope = frame["scope_class_ids"]
        if (
            not scope
            or len(set(scope)) != len(scope)
            or any(type(c) is not int or c < 0 for c in scope)
        ):
            raise ValueError("invalid class scope")
        ids = set()
        all_objects = [*frame["objects"], *frame.get("ignore_objects", [])]
        for obj in all_objects:
            if not obj["object_id"] or obj["object_id"] in ids:
                raise ValueError("duplicate/empty object ID")
            ids.add(obj["object_id"])
            if type(obj["class_id"]) is not int or obj["class_id"] not in scope:
                raise ValueError("annotation outside target scope")
            box = obj["bbox_xyxy"]
            if len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
                raise ValueError("invalid annotation box")
            if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
                raise ValueError("annotation box out of bounds or empty")
            key = (frame["scene"], obj["object_id"])
            if identities.setdefault(key, obj["class_id"]) != obj["class_id"]:
                raise ValueError("identity changes class")
    pairs = set()
    for pair in data["pairs"]:
        if pair["id"] in pairs:
            raise ValueError("duplicate pair ID")
        pairs.add(pair["id"])
        a, b = images[pair["before"]], images[pair["after"]]
        if (
            a["scene"] != b["scene"]
            or a["size"] != b["size"]
            or a["scope_class_ids"] != b["scope_class_ids"]
        ):
            raise ValueError("incompatible pair scene/dimensions/scope")
    if not images or not pairs:
        raise ValueError("empty dataset")
    return images
