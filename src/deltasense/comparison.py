"""Deterministic comparison arithmetic, independent of model detection quality."""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .errors import InputError, ObservationError
from .observations import Observation


@dataclass(frozen=True)
class ComparisonSettings:
    match_iou: float = 0.2

    def __post_init__(self):
        if isinstance(self.match_iou, bool) or not isinstance(self.match_iou, (float, int)):
            raise InputError("match_iou must be a number in (0, 1]")
        if not 0 < self.match_iou <= 1:
            raise InputError("match_iou must be finite and in (0, 1]")


def bbox_iou(a, b):
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def area_measurement(before, after):
    relative = (after - before) / before if before else None
    reason = None if before else "zero_before_area"
    if relative is not None and not math.isfinite(relative):
        relative, reason = None, "relative_delta_out_of_range"
    return {
        "before": before,
        "after": after,
        "delta": after - before,
        "relative_delta": relative,
        "relative_unavailable_reason": reason,
        "unit": "pixel_squared",
        "unavailable_reason": None,
    }


def mask_measurement(a, b):
    if a is None or b is None:
        missing = (
            "both_masks_missing"
            if a is None and b is None
            else ("before_mask_missing" if a is None else "after_mask_missing")
        )
        return {
            "before": None if a is None else int(a.sum()),
            "after": None if b is None else int(b.sum()),
            "delta": None,
            "relative_delta": None,
            "relative_unavailable_reason": missing,
            "unit": "pixel_squared",
            "unavailable_reason": missing,
            "spatial_difference": None,
        }
    measurement = area_measurement(int(a.sum()), int(b.sum()))
    measurement["spatial_difference"] = {
        "before_only_pixels": int(np.count_nonzero(a & ~b)),
        "after_only_pixels": int(np.count_nonzero(b & ~a)),
        "intersection_pixels": int(np.count_nonzero(a & b)),
        "xor_pixels": int(np.count_nonzero(a ^ b)),
    }
    return measurement


def compare_observations(before: Observation, after: Observation, settings: ComparisonSettings):
    if (before.identity.width, before.identity.height) != (
        after.identity.width,
        after.identity.height,
    ):
        raise InputError("images must have equal decoded dimensions; align them outside DeltaSense")
    if before.task != after.task:
        raise ObservationError("before/after tasks differ")
    names = {}
    for d in (*before.detections, *after.detections):
        if names.setdefault(d.class_id, d.class_name) != d.class_name:
            raise ObservationError("before/after class names differ for the same ID")
    counts_a = Counter(d.class_id for d in before.detections)
    counts_b = Counter(d.class_id for d in after.detections)
    counts = [
        {
            "class_id": cls,
            "class_name": names[cls],
            "before": counts_a[cls],
            "after": counts_b[cls],
            "delta": counts_b[cls] - counts_a[cls],
        }
        for cls in sorted(names)
    ]
    edges = []
    by_before = [[] for _ in before.detections]
    by_after = [[] for _ in after.detections]
    for i, a in enumerate(before.detections):
        for j, b in enumerate(after.detections):
            if a.class_id != b.class_id:
                continue
            iou = bbox_iou(a.bbox, b.bbox)
            if iou >= settings.match_iou:
                edge = {"before_id": i, "after_id": j, "iou": iou}
                edges.append(edge)
                by_before[i].append(edge)
                by_after[j].append(edge)
    matches, ambiguous = [], []
    used_a, used_b = set(), set()
    for edge in edges:
        i, j = edge["before_id"], edge["after_id"]
        if len(by_before[i]) != 1 or len(by_after[j]) != 1:
            ambiguous.append({**edge, "reason": "competing_candidates"})
            continue
        a, b = before.detections[i], after.detections[j]
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        match = {
            **edge,
            "class_id": a.class_id,
            "class_name": a.class_name,
            "displacement": {
                "dx": (bx1 + bx2 - ax1 - ax2) / 2,
                "dy": (by1 + by2 - ay1 - ay2) / 2,
                "unit": "pixel",
            },
            "bbox_area": area_measurement((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)),
        }
        if before.task == "segment":
            match["mask_area"] = mask_measurement(a.mask, b.mask)
        matches.append(match)
        used_a.add(i)
        used_b.add(j)

    def unmatched(adjacency, used):
        return [
            {
                "id": i,
                "reason": "ambiguous_correspondence" if candidates else "no_candidate",
                "displacement": None,
                "bbox_area_delta": None,
                "mask_area_delta": None,
            }
            for i, candidates in enumerate(adjacency)
            if i not in used
        ]

    return {
        "prediction_counts": counts,
        "matches": matches,
        "unmatched_before": unmatched(by_before, used_a),
        "unmatched_after": unmatched(by_after, used_b),
        "ambiguous_candidates": ambiguous,
    }
