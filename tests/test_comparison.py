import itertools
import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from deltasense.comparison import ComparisonSettings, bbox_iou, compare_observations
from deltasense.errors import InputError, ObservationError
from deltasense.observations import Detection, ImageIdentity, Observation
from deltasense.result import ComparisonResult


def detection(box=(10, 10, 30, 30), *, cls=0, mask=None, name=None, score=0.9):
    return Detection(cls, name or f"class{cls}", score, box, mask)


def observation(*items, task="detect", width=100, height=100):
    return Observation(ImageIdentity("a" * 64, width, height), task, items)


def compare(a, b, threshold=0.2):
    return compare_observations(a, b, ComparisonSettings(threshold))


def result(a, b):
    rgb = np.zeros((a.identity.height, a.identity.width, 3), np.uint8)
    return ComparisonResult(a, b, compare(a, b), {}, (rgb, rgb))


def test_A01_identical_predictions():
    a = observation(detection())
    c = compare(a, a)
    assert c["prediction_counts"] == [
        {"class_id": 0, "class_name": "class0", "before": 1, "after": 1, "delta": 0}
    ]
    assert c["matches"][0]["displacement"] == {"dx": 0, "dy": 0, "unit": "pixel"}
    assert c["matches"][0]["bbox_area"]["delta"] == 0
    assert c["unmatched_before"] == c["unmatched_after"] == []


def test_A02_counts_and_unmatched_evidence():
    a = observation(detection(), detection((50, 50, 60, 60)))
    b = observation(*a.detections, detection((80, 80, 90, 90)))
    c = compare(a, b)
    assert c["prediction_counts"][0]["delta"] == 1
    assert c["unmatched_after"] == [
        {
            "id": 2,
            "reason": "no_candidate",
            "displacement": None,
            "bbox_area_delta": None,
            "mask_area_delta": None,
        }
    ]
    assert "appeared" not in json.dumps(c)


def test_A03_missed_detection_is_not_disappearance():
    r = result(observation(detection()), observation())
    assert r.unmatched_before[0]["reason"] == "no_candidate"
    assert r.prediction_counts[0]["after"] == 0
    assert r.to_dict()["physical_change_verified"] is False


def test_A04_exact_signed_displacement():
    a = observation(detection((0, 0, 40, 60)))
    b = observation(detection((10, 20, 50, 80)))
    assert compare(a, b)["matches"][0]["displacement"] == {"dx": 10, "dy": 20, "unit": "pixel"}


def test_A05_bbox_area_arithmetic():
    a = observation(detection((0, 0, 10, 10)))
    b = observation(detection((0, 0, 12, 12)))
    area = compare(a, b)["matches"][0]["bbox_area"]
    assert (area["before"], area["after"], area["delta"], area["relative_delta"]) == (
        100,
        144,
        44,
        0.44,
    )


@pytest.mark.parametrize("counts", [(1, 2), (2, 1), (2, 2), (3, 3)])
def test_A06_competing_identical_candidates_remain_ambiguous(counts):
    a = observation(*(detection() for _ in range(counts[0])))
    b = observation(*(detection() for _ in range(counts[1])))
    c = compare(a, b)
    assert c["matches"] == []
    assert len(c["ambiguous_candidates"]) == counts[0] * counts[1]
    assert all(d["reason"] == "ambiguous_correspondence" for d in c["unmatched_before"])
    assert all(d["displacement"] is None for d in c["unmatched_after"])


def test_competing_but_unequal_iou_still_not_forced():
    a = observation(detection())
    b = observation(detection(), detection((12, 12, 32, 32)))
    assert compare(a, b)["matches"] == []


def test_no_implicit_resolution_of_competing_graph_chain():
    a = observation(detection((0, 0, 20, 20)), detection((10, 0, 30, 20)))
    b = observation(detection((5, 0, 25, 20)), detection((20, 0, 40, 20)))
    c = compare(a, b)
    assert len(c["ambiguous_candidates"]) == 3
    assert not c["matches"]


def test_A07_all_permutations_preserve_unique_associations():
    items = [detection((i * 30, 0, i * 30 + 10, 10)) for i in range(3)]
    for before in itertools.permutations(items):
        for after in itertools.permutations(items):
            c = compare(observation(*before), observation(*after))
            assert len(c["matches"]) == 3
            for m in c["matches"]:
                assert before[m["before_id"]].bbox == after[m["after_id"]].bbox


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_A08_empty_outputs_are_successful_observations(task):
    r = result(observation(task=task), observation(task=task))
    assert r.prediction_counts == r.matches == []
    assert r.to_dict()["physical_change_verified"] is False
    assert r.unchecked_conditions == ["alignment", "illumination", "occlusion", "physical_identity"]
    json.loads(r.to_json())


def mask_case(a, b):
    before = observation(detection(mask=a), task="segment")
    after = observation(detection(mask=b), task="segment")
    return result(before, after)


def test_A09_mask_growth_exact():
    a, b = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    a[:10, :10] = True
    b[:10, :12] = True
    area = mask_case(a, b).matches[0]["mask_area"]
    assert (area["before"], area["after"], area["delta"], area["relative_delta"]) == (
        100,
        120,
        20,
        0.2,
    )
    assert area["spatial_difference"] == {
        "before_only_pixels": 0,
        "after_only_pixels": 20,
        "intersection_pixels": 100,
        "xor_pixels": 20,
    }


def test_A10_equal_area_translated_mask_is_not_growth():
    a, b = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    a[:10, :10] = True
    b[:10, 2:12] = True
    area = mask_case(a, b).matches[0]["mask_area"]
    assert area["delta"] == 0
    assert area["spatial_difference"]["xor_pixels"] == 40


@pytest.mark.parametrize("missing", ["before", "after", "both"])
def test_A11_missing_masks_are_not_substituted_by_boxes(missing):
    mask = np.ones((100, 100), bool)
    a = None if missing in {"before", "both"} else mask
    b = None if missing in {"after", "both"} else mask
    area = mask_case(a, b).matches[0]["mask_area"]
    assert area["delta"] is area["relative_delta"] is None
    assert area["unavailable_reason"] == f"{missing}_mask{'s' if missing == 'both' else ''}_missing"


@pytest.mark.parametrize("after_area", [0, 100])
def test_A11_zero_baseline_mask_is_valid_but_ratio_unavailable(after_area):
    a, b = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    b.flat[:after_area] = True
    area = mask_case(a, b).matches[0]["mask_area"]
    assert area["before"] == 0 and area["delta"] == after_area
    assert area["relative_delta"] is None
    assert area["relative_unavailable_reason"] == "zero_before_area"
    assert area["unavailable_reason"] is None


@pytest.mark.parametrize("dimensions", [(101, 100), (100, 101)])
def test_A12_dimension_mismatch(dimensions):
    with pytest.raises(InputError, match="equal decoded"):
        compare(observation(), observation(width=dimensions[0], height=dimensions[1]))


def test_A14_capture_conditions_never_become_verified():
    r = result(observation(detection()), observation(detection()))
    assert r.to_dict()["unchecked_conditions"] == r.unchecked_conditions
    assert r.to_dict()["physical_change_verified"] is False


def test_A15_reverse_recomputes_relative_delta():
    a = observation(detection((0, 0, 10, 10)))
    b = observation(detection((0, 0, 12, 12)))
    forward = compare(a, b)["matches"][0]
    reverse = compare(b, a)["matches"][0]
    assert reverse["bbox_area"]["delta"] == -44
    assert reverse["bbox_area"]["relative_delta"] == pytest.approx(-44 / 144)
    assert reverse["displacement"]["dx"] == -forward["displacement"]["dx"]


@pytest.mark.parametrize(
    "task,kind", list(itertools.product(["detect", "segment"], ["empty", "partial", "ambiguous"]))
)
def test_A16_json_and_headless_plot_preserve_status(task, kind, monkeypatch, tmp_path):
    import PIL.Image

    monkeypatch.setattr(PIL.Image.Image, "show", lambda *args: pytest.fail("GUI opened"))
    if kind == "empty":
        a, b = observation(task=task), observation(task=task)
    elif kind == "partial":
        a, b = observation(detection(), task=task), observation(task=task)
    else:
        a = observation(detection(), task=task)
        b = observation(detection(), detection(), task=task)
    r = result(a, b)
    assert json.loads(r.to_json()) == r.to_dict()
    image = r.plot()
    assert image.mode == "RGB" and image.width >= 920
    assert list(tmp_path.iterdir()) == []
    assert "NaN" not in r.to_json() and "Infinity" not in r.to_json()


@pytest.mark.parametrize("threshold,expected", [(0.499999, 1), (0.5, 1), (0.500001, 0)])
def test_iou_threshold_inclusive_boundary(threshold, expected):
    a = observation(detection((0, 0, 10, 10)))
    b = observation(detection((0, 0, 20, 10)))
    assert len(compare(a, b, threshold)["matches"]) == expected


@pytest.mark.parametrize(
    "threshold", [0, -0.1, 1.001, float("nan"), float("inf"), True, "0.2", None]
)
def test_invalid_matching_thresholds(threshold):
    with pytest.raises(InputError):
        ComparisonSettings(threshold)


def test_classes_never_cross_match_and_count_union_is_sorted():
    a = observation(detection(cls=7))
    b = observation(detection(cls=2))
    c = compare(a, b)
    assert not c["matches"]
    assert [(x["class_id"], x["delta"]) for x in c["prediction_counts"]] == [(2, 1), (7, -1)]


def test_inconsistent_class_names_and_tasks_rejected():
    with pytest.raises(ObservationError, match="class names"):
        compare(observation(detection(name="person")), observation(detection(name="car")))
    with pytest.raises(ObservationError, match="tasks differ"):
        compare(observation(), observation(task="segment"))


def test_zero_area_boxes_do_not_acquire_identity():
    a = observation(detection((10, 10, 10, 20)))
    assert not compare(a, a)["matches"]


def test_snapshot_not_affected_by_caller_mutation():
    mask = np.ones((100, 100), bool)
    r = mask_case(mask, mask)
    mask[:] = False
    first = r.to_dict()
    first["matches"].clear()
    assert r.matches[0]["mask_area"]["before"] == 10000
    assert not r.before.detections[0].mask.flags.writeable


@given(st.lists(st.integers(0, 8), max_size=30), st.lists(st.integers(0, 8), max_size=30))
def test_property_count_union_and_reversal(before, after):
    a = observation(*(detection(cls=c) for c in before))
    b = observation(*(detection(cls=c) for c in after))
    fwd, rev = compare(a, b), compare(b, a)
    assert sum(x["before"] for x in fwd["prediction_counts"]) == len(before)
    assert sum(x["after"] for x in fwd["prediction_counts"]) == len(after)
    for c, r in zip(fwd["prediction_counts"], rev["prediction_counts"], strict=True):
        assert c["class_id"] == r["class_id"] and c["delta"] == -r["delta"]
    # A detection can never be reused or lost between accepted/unmatched sets.
    assert len(fwd["matches"]) + len(fwd["unmatched_before"]) == len(before)
    assert len(fwd["matches"]) + len(fwd["unmatched_after"]) == len(after)


@given(st.integers(1, 40), st.integers(1, 40), st.integers(-5, 5), st.integers(-5, 5))
def test_property_displacement_matches_independent_coordinates(width, height, dx, dy):
    box_a = (20, 20, 20 + width, 20 + height)
    box_b = (20 + dx, 20 + dy, 20 + dx + width, 20 + dy + height)
    c = compare(observation(detection(box_a)), observation(detection(box_b)))
    if c["matches"]:
        m = c["matches"][0]
        assert m["displacement"]["dx"] == dx and m["displacement"]["dy"] == dy
        assert m["bbox_area"]["delta"] == 0


@given(
    st.lists(st.booleans(), min_size=100, max_size=100),
    st.lists(st.booleans(), min_size=100, max_size=100),
)
def test_property_mask_support_conservation(before, after):
    a, b = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    a[:10, :10] = np.array(before).reshape(10, 10)
    b[:10, :10] = np.array(after).reshape(10, 10)
    m = mask_case(a, b).matches[0]["mask_area"]
    diff = m["spatial_difference"]
    assert m["before"] == sum(before) and m["after"] == sum(after)
    assert m["delta"] == sum(after) - sum(before)
    assert diff["before_only_pixels"] + diff["intersection_pixels"] == sum(before)
    assert diff["after_only_pixels"] + diff["intersection_pixels"] == sum(after)
    assert diff["xor_pixels"] == sum(x != y for x, y in zip(before, after, strict=True))


@given(st.integers(0, 80), st.integers(0, 80), st.integers(0, 20), st.integers(0, 20))
def test_property_iou_is_symmetric_bounded_and_degenerate_safe(x, y, width, height):
    box = (x, y, x + width, y + height)
    base = (10, 10, 30, 30)
    assert 0 <= bbox_iou(box, base) <= 1
    assert bbox_iou(box, base) == bbox_iou(base, box)
    assert bbox_iou(box, box) == (1 if width and height else 0)
