import itertools
import json
import runpy
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
METRICS = runpy.run_path(str(ROOT / "scripts/accuracy_metrics.py"))
maximum_assignment = METRICS["maximum_assignment"]
assign = METRICS["assign_to_truth"]
score_observation = METRICS["score_observation"]
score_pair = METRICS["score_pair"]


def box(identity, coords=(0, 0, 10, 10)):
    return {"object_id": identity, "class_id": 0, "bbox_xyxy": list(coords)}


def prediction(local_id, coords=(0, 0, 10, 10)):
    return {"id": local_id, "class_id": 0, "bbox_xyxy": list(coords)}


def test_assignment_maximizes_total_iou_after_cardinality():
    mapping, metrics = assign(
        [prediction(0), prediction(1, (1, 0, 11, 10))],
        [box("a"), box("b", (4, 0, 14, 10))],
        [0],
        0.3,
    )
    assert mapping == {0: "a", 1: "b"} and metrics["tp"] == 2


@given(
    n=st.integers(0, 4),
    m=st.integers(0, 4),
    entries=st.lists(st.integers(0, 10), min_size=16, max_size=16),
)
def test_assignment_matches_exhaustive_small_matrix_oracle(n, m, entries):
    scores = [[entries[i * 4 + j] / 10 for j in range(m)] for i in range(n)]
    pairs = maximum_assignment(scores)
    assert len(set(pairs.values())) == len(pairs)
    assert all(scores[i][j] > 0 for i, j in pairs.items())
    objective = (len(pairs), sum(scores[i][j] for i, j in pairs.items()))
    possibilities = []
    for columns in itertools.permutations(range(m + n), n):
        vals = [scores[i][j] for i, j in enumerate(columns) if j < m and scores[i][j] > 0]
        possibilities.append((len(vals), sum(vals)))
    expected = max(possibilities, default=(0, 0))
    assert objective[0] == expected[0]
    assert objective[1] == pytest.approx(expected[1])


def test_valid_target_has_priority_over_overlapping_ignore_annotation():
    frame = {"objects": [box("target")], "ignore_objects": [box("ignored")], "scope_class_ids": [0]}
    assigned, metrics, ignored = score_observation([prediction(0)], frame, 0.5)
    assert assigned == {0: "target"} and ignored == set()
    assert metrics["tp"] == 1 and metrics["fp"] == 0


def test_one_ignored_object_does_not_hide_duplicate_predictions():
    frame = {"objects": [], "ignore_objects": [box("ignored")], "scope_class_ids": [0]}
    assigned, metrics, ignored = score_observation([prediction(0), prediction(1)], frame, 0.5)
    assert assigned == {} and len(ignored) == 1
    assert metrics["ignored_predictions"] == 1 and metrics["fp"] == 1


def test_ignore_threshold_is_fixed_and_does_not_hide_no_overlap():
    frame = {"objects": [], "ignore_objects": [box("ignored")], "scope_class_ids": [0]}
    _, metrics, ignored = score_observation([prediction(0, (50, 50, 60, 60))], frame, 0.3)
    assert metrics["fp"] == 1 and ignored == set()


def test_ignored_links_are_saved_and_raw_count_is_never_silently_filtered():
    a = {"objects": [], "ignore_objects": [box("ignored")], "scope_class_ids": [0]}
    b = {"objects": [], "ignore_objects": [], "scope_class_ids": [0]}
    doc = {"before": {"detections": [prediction(0)]}, "after": {"detections": []}, "matches": []}
    row = score_pair(doc, a, b, 0.5)
    assert row["detection_before"]["fp"] == 0
    assert row["counts"][0]["predicted_delta"] == -1
    assert row["count_change_proxy"]["predicted_signal"] is True
    assert row["ignored_prediction_ids"]["before"] == [0]
    doc["after"]["detections"] = [prediction(0)]
    doc["matches"] = [{"class_id": 0, "before_id": 0, "after_id": 0}]
    row = score_pair(doc, a, a, 0.5)
    assert row["correspondence"]["ignored_links"] == 1
    assert row["correspondence"]["tp"] == row["correspondence"]["fp"] == 0


def test_visibility_buckets_use_truth_including_missed_people():
    frame = {
        "scope_class_ids": [0],
        "objects": [
            {**box("a"), "visibility": 0.2},
            {**box("b", (20, 0, 30, 10)), "visibility": 0.5},
            {**box("c", (40, 0, 50, 10)), "visibility": 0.9},
        ],
    }
    _, metrics, _ = score_observation([prediction(0)], frame, 0.5)
    assert metrics["by_visibility"] == {
        "low": {"tp": 1, "fn": 0},
        "medium": {"tp": 0, "fn": 1},
        "high": {"tp": 0, "fn": 1},
    }


def test_displacement_error_is_conditional_on_ground_truth_identity():
    a = {"scope_class_ids": [0], "objects": [box("a")]}
    b = {"scope_class_ids": [0], "objects": [box("a", (2, 3, 12, 13))]}
    doc = {
        "before": {"detections": [prediction(0)]},
        "after": {"detections": [prediction(0, (2, 3, 12, 13))]},
        "matches": [
            {"class_id": 0, "before_id": 0, "after_id": 0, "displacement": {"dx": 5, "dy": 7}}
        ],
    }
    row = score_pair(doc, a, b, 0.5)
    assert row["conditional_geometry"][0]["displacement_error_pixels"] == 5
    b["objects"][0]["object_id"] = "different"
    assert score_pair(doc, a, b, 0.5)["conditional_geometry"] == []


def test_mot_plan_selection_covers_all_scenes_and_three_gaps():
    namespace = runpy.run_path(str(ROOT / "scripts/prepare_mot17.py"))
    plan = json.loads((ROOT / "evaluation/mot17-plan-v1.json").read_text())
    names = namespace["select_entries"](plan)
    assert len(names) == len(set(names)) == 102
    assert sum(n.endswith(".jpg") for n in names) == 96
    assert plan["gaps"] == [1, 15, 60]


def test_mot_coordinate_conversion_is_one_based_and_clipped(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    namespace = runpy.run_path(str(ROOT / "scripts/build_mot17_manifest.py"))
    convert = namespace["convert_box"]
    assert convert([1, 1, 1, 1, 10, 20], 100, 100) == [0, 0, 10, 20]
    assert convert([1, 1, -2, -3, 10, 20], 100, 100) == [0, 0, 7, 16]
    assert convert([1, 1, 200, 200, 10, 20], 100, 100) is None


def test_duplicate_id_across_target_and_ignore_is_rejected():
    frame = {
        "id": "a",
        "scene": "scene",
        "size": [100, 100],
        "scope_class_ids": [0],
        "objects": [box("a")],
        "ignore_objects": [box("a")],
    }
    data = {"images": [frame], "pairs": [{"id": "p", "before": "a", "after": "a"}]}
    with pytest.raises(ValueError):
        METRICS["validate_manifest"](data)
