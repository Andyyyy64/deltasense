"""Independent scoring must not hide missed objects, duplicates, or wrong links."""

import copy
import json
import runpy
from pathlib import Path

import pytest

METRICS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/accuracy_metrics.py"))
assign = METRICS["assign_to_truth"]
score = METRICS["score_pair"]
aggregate = METRICS["aggregate"]
validate = METRICS["validate_manifest"]


def truth(identity="person1", box=(0, 0, 10, 10), cls=0):
    return {"object_id": identity, "bbox_xyxy": list(box), "class_id": cls}


def detection(local_id=0, box=(0, 0, 10, 10), cls=0):
    return {"id": local_id, "bbox_xyxy": list(box), "class_id": cls}


def frame(*objects):
    return {"objects": list(objects), "scope_class_ids": [0]}


def document(before=(), after=(), links=()):
    return {
        "before": {"detections": list(before)},
        "after": {"detections": list(after)},
        "matches": [{"before_id": a, "after_id": b, "class_id": 0} for a, b in links],
    }


def test_duplicate_prediction_is_false_positive_and_missing_truth_is_false_negative():
    mapping, metrics = assign(
        [detection(0), detection(1)], [truth(), truth("person2", (20, 20, 30, 30))], [0], 0.5
    )
    assert len(mapping) == 1
    assert metrics == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5}


def test_maximum_cardinality_assignment_recovers_pair_missed_by_greedy():
    # Prediction 0 overlaps both truth boxes; prediction 1 overlaps only truth 0.
    mapping, metrics = assign(
        [detection(0, (0, 0, 15, 10)), detection(1)],
        [truth("a"), truth("b", (5, 0, 15, 10))],
        [0],
        0.5,
    )
    assert mapping == {1: "a", 0: "b"} and metrics["tp"] == 2


def test_localization_boundary_and_cross_class_assignment():
    _, metrics = assign([detection(0, (0, 0, 20, 10))], [truth()], [0], 0.5)
    assert metrics["tp"] == 1
    _, metrics = assign([detection(0, (0, 0, 20, 10))], [truth()], [0], 0.50001)
    assert metrics["tp"] == 0 and metrics["fp"] == metrics["fn"] == 1
    mapping, metrics = assign([detection(cls=1)], [truth()], [0, 1], 0.5)
    assert mapping == {} and metrics["fp"] == metrics["fn"] == 1


def test_out_of_scope_is_excluded_and_preserved():
    row = score(document([detection(cls=15)], [detection(cls=15)]), frame(), frame(), 0.5)
    assert row["detection_before"]["fp"] == 0
    assert len(row["out_of_scope_predictions"]["before"]) == 1
    assert row["count_change_proxy"]["predicted_signal"] is False


def test_missed_both_objects_does_not_gain_detection_or_link_credit_for_zero_count_delta():
    row = score(document(), frame(truth()), frame(truth()), 0.5)
    assert row["signed_count_vector_correct"] is True
    assert row["detection_before"]["fn"] == row["detection_after"]["fn"] == 1
    assert row["correspondence"]["fn"] == 1
    assert row["correspondence"]["conditional_recall"] is None


def test_wrong_identity_link_is_false_positive_and_misses_both_expected_links():
    a = frame(truth("a"), truth("b", (20, 0, 30, 10)))
    predictions = [detection(0), detection(1, (20, 0, 30, 10))]
    row = score(document(predictions, predictions, [(0, 1), (1, 0)]), a, a, 0.5)
    assert row["correspondence"]["tp"] == 0
    assert row["correspondence"]["fp"] == row["correspondence"]["fn"] == 2
    assert row["correspondence"]["conditional_recall"] == 0


def test_unresolved_links_lose_recall_and_never_gain_false_precision():
    row = score(document([detection()], [detection()]), frame(truth()), frame(truth()), 0.5)
    assert row["correspondence"]["fp"] == 0 and row["correspondence"]["fn"] == 1
    assert row["correspondence"]["precision"] is None
    assert row["correspondence"]["conditional_recall"] == 0


def test_zero_opportunities_stay_null():
    result = aggregate([], {})
    assert result["count_change_proxy"]["false_alarm_rate"] is None
    assert result["count_change_proxy"]["miss_rate"] is None
    assert result["correspondence"]["precision"] is None
    assert result["signed_count_vector"]["delta_mae"] is None


def test_proxy_confusion_denominators_and_wrong_direction():
    positive = score(document([detection()], []), frame(truth()), frame(), 0.5)
    miss = score(document(), frame(truth()), frame(), 0.5)
    negative = score(document(), frame(), frame(), 0.5)
    alarm = score(document([], [detection()]), frame(), frame(), 0.5)
    opposite = score(document([], [detection()]), frame(truth()), frame(), 0.5)
    assert opposite["signed_count_vector_correct"] is False
    result = aggregate([positive, miss, negative, alarm], {})["count_change_proxy"]
    assert result == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "false_alarm_rate": 0.5,
        "miss_rate": 0.5,
    }


def test_unique_frame_detection_is_not_multiplied_by_reused_pairs():
    row = score(
        document([detection()], [detection()], [(0, 0)]), frame(truth()), frame(truth()), 0.5
    )
    result = aggregate([row, row], {"frame1": row["detection_before"]})
    assert result["detection_on_first_observation_of_each_unique_frame"]["tp"] == 1
    assert result["correspondence"]["tp"] == 2  # Pair opportunities, explicitly not unique images.


def manifest():
    return {
        "images": [{"id": "a", "scene": "s", "size": [100, 100], **frame(truth())}],
        "pairs": [{"id": "p", "before": "a", "after": "a"}],
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["images"].append(copy.deepcopy(d["images"][0])),
        lambda d: d["images"][0]["objects"].append(truth()),
        lambda d: d["images"][0]["objects"][0].update(bbox_xyxy=[0, 0, 101, 10]),
        lambda d: d["images"][0]["objects"][0].update(bbox_xyxy=[0, 0, 0, 10]),
        lambda d: d["images"][0]["objects"][0].update(class_id=15),
        lambda d: d["images"][0].update(scope_class_ids=[]),
        lambda d: d["images"][0].update(size=[True, 100]),
        lambda d: d["pairs"].append(copy.deepcopy(d["pairs"][0])),
        lambda d: d.update(pairs=[]),
    ],
)
def test_invalid_annotations_are_rejected(change):
    data = manifest()
    change(data)
    with pytest.raises(ValueError):
        validate(data)


def test_pilot_manifest_is_valid_and_explicitly_provisional():
    path = Path(__file__).resolve().parents[1] / "evaluation/pilot-v1.json"
    data = json.loads(path.read_text())
    assert len(validate(data)) == 13 and len(data["pairs"]) == 14
    assert data["independent_human_review"] is False
