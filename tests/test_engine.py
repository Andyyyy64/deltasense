import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from ultralytics.engine.results import Results

from deltasense import (
    DeltaSense,
    InferenceError,
    InputError,
    ObservationError,
    UnsupportedTaskError,
    engine,
)
from deltasense.observations import ImageIdentity, from_ultralytics


class FakeModel:
    task = "detect"
    names = {0: "person", 1: "item"}

    def __init__(self):
        self.calls = []
        self.failure = None
        self.response = None

    def predict(self, source, stream, **kwargs):
        assert stream is False
        self.calls.append((source, kwargs))
        if self.failure:
            raise self.failure
        self.predictor = SimpleNamespace(
            args=SimpleNamespace(**kwargs), imgsz=[640, 640], device="cpu"
        )
        if self.response is not None:
            return self.response
        return [
            Results(
                img,
                path="private/source.png",
                names=self.names,
                boxes=np.array([[1, 1, 8, 8, 0.9, 0]], np.float32),
            )
            for img in source
        ]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = FakeModel()
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"deterministic test checkpoint placeholder")
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    Image.new("RGB", (10, 10), (240, 20, 10)).save(before)
    Image.new("RGB", (10, 10), (20, 240, 10)).save(after)
    monkeypatch.setattr(engine, "_load_model", lambda path: model)
    return weights, before, after, model


def test_A17_public_api_local_pipeline_snapshot_provenance(setup, tmp_path):
    weights, before, after, model = setup
    original = [p.read_bytes() for p in (weights, before, after)]
    files_before = set(tmp_path.iterdir())
    r = DeltaSense(weights).compare(str(before), after)
    doc = json.loads(r.to_json())
    assert doc["matches"][0]["bbox_area"]["before"] == 49
    assert str(tmp_path) not in r.to_json() and "private/source" not in r.to_json()
    assert doc["provenance"]["deltasense_version"] == "0.1.0"
    assert len(doc["provenance"]["checkpoint_sha256"]) == 64
    assert doc["provenance"]["inference_settings"]["imgsz"] == [640, 640]
    source, settings = model.calls[0]
    assert source[0][0, 0].tolist() == [10, 20, 240]  # BGR to upstream
    assert source[1][0, 0].tolist() == [10, 240, 20]
    assert not any(
        settings[key] for key in ["show", "save", "save_txt", "save_crop", "save_frames"]
    )
    r.plot()
    assert files_before == set(tmp_path.iterdir())
    assert original == [p.read_bytes() for p in (weights, before, after)]


def test_explicit_caller_settings_are_shared_recorded_and_copied(setup):
    weights, before, after, model = setup
    classes = [1, 0, 1]
    sense = DeltaSense(weights, conf=0.6, iou=0.4, classes=classes, match_iou=0.8)
    classes.append(999)
    r = sense.compare(before, after)
    args = model.calls[0][1]
    assert args["conf"] == 0.6 and args["iou"] == 0.4 and args["classes"] == [0, 1]
    assert r.to_dict()["provenance"]["comparison_settings"]["match_iou"] == 0.8


def test_unknown_class_filter_is_an_input_error(setup):
    weights, _, _, _ = setup
    with pytest.raises(InputError, match="absent"):
        DeltaSense(weights, classes=[999])


@pytest.mark.parametrize(
    "model_value", [None, 0, [], "https://host/weights.pt", "missing.pt", "model.onnx"]
)
def test_A13_missing_or_unsupported_checkpoints_fail_before_model_load(model_value, monkeypatch):
    monkeypatch.setattr(engine, "_load_model", lambda path: pytest.fail("unexpected model load"))
    with pytest.raises(InputError):
        DeltaSense(model_value)


@pytest.mark.parametrize(
    "setting,value",
    [
        *[
            (name, value)
            for name in ["conf", "iou"]
            for value in [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None]
        ],
        *[
            (name, value)
            for name in ["imgsz", "max_det"]
            for value in [0, -1, 1.5, True, "640", None]
        ],
        *[("classes", value) for value in [[], [-1], [True], [1.5], "0", ["0"]]],
        *[("device", value) for value in [None, 0, "", " "]],
    ],
)
def test_invalid_settings_before_model_load(setup, setting, value, monkeypatch):
    weights, _, _, _ = setup
    monkeypatch.setattr(engine, "_load_model", lambda path: pytest.fail("unexpected load"))
    with pytest.raises(InputError):
        DeltaSense(weights, **{setting: value})


@pytest.mark.parametrize("task", ["classify", "pose", "obb", "depth", None])
def test_A13_unsupported_tasks_are_explicit(setup, task):
    weights, _, _, model = setup
    model.task = task
    with pytest.raises(UnsupportedTaskError):
        DeltaSense(weights)


def test_loading_failure_is_not_missing_model_or_empty_success(tmp_path):
    weights = tmp_path / "corrupt.pt"
    weights.write_bytes(b"not a pytorch model")
    with pytest.raises(InferenceError, match="load"):
        engine._load_model(weights)


def test_A12_image_dimension_error_occurs_before_inference(setup):
    weights, before, after, model = setup
    Image.new("RGB", (11, 10)).save(after)
    with pytest.raises(InputError, match="equal decoded"):
        DeltaSense(weights).compare(before, after)
    assert not model.calls


def test_A13_image_error_occurs_before_inference(setup):
    weights, before, after, model = setup
    before.unlink()
    with pytest.raises(InputError):
        DeltaSense(weights).compare(before, after)
    assert not model.calls


@pytest.mark.parametrize(
    "failure", [RuntimeError("model failure"), MemoryError("OOM"), ValueError("bad tensor")]
)
def test_inference_failures_never_become_empty_success(setup, failure):
    weights, before, after, model = setup
    model.failure = failure
    with pytest.raises(InferenceError, match="no comparison"):
        DeltaSense(weights).compare(before, after)


@pytest.mark.parametrize("response", [[], [None], [None, None, None]])
def test_wrong_result_count_is_not_success(setup, response):
    weights, before, after, model = setup
    model.response = response
    with pytest.raises(InferenceError, match="exactly one"):
        DeltaSense(weights).compare(before, after)


def test_changed_or_removed_checkpoint_is_not_silently_reused(setup):
    weights, before, after, model = setup
    sense = DeltaSense(weights)
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(InputError, match="changed"):
        sense.compare(before, after)
    weights.unlink()
    with pytest.raises(InputError, match="not readable"):
        sense.compare(before, after)
    assert not model.calls


def test_checkpoint_changed_during_loading_rejected(setup, monkeypatch):
    weights, _, _, model = setup

    def load(path):
        path.write_bytes(b"modified while loading")
        return model

    monkeypatch.setattr(engine, "_load_model", load)
    with pytest.raises(InputError, match="while loading"):
        DeltaSense(weights)


def test_repeated_and_concurrent_comparisons_remain_consistent(setup):
    weights, before, after, _ = setup
    sense = DeltaSense(weights)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: sense.compare(before, after).to_json(), range(12)))
    assert len(set(results)) == 1


@pytest.mark.parametrize("conflict", ["offline", "autoinstall", "preimport"])
def test_conflicting_upstream_network_settings_fail_explicitly(tmp_path, monkeypatch, conflict):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not reached")
    if conflict == "offline":
        monkeypatch.setenv("YOLO_OFFLINE", "false")
    elif conflict == "autoinstall":
        monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    else:
        import ultralytics.utils

        monkeypatch.setattr(ultralytics.utils, "ONLINE", True)
    with pytest.raises(InputError, match="network|requires"):
        engine._load_model(weights)


def upstream(boxes=None, masks=None, names=None, shape=(10, 10)):
    return Results(
        np.zeros((*shape, 3), np.uint8), "private", names or {0: "person"}, boxes=boxes, masks=masks
    )


def test_public_results_adapter_empty_and_full_masks():
    identity = ImageIdentity("a" * 64, 10, 10)
    empty = from_ultralytics(upstream(np.zeros((0, 6))), identity, "detect")
    assert not empty.detections
    masks = np.ones((1, 10, 10), np.uint8)
    data = from_ultralytics(
        upstream(np.array([[0, 0, 10, 10, 0.8, 0]]), masks), identity, "segment"
    )
    assert data.detections[0].mask.sum() == 100


@pytest.mark.parametrize(
    "kind",
    [
        "no_boxes",
        "wrong_shape",
        "mask_count",
        "mask_shape",
        "detect_masks",
        "unknown_class",
        "fractional_class",
        "negative_class",
        "nan_class",
        "nan_box",
        "outside_box",
        "nan_score",
        "nonbinary_mask",
        "no_result",
    ],
)
def test_malformed_upstream_results_are_explicit_errors(kind):
    identity = ImageIdentity("a" * 64, 10, 10)
    boxes = np.array([[0, 0, 10, 10, 0.8, 0]], np.float64)
    masks, task, shape = None, "segment", (10, 10)
    if kind == "no_boxes":
        boxes = None
    elif kind == "wrong_shape":
        shape = (11, 10)
    elif kind == "mask_count":
        masks = np.ones((2, 10, 10))
    elif kind == "mask_shape":
        masks = np.ones((1, 5, 5))
    elif kind == "detect_masks":
        task, masks = "detect", np.ones((1, 10, 10))
    elif kind in {"unknown_class", "fractional_class", "negative_class", "nan_class"}:
        boxes[0, 5] = {
            "unknown_class": 999,
            "fractional_class": 0.5,
            "negative_class": -1,
            "nan_class": float("nan"),
        }[kind]
    elif kind == "nan_box":
        boxes[0, 2] = float("nan")
    elif kind == "outside_box":
        boxes[0, 2] = 11
    elif kind == "nan_score":
        boxes[0, 4] = float("nan")
    elif kind == "nonbinary_mask":
        masks = np.ones((1, 10, 10)) * 0.5
    result = None if kind == "no_result" else upstream(boxes, masks, shape=shape)
    with pytest.raises(ObservationError):
        from_ultralytics(result, identity, task)
