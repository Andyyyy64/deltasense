import json
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from test_comparison import compare, detection, mask_case, observation, result
from test_engine import FakeModel

from deltasense import DeltaSense, InferenceError, ObservationError
from deltasense.observations import Detection, ImageIdentity, Observation, from_ultralytics


@pytest.mark.parametrize("name", [None, "", 123])
def test_missing_or_nonstring_names(name):
    with pytest.raises(ObservationError):
        Detection(0, name, 0.8, (0, 0, 10, 10))


@pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-1, 10)])
def test_nonpositive_observation_dimensions(width, height):
    with pytest.raises(ObservationError):
        Observation(ImageIdentity("a" * 64, width, height), "detect", ())


@pytest.mark.parametrize(
    "xyxy,conf,classes",
    [
        (np.zeros((1, 3)), np.zeros(1), np.zeros(1)),
        (np.zeros((1, 4)), np.zeros(2), np.zeros(1)),
        (np.zeros((1, 4)), np.zeros(1), np.zeros(2)),
    ],
)
def test_inconsistent_raw_arrays_are_rejected(xyxy, conf, classes):
    data = SimpleNamespace(xyxy=xyxy, conf=conf, cls=classes)
    boxes = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: data))
    with pytest.raises(ObservationError):
        from_ultralytics(
            SimpleNamespace(orig_shape=(10, 10), boxes=boxes, masks=None),
            ImageIdentity("a" * 64, 10, 10),
            "detect",
        )


def test_mask_visualization_and_result_properties_are_snapshots():
    a, b = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    a[40:50, 40:50] = True
    b[40:50, 42:52] = True
    r = mask_case(a, b)
    original = r._images[0].copy()
    plot = np.asarray(r.plot())
    np.testing.assert_array_equal(r._images[0], original)
    # Source pixels begin at y=38; x is unchanged in the before panel.
    assert (plot[80, 40] == [0, 143, 165]).all()
    assert r.ambiguous_candidates == []
    assert (
        json.loads(r.to_json())["matches"][0]["mask_area"]["spatial_difference"]["xor_pixels"] == 40
    )


def test_unicode_and_long_names_and_edge_touching_boxes_render_without_mutation():
    a = observation(detection((0, 0, 100, 100), name="物体 " + "long name " * 50))
    r = result(a, a)
    assert r.plot().height > 500
    assert (
        json.loads(r.to_json())["before"]["detections"][0]["class_name"]
        == a.detections[0].class_name
    )


def test_malformed_return_type_is_an_inference_failure(tmp_path, monkeypatch):
    from deltasense import engine

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")
    path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(path)
    model = FakeModel()
    model.predict = lambda **kwargs: None
    monkeypatch.setattr(engine, "_load_model", lambda path: model)
    with pytest.raises(InferenceError, match="exactly one"):
        DeltaSense(weights).compare(path, path)


@given(
    st.lists(
        st.tuples(
            st.integers(0, 3),
            st.integers(0, 70),
            st.integers(0, 70),
            st.integers(1, 30),
            st.integers(1, 30),
        ),
        max_size=18,
    ),
    st.lists(
        st.tuples(
            st.integers(0, 3),
            st.integers(0, 70),
            st.integers(0, 70),
            st.integers(1, 30),
            st.integers(1, 30),
        ),
        max_size=18,
    ),
    st.sampled_from([0.01, 0.2, 0.5, 1.0]),
)
def test_property_arbitrary_matching_graph_partition_and_permutation(before, after, threshold):
    def make(items):
        return observation(*(detection((x, y, x + w, y + h), cls=cls) for cls, x, y, w, h in items))

    a, b = make(before), make(after)
    c = compare(a, b, threshold)
    reverse = compare(b, a, threshold)
    assert len(c["matches"]) + len(c["unmatched_before"]) == len(before)
    assert len(c["matches"]) + len(c["unmatched_after"]) == len(after)
    assert len({m["before_id"] for m in c["matches"]}) == len(c["matches"])
    assert len({m["after_id"] for m in c["matches"]}) == len(c["matches"])
    assert {(m["before_id"], m["after_id"]) for m in c["matches"]} == {
        (m["after_id"], m["before_id"]) for m in reverse["matches"]
    }

    def signature(doc, aa, bb):
        return sorted(
            (
                aa.detections[m["before_id"]].class_id,
                aa.detections[m["before_id"]].bbox,
                bb.detections[m["after_id"]].bbox,
            )
            for m in doc["matches"]
        )

    aa, bb = make(list(reversed(before))), make(list(reversed(after)))
    permuted = compare(aa, bb, threshold)
    assert signature(c, a, b) == signature(permuted, aa, bb)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_endpoints_preserved(confidence):
    d = detection(score=confidence)
    assert (
        result(observation(d), observation(d)).to_dict()["before"]["detections"][0]["confidence"]
        == confidence
    )


def test_large_motion_stays_unmatched_without_physical_verdict():
    r = result(observation(detection((0, 0, 10, 10))), observation(detection((80, 80, 90, 90))))
    assert len(r.unmatched_before) == len(r.unmatched_after) == 1
    assert not r.matches and not r.to_dict()["physical_change_verified"]


@pytest.mark.parametrize("after_area", [0, 80])
def test_mask_shrink_signed_delta_and_ratio(after_area):
    a, b = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
    a.flat[:100] = True
    b.flat[:after_area] = True
    area = mask_case(a, b).matches[0]["mask_area"]
    assert area["delta"] == after_area - 100
    assert area["relative_delta"] == (after_area - 100) / 100
    assert area["unavailable_reason"] is None


def test_symlink_loop_checkpoint_is_an_actionable_input_error(tmp_path):
    from deltasense import InputError

    path = tmp_path / "loop.pt"
    path.symlink_to(path)
    with pytest.raises(InputError, match="resolve"):
        DeltaSense(path)


def test_embedded_null_checkpoint_is_an_input_error():
    from deltasense import InputError

    with pytest.raises(InputError):
        DeltaSense("invalid\x00.pt")


def test_wide_character_legend_wraps_at_measured_pixel_width():
    from PIL import ImageFont

    from deltasense.result import _wrapped_lines

    font = ImageFont.load_default(size=13)
    label = "W" * 1000
    lines = _wrapped_lines(label, font, 440)
    assert "".join(lines) == label
    assert all(font.getlength(line) <= 440 for line in lines)
    assert _wrapped_lines("", font, 440) == []


def test_default_max_det_dense_ambiguity_keeps_every_observation():
    a = observation(*(detection() for _ in range(300)))
    c = compare(a, a)
    assert c["matches"] == []
    assert len(c["ambiguous_candidates"]) == 90000
    assert len(c["unmatched_before"]) == len(c["unmatched_after"]) == 300
