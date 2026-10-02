"""Regressions discovered by the final v0.1 contract audit."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_comparison import compare, detection, observation, result

from deltasense import ComparisonResult, DeltaSense, InputError
from deltasense.observations import read_image


def png_chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


@pytest.mark.parametrize("color_type,channels", [(0, 1), (2, 3), (4, 2), (6, 4)])
def test_16bit_png_rejected_in_every_color_family(tmp_path, color_type, channels):
    # Pillow exposes RGB/alpha 16-bit PNG as RGB/RGBA after losing precision.
    # A valid file with genuine 16-bit samples is required to catch that path.
    header = struct.pack(">IIBBBBB", 2, 2, 16, color_type, 0, 0, 0)
    scanline = b"\0" + b"\x12\x34" * (2 * channels)
    contents = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanline * 2))
        + png_chunk(b"IEND", b"")
    )
    path = tmp_path / "16bit.png"
    path.write_bytes(contents)
    with Image.open(path) as image:
        image.load()  # Prove the regression fixture is a decodable valid PNG.
        assert image.size == (2, 2)
    with pytest.raises(InputError, match="16-bit"):
        read_image(path)
    assert path.read_bytes() == contents


@pytest.mark.parametrize("side", ["before", "after"])
def test_observations_cannot_be_reassigned_after_comparison(side):
    r = result(observation(detection()), observation(detection()))
    saved = r.to_json()
    with pytest.raises(AttributeError):
        setattr(r, side, observation())
    assert r.to_json() == saved


def test_exposed_mask_cannot_reenable_writes():
    a = observation(detection(mask=np.ones((100, 100), bool)), task="segment")
    r = result(a, a)
    saved = r.to_json()
    with pytest.raises(ValueError):
        r.before.detections[0].mask.flags.writeable = True
    assert r.to_json() == saved


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_exposed_or_original_mask_metadata_cannot_corrupt_result():
    a = observation(detection(mask=np.ones((100, 100), bool)), task="segment")
    r = result(a, a)
    saved, plot = r.to_json(), np.asarray(r.plot()).copy()
    r.before.detections[0].mask.shape = (10000, 1)
    r.after.detections[0].mask.shape = (1, 10000)
    a.detections[0].mask.shape = (10000, 1)
    assert r.before.detections[0].mask.shape == (100, 100)
    assert r.after.detections[0].mask.shape == (100, 100)
    assert r.to_json() == saved
    np.testing.assert_array_equal(r.plot(), plot)


@pytest.mark.parametrize("name", ["conf", "iou", "match_iou"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_extreme_integer_thresholds_are_input_errors(tmp_path, name, sign):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"unreached checkpoint")
    with pytest.raises(InputError):
        DeltaSense(weights, **{name: sign * 10**1000})


@pytest.mark.parametrize("kind", ["image", "checkpoint"])
def test_unreadable_path_metadata_is_an_input_error(tmp_path, monkeypatch, kind):
    path = tmp_path / ("image.png" if kind == "image" else "local.pt")
    path.write_bytes(b"unreached file")

    def denied(self):
        raise PermissionError("stat denied")

    monkeypatch.setattr(Path, "is_file", denied)
    with pytest.raises(InputError):
        if kind == "image":
            read_image(path)
        else:
            DeltaSense(path)


def test_finite_boxes_do_not_produce_infinite_relative_area_in_json():
    a = observation(detection((0, 0, 1e-160, 1e-160)))
    b = observation(detection((0, 0, 1, 1)))
    images = (np.zeros((100, 100, 3), np.uint8),) * 2
    r = ComparisonResult(a, b, compare(a, b, threshold=5e-324), {}, images)
    area = r.matches[0]["bbox_area"]
    assert area["before"] > 0 and area["delta"] == 1
    assert area["relative_delta"] is None
    assert area["relative_unavailable_reason"] == "relative_delta_out_of_range"
    assert area["unavailable_reason"] is None
    assert r.to_json()
    reverse = ComparisonResult(b, a, compare(b, a, threshold=5e-324), {}, images)
    assert reverse.matches[0]["bbox_area"]["relative_delta"] == -1
    assert reverse.matches[0]["bbox_area"]["relative_unavailable_reason"] is None
