import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deltasense.errors import InputError, ObservationError
from deltasense.observations import Detection, ImageIdentity, Observation, read_image


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".png", ".JPG", ".PNG"])
def test_valid_paths_digests_and_originals_unchanged(tmp_path, suffix):
    path = tmp_path / f"日本語 image{suffix}"
    Image.new("RGB", (13, 17), (230, 20, 10)).save(path)
    contents = path.read_bytes()
    pixels, identity = read_image(path)
    assert pixels.shape == (17, 13, 3) and pixels.dtype == np.uint8
    assert identity.sha256 == hashlib.sha256(contents).hexdigest()
    assert path.read_bytes() == contents
    assert pixels[0, 0, 0] > 220 and pixels[0, 0, 2] < 15


@pytest.mark.parametrize(
    "input_value", [None, 0, 1, [], {}, b"a.png", "https://host/image.png", "rtsp://host/image.jpg"]
)
def test_A13_unsupported_input_types_and_urls(input_value):
    with pytest.raises(InputError):
        read_image(input_value)


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "wrong_extension", "corrupt", "disguised_gif", "truncated"]
)
def test_A13_invalid_images_are_actionable_errors(tmp_path, kind):
    path = tmp_path / "input.png"
    if kind == "directory":
        path.mkdir()
    elif kind == "wrong_extension":
        path = tmp_path / "input.bmp"
        Image.new("RGB", (10, 10)).save(path)
    elif kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "disguised_gif":
        Image.new("RGB", (10, 10)).save(path, format="GIF")
    elif kind == "truncated":
        Image.new("RGB", (30, 30)).save(path)
        path.write_bytes(path.read_bytes()[:40])
    with pytest.raises(InputError):
        read_image(path)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_eight_exif_orientations(tmp_path, orientation):
    # Asymmetric corner colors independently identify every flip/rotation.
    source = np.zeros((2, 3, 3), np.uint8)
    source[..., 0] = [[10, 20, 30], [40, 50, 60]]
    path = tmp_path / f"orientation{orientation}.png"
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(source).save(path, exif=exif)
    expected = {
        1: source,
        2: source[:, ::-1],
        3: source[::-1, ::-1],
        4: source[::-1],
        5: source.transpose(1, 0, 2),
        6: np.rot90(source, -1),
        7: source.transpose(1, 0, 2)[::-1, ::-1],
        8: np.rot90(source, 1),
    }[orientation]
    pixels, identity = read_image(path)
    np.testing.assert_array_equal(pixels, expected)
    assert (identity.height, identity.width) == expected.shape[:2]


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA"])
def test_supported_png_color_modes(tmp_path, mode):
    path = tmp_path / "colors.png"
    Image.new(mode, (7, 5)).save(path)
    pixels, identity = read_image(path)
    assert pixels.shape == (5, 7, 3) and identity.width == 7


def test_transparent_png_is_composited_on_white(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (5, 5), (255, 0, 0, 0)).save(path)
    pixels, _ = read_image(path)
    assert (pixels == 255).all()


def test_palette_transparency_is_composited_on_white(tmp_path):
    path = tmp_path / "palette.png"
    image = Image.new("P", (5, 5), 0)
    image.putpalette([0, 0, 0] * 256)
    image.save(path, transparency=0)
    pixels, _ = read_image(path)
    assert (pixels == 255).all()


def test_cmyk_jpeg_is_converted_to_rgb(tmp_path):
    path = tmp_path / "cmyk.jpg"
    Image.new("CMYK", (7, 5), (0, 255, 255, 0)).save(path)
    pixels, _ = read_image(path)
    assert pixels[0, 0].tolist() == [255, 0, 0]


def test_high_bit_depth_and_animation_rejected(tmp_path):
    path = tmp_path / "16bit.png"
    Image.fromarray(np.zeros((5, 5), np.uint16)).save(path)
    with pytest.raises(InputError, match="16-bit"):
        read_image(path)
    path = tmp_path / "animated.png"
    Image.new("RGB", (5, 5)).save(
        path, save_all=True, append_images=[Image.new("RGB", (5, 5), "red")]
    )
    with pytest.raises(InputError, match="animated"):
        read_image(path)


def test_decoder_read_failures_are_input_errors(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.png"
    Image.new("RGB", (5, 5)).save(path)

    def fail(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", fail)
    with pytest.raises(InputError, match="could not decode"):
        read_image(path)


def test_image_bomb_is_not_silently_decoded(tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    Image.new("RGB", (20, 20)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(InputError, match="decode"):
        read_image(path)


@pytest.mark.parametrize(
    "box",
    [
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (10, 0, 0, 10),
        (0, 10, 10, 0),
        (0, 0, float("nan"), 10),
        (0, 0, float("inf"), 10),
        (0, 0, 10),
    ],
)
def test_malformed_boxes_rejected(box):
    with pytest.raises(ObservationError):
        Detection(0, "item", 0.9, box)


@pytest.mark.parametrize("class_id", [-1, 0.5, True, "0"])
def test_invalid_class_ids_rejected(class_id):
    with pytest.raises(ObservationError):
        Detection(class_id, "item", 0.9, (0, 0, 10, 10))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ObservationError):
        Detection(0, "item", confidence, (0, 0, 10, 10))


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(5),
        np.zeros((2, 2, 2)),
        np.full((5, 5), 0.5),
        np.full((5, 5), float("nan")),
        np.full((5, 5), 255),
    ],
)
def test_nonbinary_and_malformed_masks_rejected(mask):
    with pytest.raises(ObservationError):
        Detection(0, "item", 0.9, (0, 0, 10, 10), mask)


@pytest.mark.parametrize(
    "kind",
    ["bbox_outside", "mask_size", "mask_in_detection", "class_name_conflict", "unsupported_task"],
)
def test_inconsistent_observations_rejected(kind):
    identity = ImageIdentity("a" * 64, 10, 10)
    item = Detection(0, "item", 0.9, (0, 0, 10, 10))
    task, items = "detect", [item]
    if kind == "bbox_outside":
        items = [Detection(0, "item", 0.9, (0, 0, 11, 10))]
    elif kind == "mask_size":
        task = "segment"
        items = [Detection(0, "item", 0.9, (0, 0, 10, 10), np.zeros((5, 5)))]
    elif kind == "mask_in_detection":
        items = [Detection(0, "item", 0.9, (0, 0, 10, 10), np.zeros((10, 10)))]
    elif kind == "class_name_conflict":
        items.append(Detection(0, "different", 0.9, (0, 0, 10, 10)))
    else:
        task = "pose"
    with pytest.raises(ObservationError):
        Observation(identity, task, tuple(items))
