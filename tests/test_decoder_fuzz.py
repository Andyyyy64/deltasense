import warnings
from io import BytesIO

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

from deltasense import InputError
from deltasense.observations import read_image


def encoded(format):
    stream = BytesIO()
    Image.new("RGB", (13, 17), (50, 100, 150)).save(stream, format=format)
    return stream.getvalue()


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_every_truncation_offset_returns_an_input_error(tmp_path, format):
    contents = encoded(format)
    path = tmp_path / ("truncated.png" if format == "PNG" else "truncated.jpg")
    for cut in range(len(contents)):
        path.write_bytes(contents[:cut])
        with pytest.raises(InputError):
            read_image(path)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.sampled_from(["PNG", "JPEG"]), st.integers(0, 10000), st.integers(0, 255))
def test_mutated_image_bytes_never_return_unclassified_decode_failures(
    tmp_path, format, offset, value
):
    contents = bytearray(encoded(format))
    contents[offset % len(contents)] = value
    path = tmp_path / ("mutated.png" if format == "PNG" else "mutated.jpg")
    path.write_bytes(contents)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pixels, identity = read_image(path)
        except InputError:
            return
    assert pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
    assert (identity.height, identity.width) == pixels.shape[:2]
