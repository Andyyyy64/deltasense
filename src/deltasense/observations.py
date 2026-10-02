"""Validated snapshots in decoded, EXIF-oriented original-image coordinates."""

import hashlib
import math
import struct
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin, PngImagePlugin

from .errors import InputError, ObservationError


@dataclass(frozen=True)
class ImageIdentity:
    sha256: str
    width: int
    height: int

    def to_dict(self):
        return {"sha256": self.sha256, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ObservationError("class_id must be a nonnegative integer")
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ObservationError("class_name must be a nonempty string")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ObservationError("confidence must be finite and in [0, 1]")
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
            raise ObservationError("bbox must contain four finite xyxy coordinates")
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1:
            raise ObservationError("bbox coordinates must be nonnegative and ordered")
        object.__setattr__(self, "bbox", tuple(float(v) for v in self.bbox))
        object.__setattr__(self, "confidence", float(self.confidence))
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
                raise ObservationError("mask must be a two-dimensional binary array")
            # An owning read-only ndarray can have writes re-enabled. Bytes-backed
            # data prevents mutation even through setflags() or a base-array view.
            mask = np.frombuffer(np.asarray(mask, dtype=bool).tobytes(), dtype=bool).reshape(
                mask.shape
            )
            object.__setattr__(self, "mask", mask)

    def to_dict(self, local_id):
        return {
            "id": local_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox),
            "mask_area": None if self.mask is None else int(self.mask.sum()),
            "mask_available": self.mask is not None,
        }


@dataclass(frozen=True)
class Observation:
    identity: ImageIdentity
    task: str
    detections: tuple[Detection, ...]

    def __post_init__(self):
        if self.task not in {"detect", "segment"}:
            raise ObservationError("observation task must be detect or segment")
        if self.identity.width <= 0 or self.identity.height <= 0:
            raise ObservationError("image dimensions must be positive")
        names = {}
        object.__setattr__(self, "detections", tuple(self.detections))
        for detection in self.detections:
            if detection.bbox[2] > self.identity.width or detection.bbox[3] > self.identity.height:
                raise ObservationError("bbox is outside the decoded image")
            if names.setdefault(detection.class_id, detection.class_name) != detection.class_name:
                raise ObservationError("inconsistent names for one class ID")
            if detection.mask is not None:
                if self.task != "segment":
                    raise ObservationError("detection task unexpectedly returned a mask")
                if detection.mask.shape != (self.identity.height, self.identity.width):
                    raise ObservationError("mask dimensions do not match the decoded image")

    def to_dict(self):
        return {
            "image": self.identity.to_dict(),
            "task": self.task,
            "detections": [d.to_dict(i) for i, d in enumerate(self.detections)],
        }


def read_image(path):
    if not isinstance(path, (str, Path)):
        raise InputError("image must be a local JPEG or PNG path (str or pathlib.Path)")
    if isinstance(path, str) and "://" in path:
        raise InputError("URLs and streams are unsupported; provide a local image")
    file = Path(path)
    try:
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not file.is_file():
            raise InputError(f"not a readable local JPEG or PNG file: {file}")
        contents = file.read_bytes()
        if contents.startswith(b"\x89PNG\r\n\x1a\n"):
            if not contents.endswith(b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"):
                raise InputError("PNG is incomplete or has data after its terminal IEND chunk")
            # IHDR stores sample depth at byte 24. Pillow's decoded RGB/RGBA
            # modes hide 16-bit color depth by silently reducing it to 8 bits.
            if contents[24:25] == b"\x10":
                raise InputError("16-bit images are unsupported; use 8-bit images")
            decoder = PngImagePlugin.PngImageFile
        elif contents.startswith(b"\xff\xd8\xff"):
            if not contents.endswith(b"\xff\xd9"):
                raise InputError("JPEG is missing its terminal end-of-image marker")
            decoder = JpegImagePlugin.JpegImageFile
        else:
            raise InputError("file contents must be JPEG or PNG")
        # Ultralytics patches Image.open to try installing HEIF on any error.
        # Explicit public JPEG/PNG decoders keep corrupt inputs as input errors.
        with decoder(BytesIO(contents)) as source:
            if (
                Image.MAX_IMAGE_PIXELS is not None
                and source.width * source.height > Image.MAX_IMAGE_PIXELS
            ):
                raise InputError(
                    "could not decode image: pixel count exceeds Pillow's safety limit"
                )
            if getattr(source, "n_frames", 1) != 1:
                raise InputError("animated images are unsupported")
            source.verify()
        with decoder(BytesIO(contents)) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert(
                "RGB"
            )
            identity = ImageIdentity(hashlib.sha256(contents).hexdigest(), *image.size)
            return np.array(image, dtype=np.uint8), identity
    except InputError:
        raise
    except (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError) as exc:
        raise InputError(f"could not decode image: {file}") from exc


def from_ultralytics(result, identity, task):
    """Read public Results fields; full-size masks are required, never guessed/resized."""
    try:
        if tuple(result.orig_shape) != (identity.height, identity.width):
            raise ObservationError("model result dimensions do not match the supplied image")
        if result.boxes is None:
            raise ObservationError("supported model result is missing boxes")
        boxes = result.boxes.cpu().numpy()
        xyxy, scores, classes = boxes.xyxy, boxes.conf, boxes.cls
        count = len(xyxy)
        if xyxy.shape != (count, 4) or scores.shape != (count,) or classes.shape != (count,):
            raise ObservationError("box, score, and class dimensions are inconsistent")
        masks = None
        if result.masks is not None:
            if task != "segment":
                raise ObservationError("detection task returned segmentation masks")
            masks = result.masks.cpu().numpy().data
            if masks.shape != (count, identity.height, identity.width):
                raise ObservationError("mask count/shape is inconsistent with boxes/original image")
        detections = []
        for i in range(count):
            value = float(classes[i])
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ObservationError("model returned a nonintegral/invalid class ID")
            cls = int(value)
            if cls not in result.names:
                raise ObservationError("model class ID has no class name")
            detections.append(
                Detection(
                    cls,
                    result.names[cls],
                    float(scores[i]),
                    tuple(xyxy[i]),
                    None if masks is None else masks[i],
                )
            )
        return Observation(identity, task, tuple(detections))
    except ObservationError:
        raise
    except (AttributeError, TypeError, ValueError, IndexError, KeyError) as exc:
        raise ObservationError("malformed Ultralytics result") from exc
