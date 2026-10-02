"""Local-checkpoint integration through the public Ultralytics predict interface."""

import hashlib
import os
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from threading import Lock

from .comparison import ComparisonSettings, compare_observations
from .errors import InferenceError, InputError, UnsupportedTaskError
from .observations import from_ultralytics, read_image
from .result import ComparisonResult


def _unit_interval(name, value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise InputError(f"{name} must be numeric and in [0, 1]")
    if not 0 <= value <= 1:
        raise InputError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer")
    return value


def _checkpoint_digest(path):
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise InputError(f"checkpoint is not readable: {path}") from exc


def _load_model(path):
    # Public upstream environment controls, set before the lazy import. These are
    # process-wide: fail on conflicting/preimported settings rather than mutate
    # user settings.json or monkey-patch upstream network code.
    for key, value in (("YOLO_OFFLINE", "true"), ("YOLO_AUTOINSTALL", "false")):
        os.environ.setdefault(key, value)
    utils = sys.modules.get("ultralytics.utils")
    if utils is not None and (utils.ONLINE or utils.AUTOINSTALL):
        raise InputError(
            "Ultralytics was already imported with network/auto-install enabled. "
            "Start a fresh process with YOLO_OFFLINE=true YOLO_AUTOINSTALL=false."
        )
    if os.environ["YOLO_OFFLINE"].lower() not in {"true", "1", "yes", "on", "y", "t"} or (
        os.environ["YOLO_AUTOINSTALL"].lower() not in {"false", "0", "no", "off", "n", "f"}
    ):
        raise InputError("DeltaSense requires YOLO_OFFLINE=true and YOLO_AUTOINSTALL=false")
    try:
        from ultralytics import YOLO

        return YOLO(str(path), verbose=False)
    except Exception as exc:
        raise InferenceError(
            "could not load the selected local checkpoint; no fallback was used"
        ) from exc


class DeltaSense:
    """Compare exactly two local images, assuming the caller has aligned them.

    Only local trusted .pt detection/instance-segmentation weights are supported.
    Settings are shared by both observations. CPU is the verified baseline.
    """

    def __init__(
        self,
        model,
        *,
        device="cpu",
        conf=0.25,
        iou=0.7,
        imgsz=640,
        max_det=300,
        classes=None,
        match_iou=0.2,
    ):
        if not isinstance(model, (str, Path)) or (isinstance(model, str) and "://" in model):
            raise InputError("model must be a local .pt checkpoint path")
        try:
            path = Path(model).expanduser().resolve()
            is_checkpoint = path.suffix.lower() == ".pt" and path.is_file()
        except (OSError, ValueError, RuntimeError) as exc:
            raise InputError("could not resolve the local checkpoint path") from exc
        if not is_checkpoint:
            raise InputError(
                f"local .pt checkpoint does not exist: {path}; provision it separately"
            )
        if not isinstance(device, str) or not device.strip():
            raise InputError("device must be a nonempty string, such as cpu or cuda:0")
        if classes is not None:
            if (
                not isinstance(classes, (list, tuple))
                or not all(
                    isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in classes
                )
                or not classes
            ):
                raise InputError("classes must be None or a nonempty list/tuple of nonnegative IDs")
            classes = sorted(set(classes))
        self._settings = ComparisonSettings(match_iou)
        self._predict_settings = {
            "device": device,
            "conf": _unit_interval("conf", conf),
            "iou": _unit_interval("iou", iou),
            "imgsz": _positive_int("imgsz", imgsz),
            "max_det": _positive_int("max_det", max_det),
            "classes": classes,
            "rect": False,
            "retina_masks": True,
            "augment": False,
            "nms": True,
            "quantize": 32,
            "compile": False,
            "save": False,
            "save_txt": False,
            "save_crop": False,
            "save_frames": False,
            "show": False,
            "verbose": False,
            "visualize": False,
            "embed": None,
        }
        self._path = path
        self._digest = _checkpoint_digest(path)
        self._model = _load_model(path)
        self._task = self._model.task
        if self._task not in {"detect", "segment"}:
            raise UnsupportedTaskError(f"task {self._task!r} is unsupported; use detect or segment")
        if classes is not None and any(cls not in self._model.names for cls in classes):
            raise InputError("classes contains an ID absent from the selected model's class names")
        if _checkpoint_digest(path) != self._digest:
            raise InputError("checkpoint changed while loading; create a new DeltaSense instance")
        self._lock = Lock()

    def compare(self, before, after):
        image_a, identity_a = read_image(before)
        image_b, identity_b = read_image(after)
        if (identity_a.width, identity_a.height) != (identity_b.width, identity_b.height):
            raise InputError(
                "images must have equal decoded dimensions; no resizing/alignment is performed"
            )
        with self._lock:
            if _checkpoint_digest(self._path) != self._digest:
                raise InputError(
                    "checkpoint changed after loading; create a new DeltaSense instance"
                )
            try:
                # One fixed-size batch, with originals already decoded/oriented.
                results = self._model.predict(
                    source=[image_a[..., ::-1].copy(), image_b[..., ::-1].copy()],
                    stream=False,
                    **self._predict_settings,
                )
            except Exception as exc:
                raise InferenceError(
                    "before/after inference failed; no comparison or fallback was returned"
                ) from exc
            if not isinstance(results, (list, tuple)) or len(results) != 2:
                raise InferenceError("model must return exactly one result per supplied image")
            observation_a = from_ultralytics(results[0], identity_a, self._task)
            observation_b = from_ultralytics(results[1], identity_b, self._task)
            predictor = self._model.predictor
            effective = {k: getattr(predictor.args, k) for k in self._predict_settings}
            effective["imgsz"] = list(predictor.imgsz)
            effective["resolved_device"] = str(predictor.device)
            provenance = {
                "deltasense_version": version("deltasense"),
                "ultralytics_version": version("ultralytics"),
                "numpy_version": version("numpy"),
                "pillow_version": version("Pillow"),
                "torch_version": version("torch"),
                "checkpoint_sha256": self._digest,
                "task": self._task,
                "inference_settings": effective,
                "comparison_settings": {
                    **asdict(self._settings),
                    "matching_rule": "same_class_mutual_single_candidate",
                },
            }
        comparison = compare_observations(observation_a, observation_b, self._settings)
        return ComparisonResult(
            observation_a, observation_b, comparison, provenance, (image_a, image_b)
        )
