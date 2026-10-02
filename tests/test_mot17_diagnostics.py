import io
import runpy
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return runpy.run_path(str(ROOT / "scripts" / name))


def obj(identity, coords=(0, 0, 10, 10)):
    return {"object_id": identity, "class_id": 0, "bbox_xyxy": list(coords)}


def pred(local_id, coords=(0, 0, 10, 10)):
    return {
        "id": local_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox_xyxy": list(coords),
        "mask_available": True,
        "mask_area": 100,
    }


@pytest.mark.parametrize(
    "reason",
    [
        "correct",
        "recognition_or_localization_missing",
        "competing_candidates",
        "accepted_different_identity",
        "no_accepted_candidate",
    ],
)
def test_failure_attribution_partitions_known_identity(reason, monkeypatch):
    namespace = load("analyze_mot17.py", monkeypatch)
    a = b = {"objects": [obj("target")], "scope_class_ids": [0]}
    doc = {
        "before": {"detections": [pred(0)]},
        "after": {"detections": [pred(0)]},
        "matches": [],
        "unmatched_before": [],
        "unmatched_after": [],
    }
    if reason == "correct":
        doc["matches"] = [{"class_id": 0, "before_id": 0, "after_id": 0}]
    elif reason == "recognition_or_localization_missing":
        doc["after"]["detections"] = []
    elif reason == "competing_candidates":
        doc["unmatched_before"] = [{"id": 0, "reason": "ambiguous_correspondence"}]
    elif reason == "accepted_different_identity":
        doc["after"]["detections"].append(pred(1, (30, 0, 40, 10)))
        doc["matches"] = [{"class_id": 0, "before_id": 0, "after_id": 1}]
    else:
        doc["unmatched_before"] = [{"id": 0, "reason": "no_candidate"}]
    assert namespace["attribution"](doc, a, b) == {reason: 1}


def test_bbox_reconstruction_keeps_counts_and_rejects_changed_baseline(monkeypatch):
    ns = load("evaluate_matching.py", monkeypatch)
    side = {
        "image": {"sha256": "a" * 64, "width": 100, "height": 100},
        "task": "segment",
        "detections": [pred(0)],
    }
    original = {"before": side, "after": side}
    baseline = ns["rematch"](original, 0.2)
    baseline["matches"][0]["mask_area"] = {"before": 100, "after": 100}
    rebuilt = ns["rematch"](original, 0.2)
    ns["verify_baseline"](baseline, rebuilt)
    assert side["task"] == "segment" and side["detections"][0]["mask_area"] == 100
    baseline["matches"][0]["after_id"] = 1
    with pytest.raises(ValueError, match="correspondences differ"):
        ns["verify_baseline"](baseline, rebuilt)


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_bounded_zip_entry_extraction_and_crc_verification(compression, monkeypatch):
    ns = load("prepare_mot17.py", monkeypatch)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("frames/test.jpg", b"test-payload" * 100)
    raw = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo("frames/test.jpg")
    function = ns["extract_entry"]
    function.__globals__["get_range"] = lambda url, start, length, size, etag: raw[
        start : start + length
    ]
    plan = {
        "archive_url": "https://example.invalid/",
        "archive_size": len(raw),
        "archive_etag": "tag",
    }
    assert function(info, plan) == b"test-payload" * 100
    info.CRC += 1
    with pytest.raises(ValueError, match="CRC mismatch"):
        function(info, plan)


@pytest.mark.parametrize(
    "start,length,size", [(-1, 1, 10), (0, 20, 10), (0, 16_000_001, 20_000_000)]
)
def test_invalid_ranges_fail_before_network(start, length, size, monkeypatch):
    ns = load("prepare_mot17.py", monkeypatch)
    with pytest.raises(ValueError, match="archive range"):
        ns["get_range"]("https://example.invalid/", start, length, size, "etag")


def test_server_ignoring_range_is_rejected_before_reading_body(monkeypatch):
    ns = load("prepare_mot17.py", monkeypatch)

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            pytest.fail("must not read the whole archive")

    ns["get_range"].__globals__["urlopen"] = lambda *args, **kwargs: Response()
    with pytest.raises(ValueError, match="bounded byte range"):
        ns["get_range"]("https://example.invalid/", 0, 2, 10, "etag")
