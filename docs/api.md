# v0.1 API and result contract

## Construction

```python
DeltaSense(
    model, *, device="cpu", conf=0.25, iou=0.7, imgsz=640,
    max_det=300, classes=None, match_iou=0.2,
)
```

`model` is a local `str` or `pathlib.Path` to a trusted `.pt` checkpoint. Only
Ultralytics detection and instance segmentation are supported. PyTorch checkpoint
deserialization can execute code; obtain weights from a trusted source. Missing
checkpoints, other formats, URLs, and other tasks fail explicitly. The checkpoint
is hashed and checked for replacement on each comparison. No fallback model/backend
or automatic weight download is used.

`device` is an explicit upstream device string; `cpu` is the verified baseline.
`conf` and upstream NMS `iou` are finite numbers in [0, 1]. `imgsz` and `max_det`
are positive integers. `classes` is `None` or a nonempty list/tuple of nonnegative
class IDs present in the selected model. Class filters are copied, deduplicated,
and sorted. `match_iou` is finite
and in (0, 1]. Booleans are rejected as numeric settings. Unsupported keyword
arguments produce Python's `TypeError`.

Both observations use the same selected model and inference settings. Fixed-size
padding (`rect=False`), FP32 (`quantize=32`), NMS, and full-resolution masks are
requested. Augmentation, compilation, embedding, plotting, saving, and GUI output
are disabled upstream. Effective stride-rounded image size and resolved device
are recorded rather than assuming they equal the requested values.

On first model initialization, DeltaSense sets process environment defaults
`YOLO_OFFLINE=true` and `YOLO_AUTOINSTALL=false` before its lazy Ultralytics import.
These defaults remain in that process. Conflicting values and Ultralytics already
imported with network/auto-install enabled are rejected. If importing Ultralytics
yourself, set both variables before that import. DeltaSense does not edit user
`settings.json`. Upstream's first import can create its normal configuration file;
`YOLO_CONFIG_DIR` can select a prepared writable directory.

## Input and coordinates

`sense.compare(before, after)` takes exactly two local JPEG/PNG paths. Temporal
order is preserved. Both paths are validated and decoded before inference.
Directories, streams, URLs, unreadable/corrupt files, animated PNGs, high-bit-depth
PNG, and images above Pillow's pixel safety limit are rejected. PNG requires a
complete terminal IEND chunk; JPEG requires a terminal end-of-image marker.
The original PNG header is checked for 16-bit samples in all grayscale, RGB,
and alpha variants; a decoder's implicit reduction to 8 bits is not accepted.

EXIF orientation is applied once. Supported grayscale, palette, CMYK JPEG, and
RGB images become 8-bit RGB. Transparency is composited on white. Original files
are never changed. Already oriented arrays are passed to Ultralytics as BGR,
so upstream file decoding cannot apply a second orientation transform.

Coordinates refer to the decoded, EXIF-oriented original image: x goes right,
y goes down. Width/height must agree after decoding; no image is silently resized,
cropped, or registered to the other. Ordinary model preprocessing is mapped back
to these coordinates by Ultralytics. Full-size binary masks are required; an
unexpected mask shape/count is an observation error, not an invitation to guess
padding, resize masks, or replace them with bounding boxes.

## Matching and arithmetic

Prediction counts include the union of observed class IDs. A missing class has
an observed count of zero on that side. Deltas are always `after - before`.

Every same-class pair with bbox IoU greater than or equal to `match_iou` is a
candidate. A pair is accepted only if each endpoint has exactly one candidate.
All competing candidates stay unresolved, even if one has higher IoU. No greedy
tie breaking, forced assignment, class mixing, or detection reuse is performed.
Degenerate zero-area boxes have IoU zero and acquire no correspondence. This
conservative rule sacrifices some recall: large movement without box overlap can
remain unmatched, and crowded/repeated objects often remain ambiguous.

`unmatched_before` and `unmatched_after` include every detection without an
accepted pair, including ambiguous detections. Their reasons are `no_candidate`
or `ambiguous_correspondence`. `ambiguous_candidates` separately records the
candidate edges and IoU evidence. These are not physical removal/addition events.
Accepted geometry is invariant to input detection ordering; local list IDs can
change when detections are reordered.

For an accepted pair:

- displacement is after-center minus before-center, in pixels;
- bbox area is `(x2 - x1) * (y2 - y1)`, in pixels squared;
- mask area counts foreground pixels in the full-size binary mask;
- area delta is `after_area - before_area`;
- relative area delta divides that delta by before area, a ratio (0.2 means 20%);
  reversing a pair recomputes the denominator;
- mask support differences count before-only, after-only, intersection, and XOR
  pixels. Equal-area translation can yield zero area delta and nonzero XOR.

Missing masks retain an available side's area but have null difference/ratio and
a missing-mask reason. A valid empty mask has area zero. A zero baseline makes
only the relative ratio unavailable (`zero_before_area`); other measurements
remain valid. If finite, extremely small box areas would overflow the floating-point
relative ratio, only that ratio is null with `relative_delta_out_of_range`; the
raw areas and signed delta remain available. Ambiguous/unmatched detections have
null dependent measurements.
No NaN/Infinity is serialized. Small upstream numerical differences are retained;
there is no hidden epsilon or automatic binary change event.

## Python and JSON output

`ComparisonResult` exposes `before`, `after`, `prediction_counts`, `matches`,
`unmatched_before`, `unmatched_after`, `ambiguous_candidates`, and
`unchecked_conditions`. `to_dict()` returns a detached JSON-compatible dictionary;
`to_json(indent=2)` returns its JSON string. Summary properties return copies.
`before` and `after` are read-only properties returning detached observation
snapshots. Their masks have immutable pixel buffers. Altering a retrieved array's
shape/dtype metadata cannot change subsequent JSON, measurements, or plots.

The JSON schema version is `0.1.0`. It includes:

| Field | Meaning |
|---|---|
| `before`, `after` | Image content SHA-256/dimensions, task, local detection IDs, original class names/IDs/scores, boxes, mask availability/area |
| `prediction_counts` | Class-wise before/after counts and signed delta |
| `matches` | Proposed pairs, IoU, displacement, bbox measurements, optional mask measurements |
| `unmatched_before`, `unmatched_after` | No accepted correspondence, with reason and null dependent measurements |
| `ambiguous_candidates` | Unresolved candidate edges and their IoU |
| `unchecked_conditions` | Alignment, illumination, occlusion, physical identity |
| `measurement_assumption` | Caller-aligned images with similar capture conditions |
| `coordinate_convention` | Decoded, EXIF-oriented image; x right, y down |
| `physical_change_verified` | Always false in v0.1 |
| `provenance` | Checkpoint digest, package/runtime versions, effective inference/comparison settings |

Image digests identify original file bytes. Images, private absolute paths, and
dense mask arrays are not embedded in JSON. Provenance identifies a run; it does
not promise bit-identical inference across devices or repeated observations.

## Plotting and saving

`result.plot()` returns a new RGB `PIL.Image.Image`. It opens no window and writes
no file. Save explicitly with `.save(path)`. The result retains image/mask snapshots
for plotting, so memory grows with decoded image size and the number of masks.

Before and after panels show indexed bbox tags. Separate legends keep status,
class name, and confidence legible when boxes overlap. `M` means proposed match,
`U` unmatched, and `?` ambiguous. Text and color both encode status. Cyan/orange
show before-only/after-only mask support for accepted pairs. The footer retains
unchecked conditions and the physical-change limitation, including empty results.
Names unsupported by the bundled Latin font use readable Unicode escapes in the
plot; the original Unicode name remains unchanged in Python/JSON.

## Errors

All library failures derive from `DeltaSenseError`:

| Error | Meaning |
|---|---|
| `InputError` (`ValueError`) | Invalid image/checkpoint/setting; dimension mismatch; changed checkpoint; conflicting offline settings |
| `UnsupportedTaskError` (`InputError`) | Selected model is not detection/instance segmentation |
| `InferenceError` (`RuntimeError`) | Model load/predict failure or wrong result count/type; original exception is chained |
| `ObservationError` (`ValueError`) | Invalid result coordinates, class/score, dimensions, mask data/mapping, or inconsistent names |

A successful comparison with zero detections is different from an input/model
failure and from a successful comparison with unavailable measurements. Equal
dimensions and successful matching do not establish verified capture conditions.
Comparisons sharing a `DeltaSense` instance serialize upstream prediction calls;
construct separate instances for independent parallel models.
