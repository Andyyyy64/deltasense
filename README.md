# DeltaSense

DeltaSense compares what one computer vision model observes in two images from
different times. It reports prediction counts, proposed object correspondences,
image-space displacement, bounding-box and segmentation-mask area differences,
and unavailable-measurement reasons.

**Status:** v0.1.0 source implementation on `main`. No package or GitHub Release
has been published. License and package publication destination remain owner
decisions. Local verification and its
limits are recorded in [the validation report](docs/validation.md).
The [final local readiness review](docs/v0.1-readiness.md) maps every v0.1 requirement
to evidence and separates local completion from the deferred publication steps.
See [the changelog](CHANGELOG.md) for the v0.1.0 scope and
[GitHub Actions](https://github.com/Andyyyy64/deltasense/actions/workflows/ci.yml)
for the checks on each commit.

## Project goal

Compare corresponding regions of the same subject at different times, accounting
for differences in how the images were captured. Explain meaningful changes in
the subject, including what changed, by how much, and the supporting evidence.
Distinguish measurements from interpretations, and explain why a conclusion
cannot be reached when the evidence is insufficient.

For example, the goal is to explain whether a person has moved forward along a
road, rather than treating every image-coordinate difference as physical movement.
Comparing the appearance of hair or facial contours over time is another example;
each use case needs its own measurements and evidence for interpretation.

**v0.1 provides the comparison foundation for this goal.** It compares model
predictions in image coordinates; it does not yet determine real-world movement,
hair loss, or weight change. See [the project concept](docs/concept.md) for examples,
the distinction between measurements and interpretations, and criteria for progress.

## Install locally

The verified baseline is CPython 3.12 on Linux x86-64 with CPU inference. Runtime
dependencies are pinned to the versions used in verification. GPU, other Python
versions, other operating systems, and other Ultralytics versions are not yet
verified.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install .
```

Obtain a trusted local Ultralytics `.pt` detection or instance-segmentation
checkpoint separately. DeltaSense never acquires or substitutes model weights.

## Compare two images

Supply two local JPEG/PNG images with the same framing, scale, orientation, and
decoded dimensions. Align them outside DeltaSense when needed.

```python
from deltasense import DeltaSense

sense = DeltaSense("./models/yolo26n-seg.pt", device="cpu", conf=0.25, match_iou=0.2)
result = sense.compare("before.jpg", "after.jpg")

print(result.prediction_counts)
print(result.matches)
print(result.to_json())
result.plot().save("comparison.png")  # saving is explicit; plot() only returns an image
```

## Interpreting results

- A prediction-count delta of +1 means the model detected one more instance.
- A match is a proposed correspondence under a conservative spatial rule.
- Unmatched observations can be missed detections, occlusions, or ambiguous
  candidates. They do not confirm physical appearance or disappearance.
- Movement and areas use decoded image pixels, not physical units.
- Missing measurements are `null` with a reason; calculated zero stays zero.
- Alignment, lighting, occlusion, and physical identity are explicitly unchecked
  in every result. Equal dimensions do not verify comparable capture conditions.

Even identical image pixels can produce small numerical differences during
upstream inference. DeltaSense preserves those observed deltas and does not turn
them into a scene-level changed/unchanged verdict.

## Development and verification

```sh
.venv/bin/python -m pip install '.[dev]'
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest --cov=deltasense --cov-branch
```

Ordinary tests prohibit network connections and do not download/run pretrained
models. Explicit real-model verification is separate; see
[reproduction instructions](docs/validation.md).

- [API, matching, units, and errors](docs/api.md)
- [Release requirements](docs/requirements.md), mirrored from [issue #1](https://github.com/Andyyyy64/deltasense/issues/1)
- [Validation evidence and limitations](docs/validation.md)
- [Real-image accuracy evaluation with provisional labels (Japanese)](docs/accuracy.md)
- [Accuracy and configuration comparison on manually labeled video (Japanese)](docs/mot17-accuracy.md)
- [Evaluation data and reproduction instructions](evaluation/README.md)
- [Project concept](docs/concept.md)

v0.2 is planned to add bounded automatic camera alignment and geometric validity
checks. v0.1 relies on caller-prepared images.
