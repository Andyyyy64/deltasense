# DeltaSense

DeltaSense is a public project for comparing what a computer vision model sees in images captured at different times.

DeltaSense runs the same model on a before image and an after image, then describes how its results changed. A model may find more objects, fewer objects, a larger segmented area, or a different pose. DeltaSense keeps that comparison logic in one place.

The first version is planned as a small wrapper around [Ultralytics](https://github.com/ultralytics/ultralytics).

## Proposed usage

```python
from deltasense import DeltaSense

sense = DeltaSense("yolo26n-seg.pt")
result = sense.compare("before.jpg", "after.jpg")

result.plot()
print(result.to_json())
```

There is no installable package yet. The example above shows the intended interface.

## What DeltaSense should answer

- What appeared or disappeared?
- What moved?
- What became larger or smaller?
- Which result cannot be compared reliably?

DeltaSense will compare model output. It will not claim that a physical change occurred when blur, lighting, viewpoint, occlusion, or a missed detection could explain the difference.

Read [the project concept](docs/concept.md) for the current direction and limits.
