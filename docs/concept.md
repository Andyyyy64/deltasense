# Project concept

Most computer vision libraries analyze one image at a time. DeltaSense is about the gap between two observations.

The project starts with Ultralytics because it already gives detection, segmentation, classification, pose, oriented boxes, and depth a consistent result format. DeltaSense can run one selected model on two images and compare those results.

## Initial direction

v0.1 implements detection and instance-segmentation
comparison. The original starting direction below describes the product intent;
[release requirements](requirements.md) and [the API contract](api.md) define the
implemented scope. See [validation](validation.md) for what has actually been tested.

The first useful version should stay small:

1. Accept a before image and an after image.
2. Run the same Ultralytics model on both.
3. Compare the returned results.
4. Report changes in a plain Python object that can also be written as JSON.

v0.1 compares prediction counts, conservative spatial correspondences, bbox
displacement/area, and mask area/support differences. It does not infer physical
shape changes from a spatial mask difference. Pose and depth should wait until a
real use case needs them.

## What model output can and cannot prove

A difference between predictions is not always a difference in the photographed subject. The camera may have moved. Lighting may have changed. An object may be hidden, blurred, or missed by the model.

DeltaSense reports those limits instead of turning every difference into a change
claim. v0.1 distinguishes measurement availability and explicitly unchecked capture
conditions. It does not assign a global verified-comparable/unobservable/confounded
verdict without the evidence to support it.

## Why keep it generic

The same comparison pattern appears in many places: a scratch on a manufactured part, growth in a plant, work completed on a building, or a subtle change in a person. The models and measurements differ, but the before and after workflow repeats.

DeltaSense should not include domain logic until someone brings a real example and a test. The initial project only needs to prove that one small interface can compare Ultralytics results without hiding what the model actually observed.
