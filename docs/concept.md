# Project concept

Most computer vision libraries analyze one image at a time. DeltaSense is about the gap between two observations.

The project starts with Ultralytics because it already gives detection, segmentation, classification, pose, oriented boxes, and depth a consistent result format. DeltaSense can run one selected model on two images and compare those results.

## Initial direction

The first useful version should stay small:

1. Accept a before image and an after image.
2. Run the same Ultralytics model on both.
3. Compare the returned results.
4. Report changes in a plain Python object that can also be written as JSON.

For detections, that may mean class count changes. Segmentation can later add area and shape changes. Pose and depth should wait until a real use case needs them.

## What model output can and cannot prove

A difference between predictions is not always a difference in the photographed subject. The camera may have moved. Lighting may have changed. An object may be hidden, blurred, or missed by the model.

DeltaSense should report those limits instead of turning every difference into a change claim. A result may be comparable, unobservable, or confounded by the capture conditions.

## Why keep it generic

The same comparison pattern appears in many places: a scratch on a manufactured part, growth in a plant, work completed on a building, or a subtle change in a person. The models and measurements differ, but the before and after workflow repeats.

DeltaSense should not include domain logic until someone brings a real example and a test. The initial project only needs to prove that one small interface can compare Ultralytics results without hiding what the model actually observed.
