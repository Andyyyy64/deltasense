# Project concept

Most computer vision libraries analyze one image at a time. DeltaSense is about the gap between two observations.

## Project goal

Compare corresponding regions of the same subject at different times, accounting
for differences in how the images were captured. Explain meaningful changes in
the subject, including what changed, by how much, and the supporting evidence.
Distinguish measurements from interpretations, and explain why a conclusion
cannot be reached when the evidence is insufficient.

Users want to know what changed, by how much, and what supports that conclusion.
Matching subjects, checking whether capture conditions allow comparison,
measuring changes, and explaining their meaning all serve this goal.
When a change cannot be quantified, explain why instead of inventing a value.

### Examples of measurements and interpretations

These are intended use cases, not features already implemented in v0.1.

| User question | What to measure or observe in the images | Evidence needed for interpretation and its limits |
| --- | --- | --- |
| Did the person move forward along the same road? | The position difference for the corresponding person | Establish that it is the same person, compare positions relative to the road, and account for camera position changes. Define whether "forward" means along the road, in the direction the person faces, or toward the camera. |
| Has my hair become thinner over the past year? | Differences in the appearance of hair and scalp in corresponding head regions | Account for lighting, angle, and hairstyle. More visible scalp alone does not establish hair loss. |
| Can facial photos tell me whether I gained weight? | Differences in facial contours and cheek appearance in corresponding regions | Account for distance, angle, and expression. Facial appearance alone cannot confirm weight gain. |

A difference in a person's image coordinates is a measurement. Explaining that
the person moved forward along the road requires comparison relative to the road
and evidence supporting that interpretation. Two photos show positions at the
capture times; they do not establish the intervening path or continuous walking.

### Criteria for progress

- Compare corresponding regions of the same subject and state the comparison
  reference, direction, and units.
- Avoid claiming a subject changed when only capture conditions changed.
- Describe actual subject changes with their magnitude and supporting evidence.
- State what cannot be determined and why when evidence is insufficient.

Evaluate missed changes, false alarms, and correspondence accuracy using labeled
before/after pairs for each use case. Also measure how often the system withholds
a conclusion; returning "cannot determine" for every pair does not meet the goal.
Heads and faces are representative examples. The goal does not restrict the
product to a particular body region.

## Relationship to current versions

v0.1 provides a foundation for comparing model observations and reporting
measurements and limits. It does not automatically verify capture conditions or
interpret results as forward movement, hair loss, or weight gain.

The bounded camera alignment and geometric validity checks planned for v0.2 are
a step toward more comparable observations. They do not resolve every lighting,
pose, or depth difference, or every use case's interpretation requirements.
The goal describes the project's direction. It does not redefine the existing
v0.1 completion criteria or imply that future capabilities are already implemented.

## Initial implementation approach

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
