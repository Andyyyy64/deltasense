# DeltaSense release requirements

Local mirror of [GitHub issue #1](https://github.com/Andyyyy64/deltasense/issues/1),
read on 2026-10-02. The baseline's planning statements below are historical;
[local validation](validation.md) records current implementation/evidence. GitHub
issues/milestones were not modified. Public release remains deferred.

Status: requirements baseline for planning; no implementation or performance is claimed. GitHub issues and milestones are the source of truth for work status. Requirement IDs below identify obligations, not separate mandatory classes or modules.

## 1. Product goal and users

Help a Python developer compare what a selected computer vision model observes in images from different times, without rebuilding the comparison and result presentation for each application.

The primary user has suitable images and a suitable Ultralytics model. They may build tools for manufactured objects, buildings, plants, or people. DeltaSense must not contain rules specific to these domains or imply that a stock model recognizes every relevant feature.

The long-term goal is reliable visual change observation. The first release only compares predictions under stated capture assumptions. Model output differences, correspondence, measurement availability, and physical change are distinct concepts; see the terminology defined here: correspondence is a proposed association, prediction difference concerns model output, and physical change concerns the subject itself.

## 2. Milestone goals

| Milestone | User outcome | Required capability | Explicit boundary |
|---|---|---|---|
| v0.1 | Compare an already aligned before/after pair through a small Python API | Detection and segmentation comparison; honest missing/unknown results; JSON and a comparison image | Caller supplies comparable framing. No camera correction or automatic assurance that capture conditions match. |
| v0.2 | Compare a pair despite limited camera movement, or learn why alignment is inadequate | Automatic registration, measured geometric validity, common visible region, and rejection of unsupported spatial comparisons | A bounded, demonstrated 2D capture envelope. No universal 3D, illumination, occlusion, or physical-change guarantee. |

Do not introduce Recipe/DAG/plugin infrastructure, custom models, model training, LLM/VLM processing, a capture application, CLI/web UI, or three-or-more-timepoint analysis to deliver either milestone. Do not assign later version numbers to speculative features.

## 3. v0.1 user workflow

1. Obtain a detection or instance-segmentation model appropriate to the subject.
2. Supply two local images with the same framing, scale, and orientation, captured under sufficiently similar conditions. Prepare alignment outside DeltaSense if needed.
3. Select the model and inference/comparison settings, then compare the pair.
4. Inspect prediction counts, accepted correspondences, image-space measurements, and unavailable-result reasons.
5. Read the JSON or save the comparison image for an application or report.

The intended entry point remains `DeltaSense(model).compare(before, after)` with `result.to_json()` and `result.plot()`. Exact optional parameter names and return types must be specified in the API documentation before implementation of each interface. They must not introduce a general configuration framework.

## 4. v0.1 functional requirements

### Inputs and inference

| ID | Requirement | Acceptance evidence |
|---|---|---|
| V01-IN-01 | Accept exactly one before and one after local JPEG or PNG path, including `Path` values. Preserve temporal order. Reject directories, URLs, streams, and invalid or unreadable inputs explicitly. | Reversed pair preserves direction; unsupported and corrupt inputs produce actionable errors before a comparison is returned. |
| V01-IN-02 | Establish a single documented decoded image coordinate convention, including orientation handling. Require equal decoded dimensions for spatial comparison; do not silently resize, crop, or align one image to the other. | Orientation fixtures have consistent boxes/masks; unequal dimensions are explicitly rejected. Ordinary upstream inference resizing is mapped back to this coordinate convention. |
| V01-MOD-01 | Run the same caller-selected local Ultralytics detection or instance-segmentation checkpoint and the same effective inference settings on both images through public interfaces. Other tasks are rejected. | One real detection model and one real segmentation model run end to end; unsupported task and inference failure are distinct from empty predictions. |
| V01-MOD-02 | Allow the caller to configure model/device, detection filtering, and supported comparison thresholds. Preserve original class IDs/names/scores and effective settings. Do not switch model/backend on failure. | Caller settings affect both observations consistently; output records resolved settings and tested software/model identities. |

Model acquisition is a separate documented prerequisite for v0.1. A missing local checkpoint is an error, not an instruction to download a substitute. This keeps inference behavior explicit without adding a downloader.

### Comparison and measurements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| V01-CNT-01 | Return before/after counts and signed `after - before` deltas for the union of observed classes. Call these prediction counts. | Counts 2 to 3 produce +1; a missing class has an observed count of zero on that side; two empty outputs do not establish an unchanged physical scene. |
| V01-MAT-01 | Associate same-class detections one-to-one using a documented, deterministic spatial rule with configurable thresholds. Leave ambiguous cases unresolved rather than arbitrarily assigning identity. | No detection is reused; incompatible classes never match; tied/ambiguous candidates remain unresolved; reordering inputs preserves associations except irrelevant local IDs. |
| V01-MAT-02 | Return accepted pairs, unmatched-before, unmatched-after, and ambiguous evidence separately. Unmatched means no accepted correspondence, not confirmed disappearance or appearance. | A simulated missed detection is reported as unmatched; no physical removal claim is generated. |
| V01-BOX-01 | For accepted pairs, report center displacement and bbox area before/after, signed area delta, and relative area delta in the common image coordinates. | Center (20,30) to (30,50) gives dx=10 and dy=20 pixels; bbox area 100 to 144 gives +44 pixels squared and +0.44 relative delta. |
| V01-MSK-01 | For segmentation pairs, report mask area before/after, signed and relative area delta, plus a spatial difference visualization. Preserve the mapping between masks and detections. | Known masks with area 100 and 120 give +20 and +0.20; translating an equal-area mask gives zero area delta but a nonempty spatial difference. |
| V01-MSK-02 | Distinguish an unavailable mask from a valid empty observation. Do not substitute a bbox for a mask or invent a ratio when its baseline denominator is zero. | Unavailable ratios are null with a reason; malformed mask/instance mappings fail explicitly; empty detections remain valid observations. |

Coordinates use x rightward and y downward in the documented decoded image space. Relative area delta is `(after_area - before_area) / before_area`; values are ratios, not percentages. Spatial mask difference is evidence of different image support, not necessarily a shape change. Exact contour analysis and physical units are not required.

The library must not force every numeric difference into a binary changed/unchanged event. Raw deltas are required. If a thresholded measurement decision is exposed, it must state the metric, threshold, and assumptions; no universal scene-level change verdict is required in v0.1.

### Limits and invalid results

| ID | Requirement | Acceptance evidence |
|---|---|---|
| V01-LIM-01 | State that v0.1 does not verify alignment, illumination, or occlusion. Equal dimensions and successful matching must not be reported as machine-verified capture comparability. | Every result retains these unchecked conditions; JSON and plot cannot advertise verified physical change. |
| V01-LIM-02 | For each measurement, separate available values from unavailable values and include reasons. “Available” means calculable under the documented aligned-input assumption, not proven physical comparability. | Ambiguous matches have no invented displacement/size delta; zero is reserved for a calculated zero. |
| V01-LIM-03 | Distinguish invalid input/inference errors from a successful comparison with zero detections or partially unavailable measurements. Never emit NaN/Infinity or convert failure into an empty success. | Explicit checks cover all three cases and finite JSON output. |

Automated blur scoring, exposure equivalence, same-place identification, geometric quality scoring, and camera-motion detection are not v0.1 requirements. An unchecked condition is not a failed check and not a passed check. The original IDENTIFIABLE/UNOBSERVABLE/CONFOUNDED proposal must not become a misleading global v0.1 verdict: measurement availability and unchecked capture conditions supply the minimum necessary distinction.

### Outputs

| ID | Requirement | Acceptance evidence |
|---|---|---|
| V01-OUT-01 | Provide an inspectable Python result and JSON with observations, counts, correspondences, supported measurements, unavailable reasons, unchecked conditions, and minimal provenance. | JSON parsing preserves the Python result's meaning, units, and temporal direction; it contains no tensors or nonfinite numbers. |
| V01-OUT-02 | Provide a headless before/after plot with matched/unmatched/ambiguous annotations, segmentation differences where applicable, and visible limitations. Returning/rendering it must not open a GUI or write files without an explicit save action. | Visually inspect normal, empty, and unresolved examples; meanings are conveyed by text/labels as well as color; originals remain unchanged. |

JSON is a summary artifact: embedded input images and dense full-resolution mask arrays are not required. Minimal provenance records image identity (content digest and dimensions, not private absolute paths), checkpoint digest, DeltaSense/Ultralytics versions, and effective inference/comparison settings. This identifies a run; it does not promise bit-identical inference across devices.

## 5. v0.1 quality and delivery requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| V01-OPS-01 | Package the library with a verified Python/dependency support range. CPU is the required inference baseline; do not promise a GPU/OS matrix that has not been tested. | Build wheel/source distribution, install the wheel in a clean environment, and run the public API example. |
| V01-OPS-02 | Process local images locally. Do not add telemetry, upload inputs, bundle weights, mutate source images, or perform implicit network operations. | With dependencies and weights provisioned, run the smoke case with network access unavailable; inspect output files and source-image digests. |
| V01-OPS-03 | Provide bounded CI for deterministic comparison correctness and artifact installation. Keep explicit real-model validation separate from ordinary no-download tests. | CI records a passing build/install/check run; real-model results record their environment and model identity separately. |
| V01-VAL-01 | Verify comparison arithmetic and edge cases independently of upstream detection quality. | The acceptance cases in section 6 pass on deterministic observations; tolerances are documented for rasterization, not used to hide arithmetic errors. |
| V01-VAL-02 | Demonstrate the same API on a small, rights-cleared real dataset covering at least two scene types and both supported tasks. | Publish expected/actual observations, failure cases, model/settings, and input provenance; include unchanged recapture, visible change, and a nuisance/occlusion example. No domain-specific core branch is added. |
| V01-DOC-01 | Ship English installation, model preparation, quickstart, result/units/error reference, and limitations matching the built artifact. | Execute the quickstart from a clean install; documented examples/fields match actual output. |
| V01-REL-01 | Deliver v0.1 only after requirements, CI, real-model evidence, and documentation agree on one release candidate. | Link the exact commit and actual release/artifact read-back and clean installation evidence. Resolve owner decisions on license and publication destination before distribution; do not choose a license or presume PyPI publication. |

No arbitrary speed, accuracy, or universal micro-change claim is a release requirement. Record runtime/environment for the real examples. Missing evidence remains a release blocker for the affected claim; smaller claims must be documented explicitly rather than silently lowering a gate.

## 6. v0.1 acceptance scenarios

| Case | Given | Required outcome |
|---|---|---|
| A01 | Identical valid detections | Same counts, accepted unambiguous pairs, zero calculated deltas. |
| A02 | Class count 2 to 3 | +1 prediction-count delta; unmatched evidence without asserting a physical appearance. |
| A03 | A before detection is missed after | Unmatched-before; physical disappearance remains unproven. |
| A04 | One uniquely matched translated bbox | Exact signed center displacement in image pixels. |
| A05 | One uniquely matched resized bbox | Exact area and relative-area difference. |
| A06 | Multiple indistinguishable matching candidates | Ambiguity remains visible; no forced identity or associated spatial measurement. |
| A07 | Detections are returned in a different order | Equivalent correspondence and count results after ignoring local ordering IDs. |
| A08 | Both inference outputs are empty | Successful empty observations; no “scene unchanged” conclusion. |
| A09 | Mask grows from 100 to 120 pixels | +20 area delta and +0.20 relative delta. |
| A10 | Equal-area mask is translated | Zero area delta, nonzero spatial difference, no growth claim. |
| A11 | Mask/ratio unavailable or denominator zero | Null measurement with a reason; no bbox substitution or nonfinite value. |
| A12 | Images have unequal decoded dimensions | Actionable input failure; no implicit registration or rescaling. |
| A13 | Missing/corrupt image, missing weights, unsupported model task | Actionable failure distinguished from zero detections. |
| A14 | Same-size images differ in camera pose, lighting, or visibility | Capture conditions remain explicitly unchecked. The library may return prediction differences, but must not claim verified physical change or automatic rejection it cannot perform. |
| A15 | Before/after are reversed | Count and displacement signs reverse for the same uniquely matched observations; relative-area delta is recomputed against the new baseline. |
| A16 | JSON and plot of partial/empty results | Both preserve missing/ambiguous status and unchecked conditions. |
| A17 | Clean installed artifact, provisioned weights, network unavailable | Documented local inference and output workflow succeeds without input upload or hidden downloads. |

## 7. v0.2 goal and requirements

v0.2 adds a single capability to the validated v0.1 workflow: compensate for limited camera movement and determine whether the resulting spatial comparison is usable. It reuses the pair input, model integration, measurements, and outputs.

| ID | Requirement | Acceptance evidence |
|---|---|---|
| V02-REG-01 | Estimate and apply automatic 2D alignment using stable scene evidence, within a declared capture envelope. Record the transform and its direction. | Camera-only translation/rotation and bounded scale/perspective examples align within a predeclared tolerance. |
| V02-REG-02 | Assess registration from observable evidence such as correspondence support, spatial distribution, residual error, and transform validity. A returned matrix is not sufficient evidence of success. | Low-texture, repeated-pattern, unrelated-scene, and excessive-parallax cases reject or remain indeterminate as specified by the frozen evaluation protocol. |
| V02-VIS-01 | Identify the common visible image region and carry alignment through boxes/masks/plots consistently. Do not label crop boundaries as disappearance or compare full object size when its relevant extent is unobserved. | Partial field-of-view examples retain coverage/unavailable reasons and do not invent complete-object changes. |
| V02-LIM-01 | On alignment failure or insufficient evidence, withhold dependent spatial comparisons with a reason. Do not silently use identity alignment or fall back to a successful v0.1 spatial result. | Failed registration preserves raw observations as diagnostics but does not emit valid-looking movement/area/mask changes. |
| V02-LIM-02 | Separate verified geometric checks from unverified lighting, occlusion, and physical identity. Avoid registering away genuine object movement or deformation. | Camera motion plus a real localized change preserves the latter; unsupported cases are documented without universal physical-change claims. |
| V02-VAL-01 | Define a bounded reproducible evaluation protocol before selecting/finalizing algorithms and defaults, with separate development and held-out scenes. | Freeze tolerances and expected accept/reject outcomes before evaluation; show camera-only correction, retained genuine change, failure rejection, and v0.1 regression results on the release candidate. |
| V02-REL-01 | Publish matching English documentation and release evidence for the tested capture envelope and failure modes. | Verify the actual published artifact/installation and document exact support limits. |

“Limited camera movement” is deliberately not a promise about all handheld photographs. The v0.2 evaluation issue must turn it into a measured envelope before implementation defaults are frozen. Planar or low-parallax scenes are the starting candidate, not an already validated guarantee. Exact feature extractor, transformation model, optimizer, and thresholds are design decisions inside that issue, not prerequisites for v0.1.

## 8. Issue decomposition and source of truth

Use one issue for an independently reviewable behavior and its verification. Keep its tests with the behavior; do not create one issue per field/function/test. Split broad existing issues only when work has separate acceptance evidence: CI versus packaging, counts versus instance matching, bbox measurements versus matching, and plotting versus serialization.

Every implementation issue must reference requirement IDs, acceptance evidence, dependencies, current blockers, and its PR. GitHub holds the current status; this document holds the requirement definitions. Changing scope means updating the requirement and affected issues, not silently closing an unfinished task.

The v0.1 requirements issue tracks A01-A17 and the V01 requirements above. The v0.2 requirements issue tracks V02 requirements separately; it must not block v0.1. Release dates, assignees, license selection, and publication accounts are not invented by this plan.

## 9. Technical basis

Ultralytics documents public prediction calls and box/mask result access. These are suitable integration boundaries; DeltaSense does not need to recreate training or subclass upstream internals. Verify the exact tested versions during implementation. [Prediction documentation](https://docs.ultralytics.com/modes/predict/) and [Results reference](https://docs.ultralytics.com/reference/engine/results/).

OpenCV documents feature-based homography estimation and inlier/outlier evidence. This is a candidate building block for v0.2, not evidence that DeltaSense already handles arbitrary camera motion. [Feature matching and homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).


## Delivery traceability

| Work issue | Requirement IDs |
|---|---|
| #2 Package the library and verify clean installation | V01-OPS-01 |
| #3 Validate local image pairs and run the selected model | V01-IN-01, V01-IN-02, V01-MOD-01, V01-MOD-02, V01-OPS-02 |
| #4 Distinguish unchecked capture conditions and unavailable measurements | V01-LIM-01, V01-LIM-02, V01-LIM-03 |
| #5 Match detections one-to-one without forcing ambiguous identity | V01-MAT-01, V01-MAT-02 |
| #6 Measure matched mask areas and spatial differences | V01-MSK-01, V01-MSK-02 |
| #7 Expose the Python comparison result and JSON summary | V01-OUT-01 |
| #8 Validate v0.1 arithmetic and real image pairs | V01-VAL-01, V01-VAL-02 |
| #9 Document the v0.1 quickstart, result contract, and limits | V01-DOC-01 |
| #10 Verify and publish the v0.1 release | V01-REL-01 |
| #11 Run bounded correctness and package checks in CI | V01-OPS-03 |
| #12 Compare per-class prediction counts | V01-CNT-01 |
| #13 Measure displacement and bbox area for accepted matches | V01-BOX-01 |
| #14 Render a headless before/after comparison image | V01-OUT-02 |
