# Changelog

## 0.1.0 — 2026-10-03

First implemented version of the Python API for comparing a caller-aligned pair
of local JPEG/PNG images with a trusted local Ultralytics detection or
instance-segmentation checkpoint.

- `DeltaSense(model).compare(before, after)` returns prediction counts,
  conservative same-class spatial correspondences, unmatched/ambiguous evidence,
  bbox-center displacement and bbox/mask area differences.
- Missing measurements, zero denominators and unrepresentable relative ratios
  retain explicit reasons. Results preserve unchecked capture conditions and do
  not claim verified physical change.
- Detached result snapshots, finite JSON summaries and headless comparison
  plots support application/report integration.
- Image orientation/color handling, corrupt/unsupported input rejection,
  local checkpoint identity checks and explicit offline inference are covered
  by deterministic and real-model verification.
- Packaging includes wheel/sdist consistency checks and a bounded CI workflow.
  The recorded local suite has 279 passing tests, 100% core statement/branch
  coverage, 17 selected detected mutants, and independent installed-wheel checks.
- Reproducible real-image and human-annotated MOT17 evaluations document
  prediction quality, failure cases and exploratory settings.

The verified runtime baseline is CPython 3.12 on Linux x86-64 with CPU inference.
Automatic alignment, physical units, verified physical identity/change, and
general model-accuracy guarantees are outside this version's scope.

This entry identifies the source implementation. Package registry/GitHub Release
publication and license selection remain deferred. See
[local validation](docs/validation.md), [the API contract](docs/api.md), and
[CI results](https://github.com/Andyyyy64/deltasense/actions/workflows/ci.yml).
