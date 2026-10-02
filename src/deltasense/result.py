"""Inspectable comparison summaries and headless, explicit-save visualization."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

UNCHECKED_CONDITIONS = ("alignment", "illumination", "occlusion", "physical_identity")


def _snapshot_observation(observation):
    # Copy ndarray objects as well as their data: even immutable buffers permit
    # changes to an array's shape/dtype metadata. Public access must be detached.
    return replace(observation, detections=tuple(replace(d) for d in observation.detections))


def _wrapped_lines(text, font, max_width):
    lines, line = [], ""
    for character in text:
        if line and font.getlength(line + character) > max_width:
            lines.append(line)
            line = ""
        line += character
    if line:
        lines.append(line)
    return lines


class ComparisonResult:
    """A snapshot. to_dict/properties return copies; plot returns a new PIL Image."""

    def __init__(self, before, after, comparison, provenance, images):
        self._before = _snapshot_observation(before)
        self._after = _snapshot_observation(after)
        self._comparison = deepcopy(comparison)
        self._provenance = deepcopy(provenance)
        self._images = tuple(np.array(image, copy=True) for image in images)

    @property
    def before(self):
        return _snapshot_observation(self._before)

    @property
    def after(self):
        return _snapshot_observation(self._after)

    @property
    def prediction_counts(self):
        return deepcopy(self._comparison["prediction_counts"])

    @property
    def matches(self):
        return deepcopy(self._comparison["matches"])

    @property
    def unmatched_before(self):
        return deepcopy(self._comparison["unmatched_before"])

    @property
    def unmatched_after(self):
        return deepcopy(self._comparison["unmatched_after"])

    @property
    def ambiguous_candidates(self):
        return deepcopy(self._comparison["ambiguous_candidates"])

    @property
    def unchecked_conditions(self):
        return list(UNCHECKED_CONDITIONS)

    def to_dict(self):
        return {
            "schema_version": "0.1.0",
            "before": self._before.to_dict(),
            "after": self._after.to_dict(),
            **deepcopy(self._comparison),
            "unchecked_conditions": self.unchecked_conditions,
            "measurement_assumption": "caller_aligned_images_with_similar_capture_conditions",
            "coordinate_convention": "decoded_exif_oriented_image; x_right_y_down",
            "physical_change_verified": False,
            "provenance": deepcopy(self._provenance),
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    def plot(self):
        """Return an RGB PIL Image. Never show a GUI or save implicitly.

        Cyan marks before-only mask support; orange marks after-only support.
        These are spatial prediction differences, not growth/disappearance claims.
        """
        width, height = self._before.identity.width, self._before.identity.height
        panel_width = max(width, 460)
        font = ImageFont.load_default(size=13)
        header, footer = 38, 150
        # Put full labels in a separate legend so overlapping or edge-touching
        # boxes never erase class/confidence/status labels over the photograph.
        legends = []
        for side, observation in enumerate((self._before, self._after)):
            key = "before_id" if side == 0 else "after_id"
            accepted = {m[key]: i for i, m in enumerate(self._comparison["matches"])}
            uncertain = {e[key] for e in self._comparison["ambiguous_candidates"]}
            entries = []
            for i, detection in enumerate(observation.detections):
                if i in accepted:
                    status, color = f"M{accepted[i]} matched", "#72f1b8"
                elif i in uncertain:
                    status, color = "? ambiguous", "#ffcf5c"
                else:
                    status, color = "U unmatched", "#ff9dba"
                name = detection.class_name.encode("ascii", "backslashreplace").decode()
                label = f"#{i} {status}: {name} (confidence {detection.confidence:.4f})"
                for line in _wrapped_lines(label, font, panel_width - 20):
                    entries.append((line, color))
            legends.append(entries or [("No detections (physical scene status unknown)", "white")])
        legend_height = 22 * max(map(len, legends)) + 16
        canvas = Image.new(
            "RGB", (2 * panel_width + 12, height + header + legend_height + footer), "#141821"
        )
        draw = ImageDraw.Draw(canvas)
        overlays = [image.copy() for image in self._images]
        for match in self._comparison["matches"]:
            a = self._before.detections[match["before_id"]].mask
            b = self._after.detections[match["after_id"]].mask
            if a is not None and b is not None:
                for index, support, color in [
                    (0, a & ~b, (0, 220, 255)),
                    (1, b & ~a, (255, 150, 0)),
                ]:
                    overlays[index][support] = (
                        overlays[index][support] * 0.35 + np.array(color) * 0.65
                    ).astype(np.uint8)
        for side, observation in enumerate((self._before, self._after)):
            offset = side * (panel_width + 12)
            canvas.paste(Image.fromarray(overlays[side]), (offset, header))
            draw.text(
                (offset + 10, 10), "BEFORE" if side == 0 else "AFTER", fill="white", font=font
            )
            key = "before_id" if side == 0 else "after_id"
            accepted = {m[key]: i for i, m in enumerate(self._comparison["matches"])}
            uncertain = {e[key] for e in self._comparison["ambiguous_candidates"]}
            for i, detection in enumerate(observation.detections):
                if i in accepted:
                    status, color = f"M{accepted[i]} matched", "#72f1b8"
                elif i in uncertain:
                    status, color = "? ambiguous", "#ffcf5c"
                else:
                    status, color = "U unmatched", "#ff9dba"
                x1, y1, x2, y2 = detection.bbox
                draw.rectangle(
                    (offset + x1, header + y1, offset + x2, header + y2), outline=color, width=2
                )
                # Backslash escapes keep unsupported glyphs legible in the bundled Latin font.
                label = f"#{i} {status.split()[0]}"
                label_x = offset + min(x1, max(0, panel_width - draw.textlength(label, font=font)))
                label_y = header + max(0, y1 - 16)
                bounds = draw.textbbox((label_x, label_y), label, font=font)
                draw.rectangle(bounds, fill="#141821")
                draw.text((label_x, label_y), label, fill=color, font=font)
            for row, (label, color) in enumerate(legends[side]):
                draw.text(
                    (offset + 10, height + header + 8 + 22 * row), label, fill=color, font=font
                )
        y = height + header + legend_height + 10
        lines = [
            f"Prediction comparison: {len(self.matches)} accepted pairs; "
            f"{len(self.unmatched_before)} unmatched before; "
            f"{len(self.unmatched_after)} unmatched after.",
            "M = proposed correspondence; U / ? = no accepted correspondence. "
            "Physical appearance/removal is unproven.",
            "Mask difference: cyan = before only; orange = after only. See JSON for measurements.",
            "UNCHECKED: alignment, illumination, occlusion, physical identity.",
            "Measurements assume caller-aligned images. Physical change is NOT verified.",
        ]
        for line in lines:
            draw.text((10, y), line, fill="white", font=font)
            y += 22
        return canvas
