"""Make a standalone scientific comparison figure from saved run reports."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads((r / "report.json").read_text()) for r in args.runs]
    attribution = json.loads(args.attribution.read_text())
    if len({r["manifest_sha256"] for r in reports}) != 1:
        raise ValueError("cannot compare runs on different datasets")
    labels = [f"{r['settings']['imgsz']} / conf {r['settings']['conf']:.2f}" for r in reports]
    first = reports[0]["models"]["detect"]["scores_by_localization_iou"]["0.5"]["summary"]
    detection_denominator = (
        first["detection_on_first_observation_of_each_unique_frame"]["tp"]
        + first["detection_on_first_observation_of_each_unique_frame"]["fn"]
    )
    correspondence_denominator = first["correspondence"]["tp"] + first["correspondence"]["fn"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(3)
    for offset, task, color in [(-0.18, "detect", "#3268a8"), (0.18, "segment", "#d98325")]:
        summaries = [
            r["models"][task]["scores_by_localization_iou"]["0.5"]["summary"] for r in reports
        ]
        for ax, values in [
            (
                axes[0],
                [
                    s["detection_on_first_observation_of_each_unique_frame"]["recall"]
                    for s in summaries
                ],
            ),
            (axes[1], [s["correspondence"]["recall"] for s in summaries]),
        ]:
            bars = ax.bar(x + offset, np.array(values) * 100, width=0.34, label=task, color=color)
            ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in values], fontsize=9, padding=3)
            ax.set_xticks(x, labels, rotation=18, ha="right")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Recall (%)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
    axes[0].set_title(f"Person detection: {detection_denominator:,} annotated boxes")
    axes[1].set_title(f"Identity correspondence: {correspondence_denominator:,} opportunities")
    axes[0].legend(frameon=False)
    baseline = attribution["runs"][args.runs[0].name]["models"]
    bottom = np.zeros(2)
    for key, label, color in [
        ("correct", "Correct", "#4f8d64"),
        ("recognition_or_localization_missing", "Missing / poorly localized", "#aeb8c5"),
        ("competing_candidates", "Competing candidates", "#d98325"),
        ("accepted_different_identity", "Wrong identity", "#ac4d55"),
        ("no_accepted_candidate", "No accepted candidate", "#78619d"),
    ]:
        values = (
            np.array([baseline[t]["attribution"].get(key, 0) for t in ("detect", "segment")])
            / correspondence_denominator
            * 100
        )
        axes[2].bar([0, 1], values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[2].set_xticks([0, 1], ["detect", "segment"])
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Baseline correspondence outcomes")
    axes[2].set_ylabel("Share of expected identities (%)")
    axes[2].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("DeltaSense local MOT17 subset — published human annotations", fontsize=14)
    fig.text(
        0.5,
        0.01,
        f"{first['unique_frame_count']} images / {first['pair_count']} pairs. "
        "Correlated fixed-camera training samples; exploratory settings, not held-out accuracy.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.24, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
