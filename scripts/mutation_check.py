"""Bounded manual mutation audit in isolated copies; never alters source files."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MUTANTS = [
    (
        "count_sign",
        "comparison.py",
        '"delta": counts_b[cls] - counts_a[cls]',
        '"delta": counts_a[cls] - counts_b[cls]',
    ),
    (
        "strict_iou_boundary",
        "comparison.py",
        "if iou >= settings.match_iou:",
        "if iou > settings.match_iou:",
    ),
    ("cross_class_matching", "comparison.py", "if a.class_id != b.class_id:", "if False:"),
    (
        "forced_ambiguity",
        "comparison.py",
        "if len(by_before[i]) != 1 or len(by_after[j]) != 1:",
        "if False:",
    ),
    ("wrong_mask_delta", "comparison.py", '"delta": after - before,', '"delta": before - after,'),
    (
        "wrong_ratio_denominator",
        "comparison.py",
        "(after - before) / before if before else None",
        "(after - before) / after if after else None",
    ),
    (
        "wrong_displacement_sign",
        "comparison.py",
        "(bx1 + bx2 - ax1 - ax2) / 2",
        "(ax1 + ax2 - bx1 - bx2) / 2",
    ),
    ("mask_xor_as_union", "comparison.py", "np.count_nonzero(a ^ b)", "np.count_nonzero(a | b)"),
    (
        "physical_verdict",
        "result.py",
        '"physical_change_verified": False',
        '"physical_change_verified": True',
    ),
    ("silent_mask_binarization", "observations.py", "not np.isin(mask, [0, 1]).all()", "False"),
    (
        "allow_16bit_color",
        "observations.py",
        'if contents[24:25] == b"\\x10":',
        "if False:",
    ),
    (
        "mutable_mask_buffer",
        "observations.py",
        "dtype=bool).reshape(",
        "dtype=bool).copy().reshape(",
    ),
    (
        "before_snapshot_leak",
        "result.py",
        "return _snapshot_observation(self._before)",
        "return self._before",
    ),
    (
        "after_snapshot_leak",
        "result.py",
        "return _snapshot_observation(self._after)",
        "return self._after",
    ),
    (
        "threshold_overflow",
        "engine.py",
        "if not 0 <= value <= 1:",
        'if not __import__("math").isfinite(value) or not 0 <= value <= 1:',
    ),
    (
        "unclassified_checkpoint_stat_error",
        "engine.py",
        "except (OSError, ValueError, RuntimeError) as exc:",
        "except (ValueError, RuntimeError) as exc:",
    ),
    (
        "infinite_relative_ratio",
        "comparison.py",
        "if relative is not None and not math.isfinite(relative):",
        "if False:",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="baseline-", dir=args.output))
    shutil.copytree(
        root / "src/deltasense",
        snapshot / "deltasense",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    source_sha256 = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (snapshot / "deltasense").glob("*.py")
    }
    baseline_env = {
        **os.environ,
        "PYTHONPATH": str(snapshot.resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=root,
        env=baseline_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (args.output / "baseline.txt").write_text(baseline.stdout + baseline.stderr)
    if baseline.returncode:
        raise RuntimeError("unmodified baseline must pass before mutation detection is meaningful")
    results = []
    for name, filename, old, new in MUTANTS:
        directory = Path(tempfile.mkdtemp(prefix=name + "-", dir=args.output))
        shutil.copytree(snapshot / "deltasense", directory / "deltasense")
        file = directory / "deltasense" / filename
        text = file.read_text()
        if text.count(old) != 1:
            raise RuntimeError(f"mutant {name} no longer targets exactly one source expression")
        file.write_text(text.replace(old, new))
        env = {**os.environ, "PYTHONPATH": str(directory.resolve()), "PYTHONDONTWRITEBYTECODE": "1"}
        probe = subprocess.check_output(
            [sys.executable, "-c", "import deltasense; print(deltasense.__file__)"],
            env=env,
            text=True,
        )
        assert str(directory.resolve()) in probe
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short", "-x"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        (directory / "pytest.txt").write_text(run.stdout + run.stderr)
        entry = {
            "name": name,
            "exit_code": run.returncode,
            "detected": run.returncode == 1,
            "evidence_directory": directory.name,
        }
        results.append(entry)
        print(f"{name}: {'detected' if entry['detected'] else 'NOT DETECTED'}", flush=True)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "baseline_exit_code": baseline.returncode,
                "source_sha256": source_sha256,
                "mutants": results,
            },
            indent=2,
        )
        + "\n"
    )
    assert all(r["detected"] for r in results), (
        "a mutant survived or failed for an unrelated reason"
    )


if __name__ == "__main__":
    main()
