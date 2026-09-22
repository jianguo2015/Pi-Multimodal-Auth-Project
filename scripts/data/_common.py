#!/usr/bin/env python
"""Shared helpers for the feature-building scripts.

Safety contract:

* the raw input directory is opened **read-only** (``data/raw/...``);
* features are written to ``outputs/features/<modality>/`` through
  :func:`multimodal_auth.safety.prepare_output_dir`, which refuses to write
  anywhere else and fails closed if the directory already holds files and
  ``--force`` was not given. The original scripts wrote into ``data/``;
* ``--compare-to`` re-reads an existing feature directory (for example the
  ``data/processed`` folder produced by the original pipeline) and reports
  bit-equality per file. That is how the Phase 1 refactor was validated.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multimodal_auth.config import load_config  # noqa: E402
from multimodal_auth.errors import InvalidInputError  # noqa: E402
from multimodal_auth.safety import prepare_output_dir  # noqa: E402


def collect_inputs(input_dir: Path, suffixes) -> Dict[str, List[Path]]:
    """Group input files by their immediate subdirectory (one identity each)."""
    if not input_dir.is_dir():
        raise FileNotFoundError("input directory not found: %s" % input_dir)
    grouped: Dict[str, List[Path]] = {}
    for directory in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        files = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes
        )
        if files:
            grouped[directory.name] = files
    if not grouped:
        raise FileNotFoundError(
            "no input files with suffix %s under %s" % (tuple(sorted(suffixes)), input_dir)
        )
    return grouped


def compare_directory(out_dir: Path, reference_dir: Path) -> dict:
    """Report bit-equality between freshly written features and a reference set."""
    equal = 0
    differing = []
    missing = []
    total = 0
    worst = 0.0
    for produced in sorted(out_dir.rglob("*.npy")):
        relative = produced.relative_to(out_dir)
        reference = reference_dir / relative
        total += 1
        if not reference.is_file():
            missing.append(str(relative))
            continue
        fresh = np.load(produced)
        stored = np.load(reference)
        if fresh.shape != stored.shape:
            differing.append("%s (shape %s vs %s)" % (relative, fresh.shape, stored.shape))
            continue
        if np.array_equal(fresh, stored):
            equal += 1
        else:
            worst = max(worst, float(np.abs(fresh.astype(np.float64) - stored.astype(np.float64)).max()))
            differing.append("%s (max abs diff %.3g)" % (relative, worst))
    return {
        "reference_dir": str(reference_dir),
        "compared": total,
        "bit_identical": equal,
        "differing": differing[:20],
        "missing_in_reference": missing[:20],
        "max_abs_difference": worst,
        "all_bit_identical": equal == total and not missing,
    }


def build_features(
    modality: str,
    section: str,
    preprocess: Callable,
    suffixes,
    input_dir: Path | None = None,
    force: bool = False,
    compare_to: Path | None = None,
) -> int:
    config = load_config(root=ROOT)
    sub_config = getattr(config, section)
    input_dir = Path(input_dir) if input_dir else ROOT / "data" / "raw" / modality
    grouped = collect_inputs(input_dir, suffixes)
    out_dir = prepare_output_dir(ROOT, "features/%s" % modality, None, force=force)

    metadata = {
        "modality": modality,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "config": str(config.source),
        "identities": {},
        "failures": [],
    }

    total = 0
    for label, files in grouped.items():
        target_dir = out_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for path in files:
            try:
                feature = preprocess(path, sub_config)
            except InvalidInputError as exc:
                print("  !! skipped %s: %s" % (path.name, exc))
                metadata["failures"].append({"file": str(path), "error": str(exc)})
                continue
            np.save(target_dir / (path.stem + ".npy"), feature)
            written += 1
            total += 1
            if written % 10 == 0 or written == len(files):
                print("  %s -> %d/%d" % (label, written, len(files)))
        metadata["identities"][label] = {"inputs": len(files), "written": written}

    if compare_to is not None:
        metadata["comparison"] = compare_directory(out_dir, Path(compare_to))

    (out_dir / "features.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[+] %d features written to %s" % (total, out_dir))
    if metadata["failures"]:
        print("[!] %d input(s) failed and were skipped" % len(metadata["failures"]))
    if "comparison" in metadata:
        comparison = metadata["comparison"]
        print(
            "[=] vs %s -> %d/%d bit-identical (max abs diff %.3g)"
            % (
                comparison["reference_dir"],
                comparison["bit_identical"],
                comparison["compared"],
                comparison["max_abs_difference"],
            )
        )
        if not comparison["all_bit_identical"]:
            print("    differences: %s" % comparison["differing"])
            return 1
    return 0
