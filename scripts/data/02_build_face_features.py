#!/usr/bin/env python
"""Build face features from raw images (non-destructive).

Replacement for the frozen ``scripts_pc/02_process_face.py``.

* reads ``data/raw/face/<identity>/*.jpg`` (also ``.jpeg``/``.png``) read-only;
* writes ``outputs/features/face/<identity>/<name>.npy`` of shape
  ``(3, 112, 112)`` - never into ``data/``, and refuses to overwrite an existing
  feature directory unless ``--force`` is given;
* ``--compare-to`` checks the output against an existing feature set (for
  example the original ``data/processed/face_features``) and reports
  bit-equality per file.

Usage::

    python scripts/data/02_build_face_features.py
    python scripts/data/02_build_face_features.py --compare-to data/processed/face_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import ROOT, build_features  # noqa: E402

from multimodal_auth.preprocessing import preprocess_face  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(ROOT / "data" / "raw" / "face"))
    parser.add_argument(
        "--compare-to",
        default=None,
        help="existing feature directory to check bit-equality against",
    )
    parser.add_argument("--force", action="store_true", help="allow overwriting existing features")
    args = parser.parse_args(argv)

    return build_features(
        modality="face",
        section="face",
        preprocess=preprocess_face,
        suffixes={".jpg", ".jpeg", ".png"},
        input_dir=args.input,
        force=args.force,
        compare_to=args.compare_to,
    )


if __name__ == "__main__":
    raise SystemExit(main())
