#!/usr/bin/env python
"""Build MFCC voice features from raw audio (non-destructive).

Replacement for the frozen ``scripts_pc/01_process_audio.py``.

* reads ``data/raw/voice/<identity>/*.wav`` (also ``.flac``) read-only;
* writes ``outputs/features/voice/<identity>/<name>.npy`` of shape
  ``(1, 40, 400)`` - never into ``data/``, and refuses to overwrite an existing
  feature directory unless ``--force`` is given;
* ``--compare-to`` checks the output against an existing feature set
  (for example the original ``data/processed/voice_features``) and reports
  bit-equality per file.

Usage::

    python scripts/data/01_build_voice_features.py
    python scripts/data/01_build_voice_features.py --compare-to data/processed/voice_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import ROOT, build_features  # noqa: E402

from multimodal_auth.preprocessing import preprocess_voice  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(ROOT / "data" / "raw" / "voice"))
    parser.add_argument(
        "--compare-to",
        default=None,
        help="existing feature directory to check bit-equality against",
    )
    parser.add_argument("--force", action="store_true", help="allow overwriting existing features")
    args = parser.parse_args(argv)

    return build_features(
        modality="voice",
        section="voice",
        preprocess=preprocess_voice,
        suffixes={".wav", ".flac"},
        input_dir=args.input,
        force=args.force,
        compare_to=args.compare_to,
    )


if __name__ == "__main__":
    raise SystemExit(main())
