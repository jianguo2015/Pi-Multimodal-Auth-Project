#!/usr/bin/env python
"""Thin wrapper: run a single face+voice verification from the command line.

The implementation lives in ``src/multimodal_auth/cli.py``; this file only
makes the repository runnable without installing the package::

    python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav

Add ``--backend pytorch`` to also see the attention-gate weights, ``--json``
for machine-readable output, ``--output out.json`` to save the result.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multimodal_auth.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
