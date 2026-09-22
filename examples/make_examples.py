"""Generate small, fully synthetic example inputs (no personal data).

The repository ships no biometric data. This script creates a placeholder face
image and a placeholder voice clip so that the pipeline can be exercised
end-to-end straight after cloning::

    python examples/make_examples.py
    python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav

Both files are deterministic for a given seed, so the resulting scores are
reproducible. They are *not* meaningful identity samples: the image is a
drawing and the audio is a harmonic tone. Any accept/reject value produced from
them says nothing about the model's accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import soundfile as sf  # noqa: E402

from multimodal_auth.preprocessing import write_image_bgr  # noqa: E402

FACE_SIZE = 512
DEFAULT_SEED = 20260419
DEFAULT_SECONDS = 2.0
SAMPLE_RATE = 16000


def make_face(seed: int = DEFAULT_SEED) -> np.ndarray:
    """Draw a very simple face-like picture (BGR uint8)."""
    rng = np.random.default_rng(seed)
    canvas = np.full((FACE_SIZE, FACE_SIZE, 3), 60, dtype=np.uint8)
    canvas[:, :] = np.clip(
        canvas.astype(np.int16) + rng.integers(-8, 8, canvas.shape, dtype=np.int16), 0, 255
    ).astype(np.uint8)

    centre = (FACE_SIZE // 2, FACE_SIZE // 2)
    cv2.ellipse(canvas, centre, (150, 190), 0, 0, 360, (190, 200, 205), -1)      # head
    cv2.ellipse(canvas, (centre[0] - 55, centre[1] - 45), (26, 16), 0, 0, 360, (235, 240, 245), -1)
    cv2.ellipse(canvas, (centre[0] + 55, centre[1] - 45), (26, 16), 0, 0, 360, (235, 240, 245), -1)
    cv2.circle(canvas, (centre[0] - 55, centre[1] - 45), 9, (35, 35, 40), -1)   # pupils
    cv2.circle(canvas, (centre[0] + 55, centre[1] - 45), 9, (35, 35, 40), -1)
    cv2.ellipse(canvas, (centre[0], centre[1] + 30), (22, 34), 0, 0, 360, (150, 160, 170), -1)
    cv2.ellipse(canvas, (centre[0], centre[1] + 95), (70, 34), 0, 0, 180, (90, 80, 85), 5)
    return cv2.GaussianBlur(canvas, (5, 5), 0)


def make_voice(seed: int = DEFAULT_SEED, seconds: float = DEFAULT_SECONDS) -> np.ndarray:
    """A deterministic harmonic tone with slow amplitude modulation."""
    rng = np.random.default_rng(seed + 1)
    t = np.linspace(0.0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    f0 = 115.0 + 12.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    signal = sum(np.sin(k * phase) / (k ** 1.3) for k in range(1, 12))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.4 * t - 1.2)
    signal = signal * envelope + 0.01 * rng.standard_normal(t.size)
    signal = signal / (np.abs(signal).max() + 1e-9) * 0.85
    return signal.astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(ROOT / "examples"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    face_path = out_dir / "sample_face.jpg"
    voice_path = out_dir / "sample_voice.wav"

    for path in (face_path, voice_path):
        if path.exists() and not args.force:
            print("exists, keeping (use --force to regenerate): %s" % path)
            continue
        if path.suffix == ".jpg":
            write_image_bgr(face_path, make_face(args.seed))
            print("wrote %s" % face_path)
        else:
            sf.write(str(voice_path), make_voice(args.seed, args.seconds), SAMPLE_RATE)
            print("wrote %s" % voice_path)

    print("\nThese are synthetic placeholders, not identity samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
