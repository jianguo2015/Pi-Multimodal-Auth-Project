"""Face preprocessing: image file -> (3, S, S) float32 tensor.

The numerical pipeline is copied verbatim from the audited original
``utils/vision_ops.py::extract_face`` (OpenCV decode, Haar detection, largest
face, resize, BGR->RGB, CHW transpose, ``(x - 127.5) / 128.0``). Only three
things changed, none of which alters the numbers for readable inputs:

1. parameters come from config instead of literals;
2. failures raise :class:`InvalidImageError` instead of returning ``None``;
3. decoding falls back to :func:`cv2.imdecode` when :func:`cv2.imread` cannot
   handle the path (e.g. non-ASCII characters on Windows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config.loader import FacePreprocessConfig
from ..errors import InvalidImageError

#: Detection box as (x, y, w, h).
Box = Tuple[int, int, int, int]


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image as a BGR ``uint8`` array.

    ``cv2.imread`` is tried first so the behaviour matches the audited original
    exactly. For paths containing non-ASCII characters it is skipped, because
    OpenCV's Windows file API cannot open them (it fails with a warning and
    returns ``None``). In both cases the fallback reads the raw bytes and uses
    :func:`cv2.imdecode`, which is decoder-identical to ``imread``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise InvalidImageError("image file not found: %s" % file_path)

    image = None
    if _is_ascii_path(file_path):
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

    if image is None:
        try:
            raw = file_path.read_bytes()  # Python I/O handles non-ASCII paths
        except OSError as exc:
            raise InvalidImageError("cannot read image %s: %s" % (file_path, exc))
        if not raw:
            raise InvalidImageError("image file is empty: %s" % file_path)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise InvalidImageError(
            "cannot decode image (unsupported or corrupt): %s" % file_path
        )
    return image


def write_image_bgr(path: str | Path, image: np.ndarray, quality: int = 95) -> Path:
    """Write a BGR image to disk, non-ASCII paths included.

    ``cv2.imwrite`` cannot open paths containing non-ASCII characters on
    Windows, so the encode + ``write_bytes`` fallback is used there.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix.lower() or ".jpg"
    if extension in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    else:
        params = []

    if _is_ascii_path(target) and cv2.imwrite(str(target), image, params):
        return target

    ok, buffer = cv2.imencode(extension, image, params)
    if not ok:
        raise InvalidImageError("cannot encode image for %s" % target)
    target.write_bytes(buffer.tobytes())
    return target


def build_cascade(config: FacePreprocessConfig) -> cv2.CascadeClassifier:
    """Load the Haar cascade specified by the configuration."""
    cascade_path = config.cascade
    if not Path(cascade_path).is_absolute():
        cascade_path = str(Path(cv2.data.haarcascades) / cascade_path)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise InvalidImageError("Haar cascade not available: %s" % cascade_path)
    return cascade


def detect_faces(
    gray: np.ndarray,
    config: FacePreprocessConfig,
    cascade: Optional[cv2.CascadeClassifier] = None,
) -> Tuple[Box, ...]:
    """Run Haar detection with the configured parameters."""
    cascade = cascade if cascade is not None else build_cascade(config)
    faces = cascade.detectMultiScale(gray, config.scale_factor, config.min_neighbors)
    return tuple(tuple(int(v) for v in box) for box in faces)


def _crop_largest_face(image: np.ndarray, faces: Sequence[Box]) -> np.ndarray:
    # identical ordering rule as the original: largest area wins
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return image[y:y + h, x:x + w]


def _center_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    min_dim = min(height, width)
    top = (height - min_dim) // 2
    left = (width - min_dim) // 2
    return image[top:top + min_dim, left:left + min_dim]


def preprocess_face(
    source: str | Path | np.ndarray,
    config: FacePreprocessConfig | None = None,
    *,
    cascade: Optional[cv2.CascadeClassifier] = None,
) -> np.ndarray:
    """Convert an image into the tensor expected by ``FaceExtractor``.

    Parameters
    ----------
    source:
        Path to an image file, or an already decoded BGR ``uint8`` array.
    config:
        Preprocessing parameters; defaults to the frozen audit values.

    Returns
    -------
    ``numpy.ndarray`` of shape ``(3, image_size, image_size)`` and dtype
    ``float32``.
    """
    config = config or FacePreprocessConfig()
    target_size = (config.image_size, config.image_size)

    image = read_image_bgr(source) if isinstance(source, (str, Path)) else np.asarray(source)
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(
            "expected a 3-channel BGR image, got shape %s" % (image.shape,)
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, config, cascade)

    if len(faces) > 0:
        face_image = _crop_largest_face(image, faces)
    elif config.center_crop_fallback:
        face_image = _center_crop(image)
    else:
        raise InvalidImageError("no face detected and the centre-crop fallback is disabled")

    face_image = cv2.resize(face_image, target_size)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = np.transpose(face_image, (2, 0, 1))
    tensor = (face_image.astype(np.float32) - config.mean) / config.scale
    return np.ascontiguousarray(tensor, dtype=np.float32)
