"""Write-safety primitives.

Phase 0 ended with a real incident: three ``.pth`` files were overwritten
because a training script wrote back into its own input directory. Phase 1
fixes the *class* of bug, not just the instance:

* every script that produces artefacts must go through
  :func:`prepare_output_dir`, which only ever creates directories inside
  ``<project>/outputs``;
* it fails closed: if the target directory already contains data and the
  caller did not explicitly pass ``--force``, a :class:`SafetyError` is raised
  before anything is written;
* :func:`assert_not_protected` refuses any path under ``weights/``, ``data/``,
  ``configs/``, ``src/``, ``tests/``, ``app_pi/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Union

from .errors import SafetyError

#: Directory names (relative to the project root) that scripts must never write to.
PROTECTED_DIRS = (
    "weights",
    "data",
    "configs",
    "src",
    "tests",
    "docs",
    "app_pi",
    "examples",
)

#: Only this directory may receive generated artefacts.
OUTPUT_DIRNAME = "outputs"

PathLike = Union[str, Path]


def is_within(path: PathLike, parent: PathLike) -> bool:
    """True when ``path`` is ``parent`` itself or lives underneath it."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def assert_not_protected(target: PathLike, project_root: PathLike) -> Path:
    """Refuse to write to any protected directory of the repository."""
    resolved = Path(target).resolve()
    root = Path(project_root).resolve()
    for name in PROTECTED_DIRS:
        protected = root / name
        if is_within(resolved, protected):
            raise SafetyError(
                "refusing to write to %s: %s is a protected directory of this repository"
                % (resolved, name)
            )
    return resolved


def prepare_output_dir(
    project_root: PathLike,
    subdir: str,
    run_name: str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Create and return ``<project_root>/outputs/<subdir>/<run_name>``.

    Fails closed when the directory already holds files and ``force`` is not
    set, so a re-run can never silently clobber earlier artefacts.
    """
    root = Path(project_root).resolve()
    if Path(subdir).is_absolute() or ".." in Path(subdir).parts:
        raise SafetyError("output subdir must be a relative path, got %r" % subdir)
    if run_name is not None and (Path(run_name).is_absolute() or ".." in Path(run_name).parts):
        raise SafetyError("run name must be a simple relative name, got %r" % run_name)

    target = (root / OUTPUT_DIRNAME / subdir)
    if run_name is not None:
        target = target / run_name
    target = target.resolve()

    output_root = (root / OUTPUT_DIRNAME).resolve()
    if not is_within(target, output_root):
        raise SafetyError("refusing to write outside %s: %s" % (output_root, target))
    assert_not_protected(target, root)

    if target.exists() and not force:
        existing = [p for p in target.rglob("*") if p.is_file()]
        if existing:
            raise SafetyError(
                "output directory %s already contains %d file(s); "
                "pass --force to overwrite, or choose another --run-name"
                % (target, len(existing))
            )
    target.mkdir(parents=True, exist_ok=True)
    return target


def portable_path(value: PathLike, root: PathLike) -> str:
    """Render a path so reports never expose a machine-specific location.

    Absolute Windows paths contain the local user name. The JSON produced by
    ``scripts/infer.py`` is meant to be copy-pasteable (an issue, a README, CI
    output), so it must not leak one. Paths inside ``root`` become POSIX-relative
    (``weights/onnx/face_extractor_quant.onnx``); anything outside keeps only the
    file name (``<outside the project>/me.jpg``).
    """
    resolved = Path(value).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return "<outside the project>/%s" % resolved.name


def file_digest(path: PathLike, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Hex digest of a file (default SHA-256), streamed so big files are fine."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digests(paths: Iterable[PathLike]) -> Dict[str, str]:
    """Map ``absolute path -> sha256`` for a set of files (used by tests)."""
    return {str(Path(p).resolve()): file_digest(p) for p in paths}
