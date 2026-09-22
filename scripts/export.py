#!/usr/bin/env python
"""Export the PyTorch checkpoints to ONNX fp32 and dynamic-INT8 (research only).

Port of the frozen ``scripts_pc/05_export_and_quantize.py`` with two fixes:

* **outputs are sandboxed** - the graphs are written to
  ``outputs/exports/<run>/`` only; the shipped files in ``weights/onnx/`` are
  never overwritten (that is what docs/MODEL_ARTIFACT_MANIFEST.md protects);
* **no environment leakage** - the original repointed ``TMP``/``TEMP`` to a
  temp directory and left them pointing there for the rest of the process;
  here they are restored in a ``finally`` block.

After exporting, every new graph is loaded and compared against the PyTorch
model (fp32) and against the fp32 graph (INT8), so the quantisation error is
measured rather than assumed.

Usage::

    python scripts/export.py --run-name local-export
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multimodal_auth.config import load_config  # noqa: E402
from multimodal_auth.fusion import ModalAttentionFusion  # noqa: E402
from multimodal_auth.models import FaceExtractor, VoiceExtractor  # noqa: E402
from multimodal_auth.safety import file_digest, prepare_output_dir  # noqa: E402


def load_models(config):
    """Rebuild the three networks and load the shipped checkpoints strictly."""
    voice = VoiceExtractor(n_mfcc=config.voice.n_mfcc, embed_dim=config.models["voice"].embed_dim)
    face = FaceExtractor(embed_dim=config.models["face"].embed_dim)
    fusion = ModalAttentionFusion(
        voice_dim=config.models["voice"].embed_dim,
        face_dim=config.models["face"].embed_dim,
        hidden_dim=config.models["fusion"].hidden_dim or 64,
    )
    for role, model in (("voice", voice), ("face", face), ("fusion", fusion)):
        model.load_state_dict(torch.load(str(config.pytorch_path(role)), map_location="cpu"))
        model.eval()
    return voice, face, fusion


def dummy_inputs(config):
    size = config.face.image_size
    return {
        "voice": torch.randn(1, 1, config.voice.n_mfcc, config.voice.max_pad_len),
        "face": torch.randn(1, 3, size, size),
        "fusion": (
            torch.randn(1, config.models["voice"].embed_dim),
            torch.randn(1, config.models["face"].embed_dim),
        ),
    }


def export_fp32(model, inputs, out_path: Path, names, dynamic_axes) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(out_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def ascii_scratch_dir(preferred: Path, run_name: str) -> Path:
    """Return a writable, ASCII-only scratch directory.

    ``onnxruntime.quantization`` pre-processing calls
    ``onnx.shape_inference.infer_shapes_path``, which uses the narrow-character
    (ANSI) file API on Windows. It therefore cannot write into a path that
    contains non-ASCII characters - and the development checkout of this
    project lives under such a path. When that happens the intermediate file is
    simply never created and the library fails with a confusing
    ``FileNotFoundError``.

    The original ``scripts_pc/05_export_and_quantize.py`` worked around this by
    staging everything in a drive-root directory. The same idea is used here,
    but with an explicit writability probe and a clear error message.
    """
    candidates = [
        preferred / ("ma_quant_%s" % run_name),
        Path((preferred.drive or "C:") + os.sep) / "multimodal_auth_export_tmp" / run_name,
        Path(tempfile.gettempdir()) / ("ma_quant_%s" % run_name),
    ]
    for candidate in candidates:
        try:
            str(candidate).encode("ascii")
        except UnicodeEncodeError:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError:
            continue
        if candidate.resolve() != (preferred / ("ma_quant_%s" % run_name)).resolve():
            print(
                "[!] non-ASCII path detected; staging ONNX quantisation in %s\n"
                "    (onnxruntime cannot write its intermediates next to a "
                "non-ASCII path)" % candidate
            )
        return candidate
    raise SystemExit(
        "no writable ASCII-only scratch directory found. Cloning this repository "
        "into an ASCII-only path (e.g. <ascii-only-root>\\multimodal-auth) fixes it, or run "
        "with --skip-quantise to export fp32 graphs only."
    )


def quantise(fp32_path: Path, int8_path: Path, scratch: Path, stem: str) -> None:
    """Dynamic INT8 quantisation inside an ASCII-only scratch directory."""
    from onnxruntime.quantization import QuantType, quantize_dynamic, shape_inference

    scratch.mkdir(parents=True, exist_ok=True)
    staged = scratch / ("%s.onnx" % stem)
    prepared = scratch / ("%s_prep.onnx" % stem)
    output = scratch / ("%s_quant.onnx" % stem)
    shutil.copyfile(fp32_path, staged)

    previous_env = {key: os.environ.get(key) for key in ("TMP", "TEMP")}
    previous_tempdir = tempfile.tempdir
    os.environ["TMP"] = str(scratch)
    os.environ["TEMP"] = str(scratch)
    # tempfile caches the directory, so setting the environment is not enough
    tempfile.tempdir = str(scratch)
    try:
        shape_inference.quant_pre_process(str(staged), str(prepared), skip_symbolic_shape=False)
        quantize_dynamic(str(prepared), str(output), weight_type=QuantType.QInt8)
        shutil.copyfile(output, int8_path)
    finally:
        tempfile.tempdir = previous_tempdir
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def session_for(path: Path, config):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = config.runtime.intra_op_num_threads
    return ort.InferenceSession(str(path), sess_options=options, providers=[config.runtime.provider])


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).max())


def _feed(role: str, inputs) -> dict:
    """Build the ONNX feed dict with the exact input names used by the export."""
    if role == "face":
        return {"face_input": inputs.detach().numpy().astype(np.float32)}
    if role == "voice":
        return {"voice_input": inputs.detach().numpy().astype(np.float32)}
    voice, face = inputs
    return {
        "voice_input": voice.detach().numpy().astype(np.float32),
        "face_input": face.detach().numpy().astype(np.float32),
    }


def verify(role, config, model, inputs, fp32_path, int8_path, want_quantised) -> dict:
    """Compare the exported graphs against the PyTorch model numerically."""
    fp32_session = session_for(fp32_path, config)
    with torch.no_grad():
        reference = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
    reference = reference.numpy()
    fp32_output = fp32_session.run(["output"], _feed(role, inputs))[0]

    result = {
        "fp32_output_shape": list(np.asarray(fp32_output).shape),
        "fp32_vs_pytorch_max_abs_diff": max_abs_difference(reference, fp32_output),
        "onnx_input_order": [item.name for item in fp32_session.get_inputs()],
    }
    if want_quantised:
        int8_session = session_for(int8_path, config)
        int8_output = int8_session.run(["output"], _feed(role, inputs))[0]
        result["int8_vs_fp32_max_abs_diff"] = max_abs_difference(fp32_output, int8_output)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default=datetime.now().strftime("export-%Y%m%d-%H%M%S"))
    parser.add_argument("--force", action="store_true", help="allow writing into a non-empty run dir")
    parser.add_argument("--skip-quantise", action="store_true", help="fp32 ONNX only")
    args = parser.parse_args(argv)

    config = load_config(root=ROOT)
    out_dir = prepare_output_dir(ROOT, "exports", args.run_name, force=args.force)
    scratch = None if args.skip_quantise else ascii_scratch_dir(out_dir, args.run_name)

    print("[*] loading checkpoints from %s (read-only)" % config.pytorch_dir)
    voice, face, fusion = load_models(config)
    inputs = dummy_inputs(config)

    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "opset": 11,
        "pytorch_sources": {
            role: {
                "path": str(config.pytorch_path(role)),
                "sha256": file_digest(config.pytorch_path(role)),
            }
            for role in ("face", "voice", "fusion")
        },
        "outputs": {},
        "note": (
            "Exports are diagnostic copies. weights/onnx/ is never modified; "
            "compare hashes with docs/MODEL_ARTIFACT_MANIFEST.md before ever "
            "considering a replacement."
        ),
    }

    for role, model in (("voice", voice), ("face", face), ("fusion", fusion)):
        stem = "attention_fusion" if role == "fusion" else "%s_extractor" % role
        fp32_path = out_dir / ("%s_fp32.onnx" % stem)
        int8_path = out_dir / ("%s_quant.onnx" % stem)

        if role == "fusion":
            names = ["voice_input", "face_input"]
            dynamic_axes = {
                "voice_input": {0: "batch_size"},
                "face_input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        else:
            names = ["%s_input" % role]
            dynamic_axes = {names[0]: {0: "batch_size"}, "output": {0: "batch_size"}}

        print("[*] exporting %s -> %s" % (role, fp32_path.name))
        export_fp32(model, inputs[role], fp32_path, names, dynamic_axes)

        if not args.skip_quantise:
            print("[*] quantising %s -> %s (dynamic QInt8)" % (role, int8_path.name))
            quantise(fp32_path, int8_path, scratch, stem)

        entry = {
            "fp32": {"file": fp32_path.name, "bytes": fp32_path.stat().st_size,
                     "sha256": file_digest(fp32_path)},
            "verification": verify(
                role, config, model, inputs[role], fp32_path, int8_path, not args.skip_quantise
            ),
        }
        if not args.skip_quantise:
            entry["int8"] = {"file": int8_path.name, "bytes": int8_path.stat().st_size,
                             "sha256": file_digest(int8_path)}
            shipped = config.onnx_path(role)
            entry["shipped_onnx_sha256"] = file_digest(shipped) if shipped.is_file() else None
            entry["matches_shipped_bytes"] = (
                entry["int8"]["sha256"] == entry["shipped_onnx_sha256"]
            )
        metadata["outputs"][role] = entry
        print("    verification: %s" % entry["verification"])

    if scratch is not None and scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)

    (out_dir / "export.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[+] exports written to %s" % out_dir)
    print("    weights/onnx/ untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
