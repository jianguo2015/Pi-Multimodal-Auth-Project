#!/usr/bin/env python
"""Latency benchmark for the shipped artifacts.

Reports per-model and end-to-end latency for

  * the INT8 graphs actually shipped in ``weights/onnx/``,
  * the fp32 ONNX reference graphs,
  * optionally the PyTorch checkpoints (``--include-pytorch``).

This replaces the unsupported "INT8 is 22x faster" claim of the original
report. Every number it prints is measured on the machine it runs on; results
are written to ``outputs/benchmarks/<run-name>/`` (never into ``weights/``).

Usage::

    python scripts/benchmark.py --warmup 5 --runs 30
    python scripts/benchmark.py --face examples/sample_face.jpg --voice examples/sample_voice.wav
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multimodal_auth.config import load_config  # noqa: E402
from multimodal_auth.pipeline.onnx_backend import OnnxBackend  # noqa: E402
from multimodal_auth.preprocessing import preprocess_face, preprocess_voice  # noqa: E402
from multimodal_auth.safety import prepare_output_dir  # noqa: E402


def timeit(fn, warmup: int, runs: int) -> dict:
    for _ in range(max(1, warmup)):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return {
        "runs": runs,
        "warmup": warmup,
        "mean_ms": round(statistics.fmean(samples), 4),
        "median_ms": round(statistics.median(samples), 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "stdev_ms": round(statistics.pstdev(samples), 4) if len(samples) > 1 else 0.0,
    }


def onnx_session(config, role: str, variant: str):
    import onnxruntime as ort

    spec = config.models[role]
    name = spec.onnx_file if variant == "int8" else spec.onnx_file.replace("_quant", "_fp32")
    path = config.onnx_dir / name
    options = ort.SessionOptions()
    options.intra_op_num_threads = config.runtime.intra_op_num_threads
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=[config.runtime.provider]
    )
    return session, path


def _fusion_fn(session, voice_embedding, face_embedding):
    voices = np.ascontiguousarray(voice_embedding[None, :], dtype=np.float32)
    faces = np.ascontiguousarray(face_embedding[None, :], dtype=np.float32)
    return lambda: session.run(["output"], {"voice_input": voices, "face_input": faces})


def _single_fn(session, input_name, tensor):
    batch = np.ascontiguousarray(tensor[None, ...], dtype=np.float32)
    return lambda: session.run(["output"], {input_name: batch})


def _portable(path) -> str:
    """Report paths relative to the repository so reports stay publishable.

    Absolute Windows paths contain the local user name; benchmark evidence is
    committed to ``docs/benchmarks/``, so the user name must not leak into it.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return "<outside the repository>"


def benchmark_models(config, face_tensor, voice_tensor, report, warmup: int, runs: int) -> None:
    """Time each graph in both precisions and record the results."""
    for variant in ("int8", "fp32"):
        for role, tensor in (("face", face_tensor), ("voice", voice_tensor)):
            session, path = onnx_session(config, role, variant)
            input_name = session.get_inputs()[0].name
            stats = timeit(_single_fn(session, input_name, tensor), warmup, runs)
            stats["file"] = path.name
            stats["bytes"] = path.stat().st_size
            report["onnx_models"]["%s_%s" % (role, variant)] = stats
            print("    %-14s %8.3f ms (median)  %8.1f KB  %s"
                  % (role + "/" + variant, stats["median_ms"], stats["bytes"] / 1024, path.name))

        fusion_session, fusion_path = onnx_session(config, "fusion", variant)
        voice_embedding = report["_voice_embedding"]
        face_embedding = report["_face_embedding"]
        stats = timeit(
            _fusion_fn(fusion_session, voice_embedding, face_embedding), warmup, runs
        )
        stats["file"] = fusion_path.name
        stats["bytes"] = fusion_path.stat().st_size
        report["onnx_models"]["fusion_%s" % variant] = stats
        print("    %-14s %8.3f ms (median)  %8.1f KB  %s"
              % ("fusion/" + variant, stats["median_ms"], stats["bytes"] / 1024, fusion_path.name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--face", default=str(ROOT / "examples" / "sample_face.jpg"))
    parser.add_argument("--voice", default=str(ROOT / "examples" / "sample_voice.wav"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--run-name", default=None, help="output subdirectory name")
    parser.add_argument("--force", action="store_true", help="allow writing into an existing run dir")
    parser.add_argument("--include-pytorch", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(root=ROOT)
    for label, path in (("face", args.face), ("voice", args.voice)):
        if not Path(path).is_file():
            print(
                "missing %s sample: %s\nrun `python examples/make_examples.py` first, "
                "or pass --%s <your file>" % (label, path, label),
                file=sys.stderr,
            )
            return 2

    import onnxruntime as ort

    print("[*] preprocessing inputs...")
    face_tensor = preprocess_face(args.face, config.face)
    voice_tensor = preprocess_voice(args.voice, config.voice)
    backend = OnnxBackend(config)
    face_embedding = backend.embed_face(face_tensor)
    voice_embedding = backend.embed_voice(voice_tensor)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
            "intra_op_num_threads": config.runtime.intra_op_num_threads,
        },
        "inputs": {
            "face": _portable(args.face),
            "voice": _portable(args.voice),
            "face_tensor_shape": list(face_tensor.shape),
            "voice_tensor_shape": list(voice_tensor.shape),
        },
        "warmup": args.warmup,
        "runs": args.runs,
        "onnx_models": {},
        "pipelines": {},
        "_face_embedding": face_embedding,
        "_voice_embedding": voice_embedding,
    }

    print("[*] benchmarking ONNX graphs (int8 vs fp32)...")
    benchmark_models(config, face_tensor, voice_tensor, report, args.warmup, args.runs)
    report.pop("_face_embedding")
    report.pop("_voice_embedding")

    print("[*] benchmarking end-to-end pipelines...")
    from multimodal_auth.pipeline import MultimodalPipeline

    pipeline_runs = max(3, args.runs // 3)
    onnx_pipeline = MultimodalPipeline(config, backend="onnx")
    onnx_pipeline.run(args.face, args.voice)  # warm up caches and sessions
    report["pipelines"]["onnx_int8_end_to_end"] = timeit(
        lambda: onnx_pipeline.run(args.face, args.voice), args.warmup, pipeline_runs
    )

    if args.include_pytorch:
        try:
            torch_pipeline = MultimodalPipeline(config, backend="pytorch")
            torch_pipeline.run(args.face, args.voice)
            report["pipelines"]["pytorch_end_to_end"] = timeit(
                lambda: torch_pipeline.run(args.face, args.voice), args.warmup, pipeline_runs
            )
        except Exception as exc:  # noqa: BLE001
            report["pipelines"]["pytorch_end_to_end"] = {"error": str(exc)}
            print("    pytorch backend unavailable: %s" % exc)

    int8_total = sum(
        v["median_ms"] for k, v in report["onnx_models"].items() if k.endswith("_int8")
    )
    fp32_total = sum(
        v["median_ms"] for k, v in report["onnx_models"].items() if k.endswith("_fp32")
    )
    report["summary"] = {
        "int8_three_model_total_ms": round(int8_total, 4),
        "fp32_three_model_total_ms": round(fp32_total, 4),
        "int8_vs_fp32_speed_ratio": round(int8_total / fp32_total, 3) if fp32_total else None,
        "note": (
            "A ratio > 1 means INT8 is SLOWER than fp32 on this host. The original "
            "report's 'INT8 is 22x faster' claim was never measured; it does not hold "
            "on this x86_64 host (see docs/LIMITATIONS.md)."
        ),
    }

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = prepare_output_dir(ROOT, "benchmarks", run_name, force=args.force)
    (out_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")

    print("\n=== summary (median, ms) ===")
    for name, stats in report["onnx_models"].items():
        print("  %-16s %8.3f" % (name, stats["median_ms"]))
    print("  %-16s %8.3f" % ("int8 3-model total", int8_total))
    print("  %-16s %8.3f" % ("fp32 3-model total", fp32_total))
    print(
        "  INT8 / fp32 ratio = %.2fx  (>1 means INT8 is slower)"
        % report["summary"]["int8_vs_fp32_speed_ratio"]
    )
    print(
        "  end-to-end incl. preprocessing: median %.1f ms"
        % report["pipelines"]["onnx_int8_end_to_end"]["median_ms"]
    )
    print("\nresults written to %s" % out_dir)
    return 0


def render_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Benchmark report",
        "",
        "Generated: `%s`" % report["generated_at"],
        "",
        "| environment | value |",
        "| --- | --- |",
        "| Python | %s |" % env["python"],
        "| platform | `%s` |" % env["platform"],
        "| processor | %s |" % (env["processor"] or "n/a"),
        "| onnxruntime | %s |" % env["onnxruntime"],
        "| available providers | %s |" % ", ".join(env["providers"]),
        "| intra_op_num_threads | %s |" % env["intra_op_num_threads"],
        "| warmup / runs | %s / %s |" % (report["warmup"], report["runs"]),
        "",
        "Inputs: `%s` (face) and `%s` (voice)." % (report["inputs"]["face"], report["inputs"]["voice"]),
        "",
        "## Per-model latency",
        "",
        "| model | precision | file | KB | median ms | mean ms | min ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in sorted(report["onnx_models"].items()):
        role, _, variant = name.rpartition("_")
        lines.append(
            "| %s | %s | `%s` | %.1f | %.3f | %.3f | %.3f | %.3f |"
            % (
                role,
                variant,
                stats["file"],
                stats["bytes"] / 1024,
                stats["median_ms"],
                stats["mean_ms"],
                stats["min_ms"],
                stats["p95_ms"],
            )
        )

    summary = report["summary"]
    lines += [
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
        "| INT8 three-model total (median) | %.3f ms |" % summary["int8_three_model_total_ms"],
        "| fp32 three-model total (median) | %.3f ms |" % summary["fp32_three_model_total_ms"],
        "| INT8 / fp32 ratio | %.2fx |" % summary["int8_vs_fp32_speed_ratio"],
        "",
        "> %s" % summary["note"],
        "",
        "## End-to-end",
        "",
        "| pipeline | median ms | mean ms | p95 ms |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in report["pipelines"].items():
        if "error" in stats:
            lines.append("| %s | n/a (%s) | – | – |" % (name, stats["error"]))
            continue
        lines.append(
            "| %s | %.3f | %.3f | %.3f |" % (name, stats["median_ms"], stats["mean_ms"], stats["p95_ms"])
        )
    lines += [
        "",
        "These numbers are host-specific. No Raspberry Pi measurement exists for this",
        "project, so nothing here may be presented as edge-device performance.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
