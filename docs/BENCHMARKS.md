# Observed benchmark

Measurement of the *shipped* artifacts on **one** development host. It is an
observation, not a performance specification: the same graphs behave differently
on another CPU, another runtime version or another thread count. The raw evidence
for the run below is committed next to this file
(`docs/benchmarks/20260922-development-host.{md,json}`), and it can be
regenerated at any time with:

```
python scripts/benchmark.py --runs 30
# results land in outputs/benchmarks/<timestamp>/  (gitignored)
```

## Environment

| item | value |
| --- | --- |
| operating system | Windows 11 Home 25H2, build 10.0.26200, x86_64 |
| CPU | AMD Ryzen 9 7945HX (16 cores / 32 logical processors) |
| GPU | NVIDIA GeForce RTX 4060 Laptop — **not used** by this benchmark |
| RAM | 15.7 GB total; peak RSS during the run: **N/A — not measured** |
| Python | 3.13.1 |
| onnxruntime | 1.24.1 (`CPUExecutionProvider`; `AzureExecutionProvider` is also listed by the runtime but not selected) |
| intra_op_num_threads | 2 |
| models | the nine files in `weights/` (sizes in the table below) |
| inputs | `examples/sample_face.jpg` (512×512), `examples/sample_voice.wav` (2.0 s, 16 kHz) |
| warmup / runs | 5 / 30 per model (10 runs for the end-to-end row) |
| model load time / cold start | **N/A — not measured** |

No Raspberry Pi and no other accelerator was involved. Nothing in this table is a
Pi measurement.

Committed raw evidence: [`docs/benchmarks/20260922-development-host.md`](benchmarks/20260922-development-host.md)
and the matching `.json` next to it.

## Per-model latency (median, after warm-up)

| model | precision | file | size | median | mean | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| face | INT8 | `face_extractor_quant.onnx` | 422.7 KB | **7.859 ms** | 7.840 ms | 8.115 ms |
| voice | INT8 | `voice_extractor_quant.onnx` | 114.5 KB | **25.441 ms** | 25.218 ms | 26.448 ms |
| fusion | INT8 | `attention_fusion_quant.onnx` | 39.2 KB | **0.021 ms** | 0.021 ms | 0.024 ms |
| face | fp32 | `face_extractor_fp32.onnx` | 1648.3 KB | **0.310 ms** | 0.314 ms | 0.335 ms |
| voice | fp32 | `voice_extractor_fp32.onnx` | 428.6 KB | **0.941 ms** | 0.947 ms | 0.968 ms |
| fusion | fp32 | `attention_fusion_fp32.onnx` | 109.5 KB | **0.015 ms** | 0.016 ms | 0.017 ms |

| aggregate | value |
| --- | --- |
| INT8, three models | 33.321 ms |
| fp32, three models | 1.267 ms |
| **INT8 / fp32 ratio** | **26.30× (>1 means INT8 is SLOWER)** |
| end-to-end incl. preprocessing (INT8) | 59.973 ms median (60.520 ms mean) |

## Interpretation

> On the tested x86 environment, the shipped INT8 model was substantially slower
> than FP32. This benchmark does not imply that INT8 is universally slower;
> runtime and hardware strongly affect quantized inference performance.

Quantisation is treated here as a **deployment experiment**, not as an assumption
that quantisation must be faster. On this host it is not, and that is the result
that gets published.

1. **The "INT8 is 22× faster" claim in the original report is not reproducible.**
   On this x86_64 host the three INT8 graphs are ~26× *slower* than the fp32
   graphs. Phase 0 measured 33.42 ms vs 1.48 ms with a different measurement
   method; the two runs agree within noise (33.32 ms vs 1.27 ms here).
2. **Why:** dynamic INT8 quantisation inserts `DynamicQuantizeLinear` /
   `MatMulInteger` / `ConvInteger` pairs per layer. On x86_64 with only 2
   intra-op threads the int8 kernels are not faster than the fp32 ones; they
   mainly shrink the files (1.9 MB → 0.6 MB total).
3. **On a Raspberry Pi the outcome is unmeasured.** No Pi was available and no
   number from any Pi should be quoted for this project.
4. **Preprocessing dominates the end-to-end time.** Haar detection on a
   synthetic 512×512 image costs ~34 ms of the ~60 ms; on a real multi-megapixel
   photo it is slower still. The three networks together need under 10 ms in
   fp32, so if latency matters use fp32 and optimise detection.
5. The fusion head is essentially free (0.02 ms): the cost is in the two
   extractors.

## What this does *not* measure

* Accuracy or separation (that is `docs/REPRODUCIBILITY.md`).
* Peak RAM, model load time, or cold-start cost.
* Multi-request throughput, batching, GPU, or any provider other than
  CPUExecutionProvider.
* Battery / thermal behaviour.

`scripts/benchmark.py --include-pytorch` adds the PyTorch end-to-end row when
torch is installed.
