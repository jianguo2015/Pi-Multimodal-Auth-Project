# Benchmark report

Generated: `2026-09-22T20:12:02`

| environment | value |
| --- | --- |
| Python | 3.13.1 |
| platform | `Windows-11-10.0.26200-SP0` |
| processor | AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD |
| onnxruntime | 1.24.1 |
| available providers | AzureExecutionProvider, CPUExecutionProvider |
| intra_op_num_threads | 2 |
| warmup / runs | 5 / 30 |

Inputs: `examples/sample_face.jpg` (face) and `examples/sample_voice.wav` (voice).

## Per-model latency

| model | precision | file | KB | median ms | mean ms | min ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| face | fp32 | `face_extractor_fp32.onnx` | 1648.3 | 0.310 | 0.314 | 0.306 | 0.335 |
| face | int8 | `face_extractor_quant.onnx` | 422.7 | 7.859 | 7.840 | 7.551 | 8.115 |
| fusion | fp32 | `attention_fusion_fp32.onnx` | 109.5 | 0.015 | 0.016 | 0.015 | 0.017 |
| fusion | int8 | `attention_fusion_quant.onnx` | 39.2 | 0.021 | 0.021 | 0.021 | 0.024 |
| voice | fp32 | `voice_extractor_fp32.onnx` | 428.6 | 0.941 | 0.947 | 0.928 | 0.968 |
| voice | int8 | `voice_extractor_quant.onnx` | 114.5 | 25.441 | 25.218 | 23.765 | 26.448 |

## Summary

| metric | value |
| --- | --- |
| INT8 three-model total (median) | 33.321 ms |
| fp32 three-model total (median) | 1.267 ms |
| INT8 / fp32 ratio | 26.30x |

> A ratio > 1 means INT8 is SLOWER than fp32 on this host. The original report's 'INT8 is 22x faster' claim was never measured; it does not hold on this x86_64 host (see docs/LIMITATIONS.md).

## End-to-end

| pipeline | median ms | mean ms | p95 ms |
| --- | --- | --- | --- |
| onnx_int8_end_to_end | 59.973 | 60.520 | 63.185 |

These numbers are host-specific. No Raspberry Pi measurement exists for this
project, so nothing here may be presented as edge-device performance.
