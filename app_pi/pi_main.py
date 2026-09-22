# ============================================================================
# HISTORICAL / EDGE-ONLY - Raspberry Pi 4B entry point. Preserved for
# provenance; not part of the supported Phase 1 workflow.
#
# * It was never actually run on a Raspberry Pi in this project, and the
#   "INT8 is 22x faster" claim in the original report was never reproduced.
#   On x86 the INT8 graphs are ~22x SLOWER than fp32 (see docs/benchmarks/).
# * It cannot run on Windows or macOS: it needs sounddevice capture and
#   /sys/firmware/... device files, and it hard-codes the 2-thread Pi setting.
# * Its dependency pin (requirements_pi.txt: onnxruntime==1.16.0) cannot load
#   the shipped INT8 graphs, which need onnxruntime >= 1.24.1.
# * Phase 1 renamed the weight directories (weights/onnx_int8 -> weights/onnx);
#   the update below is the only functional change made to this file.
#
# The supported, testable equivalent is:  python -m multimodal_auth --help
# See docs/HISTORICAL_EDGE_DEPLOYMENT.md and docs/LIMITATIONS.md.
# ============================================================================
import os
import time
import numpy as np
import onnxruntime as ort
import cv2
import sounddevice as sd
from hardware_monitor import PiHardwareMonitor
import yaml


class PiMultimodalAuth:
    def __init__(self, config_path="../config/settings.yaml"):
        # 读取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        print("[Pi] 初始化多模态推理引擎...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # 限制线程防卡死

        onnx_dir = self.cfg['paths']['onnx_weights_dir']
        # 树莓派上运行的是极致压缩的 _quant (INT8) 模型
        self.voice_sess = ort.InferenceSession(os.path.join(onnx_dir, "voice_extractor_quant.onnx"), opts)
        self.face_sess = ort.InferenceSession(os.path.join(onnx_dir, "face_extractor_quant.onnx"), opts)
        self.fusion_sess = ort.InferenceSession(os.path.join(onnx_dir, "attention_fusion_quant.onnx"), opts)

        self.monitor = PiHardwareMonitor()

    def dummy_capture(self):
        """此处模拟从麦克风和摄像头获取经过预处理的数据，
           实际部署时需要换成你的 opencv 和 sounddevice 真实采集代码"""
        voice_feat = np.random.randn(1, 1, self.cfg['model']['voice_n_mfcc'],
                                     self.cfg['model']['voice_max_pad_len']).astype(np.float32)
        face_feat = np.random.randn(1, 3, self.cfg['model']['face_image_size'],
                                    self.cfg['model']['face_image_size']).astype(np.float32)
        return voice_feat, face_feat

    def authenticate(self):
        print("\n=== 开始基于树莓派的多模态身份认证 ===")
        t_start = time.time()

        # 1. 采集特征 (模拟)
        voice_input, face_input = self.dummy_capture()
        t_capture = time.time()

        # 2. 单模态特征提取
        v_emb = self.voice_sess.run(['output'], {'voice_input': voice_input})[0]
        f_emb = self.face_sess.run(['output'], {'face_input': face_input})[0]

        # 3. 创新方案B：Attention 融合判断
        fusion_inputs = {'voice_input': v_emb, 'face_input': f_emb}
        score = self.fusion_sess.run(['output'], fusion_inputs)[0][0][0]

        t_end = time.time()

        # 4. 生成性能报告
        stats = self.monitor.get_system_stats()
        latency = (t_end - t_capture) * 1000
        threshold = self.cfg['inference']['fusion_threshold']

        print(f"-> 融合置信度: {score:.4f} (阈值: {threshold})")
        print("-> 结果: 🟢 认证通过" if score >= threshold else "-> 结果: 🔴 拒绝访问")
        print("--- 树莓派性能实测报告 ---")
        print(f" * 端到端推理延迟 : {latency:.1f} ms")
        print(f" * CPU 占用率     : {stats['cpu_percent']} %")
        print(f" * RAM 占用率     : {stats['ram_percent']} % ({stats['ram_used_mb']:.0f} MB)")
        print(f" * 核心温度       : {stats['temp_c']} °C\n")


if __name__ == "__main__":
    # 请确保在项目根目录外层运行，或正确设置相对路径
    try:
        app = PiMultimodalAuth()
        app.authenticate()
    except Exception as e:
        print(f"初始化失败: {e}\n(请确认是否已在 PC 端完成了量化并把模型放置到了 weights/onnx/)")