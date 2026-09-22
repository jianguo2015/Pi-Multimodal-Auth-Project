import torch
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
import os
import shutil
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.voice_extractor import VoiceExtractor
from models.face_extractor import FaceExtractor
from models.attention_fusion import ModalAttentionFusion


def export_and_quantize_safely(model, dummy_input, model_name, out_dir, input_names, dynamic_axes):
    fp32_path = os.path.abspath(os.path.join(out_dir, f"{model_name}_fp32.onnx"))
    int8_path = os.path.abspath(os.path.join(out_dir, f"{model_name}_quant.onnx"))

    torch.onnx.export(
        model, dummy_input, fp32_path,
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=input_names, output_names=['output'], dynamic_axes=dynamic_axes
    )

    current_drive = os.path.splitdrive(os.path.abspath(__file__))[0] or "C:"
    safe_dir = os.path.join(current_drive, os.sep, "ort_temp_pi_quant")
    os.makedirs(safe_dir, exist_ok=True)

    safe_fp32 = os.path.join(safe_dir, f"{model_name}.onnx")
    safe_prep = os.path.join(safe_dir, f"{model_name}_prep.onnx")
    safe_int8 = os.path.join(safe_dir, f"{model_name}_quant.onnx")

    shutil.copy(fp32_path, safe_fp32)
    os.environ["TMP"] = safe_dir
    os.environ["TEMP"] = safe_dir

    shape_inference.quant_pre_process(safe_fp32, safe_prep, skip_symbolic_shape=False)
    quantize_dynamic(safe_prep, safe_int8, weight_type=QuantType.QInt8)

    shutil.copy(safe_int8, int8_path)
    try:
        shutil.rmtree(safe_dir)
    except:
        pass
    print(f"[+] 模型 {model_name} 量化完成: {int8_path}")


def main():
    with open("../config/settings.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 权重所在目录
    weights_dir = cfg['paths']['pytorch_weights_dir']
    out_dir = cfg['paths']['onnx_weights_dir']
    os.makedirs(out_dir, exist_ok=True)

    # === 1. 导出声纹提取网络 ===
    print("\n[*] 正在加载权重并导出声纹模型...")
    voice_net = VoiceExtractor(n_mfcc=cfg['model']['voice_n_mfcc'], embed_dim=cfg['model']['embed_dim'])
    # 【核心修正】：加载 03 步训练的权重
    v_path = os.path.join(weights_dir, "voice_extractor_weights.pth")
    voice_net.load_state_dict(torch.load(v_path))
    voice_net.eval()

    dummy_v = torch.randn(1, 1, cfg['model']['voice_n_mfcc'], cfg['model']['voice_max_pad_len'])
    export_and_quantize_safely(voice_net, dummy_v, "voice_extractor", out_dir,
                               input_names=['voice_input'],
                               dynamic_axes={'voice_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # === 2. 导出人脸提取网络 ===
    print("\n[*] 正在加载权重并导出视觉模型...")
    face_net = FaceExtractor(embed_dim=cfg['model']['embed_dim'])
    # 【核心修正】：加载 03 步训练的权重
    f_path = os.path.join(weights_dir, "face_extractor_weights.pth")
    face_net.load_state_dict(torch.load(f_path))
    face_net.eval()

    dummy_f = torch.randn(1, 3, cfg['model']['face_image_size'], cfg['model']['face_image_size'])
    export_and_quantize_safely(face_net, dummy_f, "face_extractor", out_dir,
                               input_names=['face_input'],
                               dynamic_axes={'face_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    # === 3. 导出融合网络 ===
    print("\n[*] 正在加载权重并导出多模态融合网络...")
    fusion_net = ModalAttentionFusion(voice_dim=cfg['model']['embed_dim'],
                                      face_dim=cfg['model']['embed_dim'])
    # 【核心修正】：加载 04 步训练的权重
    u_path = os.path.join(weights_dir, "attention_fusion_weights.pth")
    fusion_net.load_state_dict(torch.load(u_path))
    fusion_net.eval()

    dummy_v_emb = torch.randn(1, cfg['model']['embed_dim'])
    dummy_f_emb = torch.randn(1, cfg['model']['embed_dim'])
    export_and_quantize_safely(fusion_net, (dummy_v_emb, dummy_f_emb), "attention_fusion", out_dir,
                               input_names=['voice_input', 'face_input'],
                               dynamic_axes={'voice_input': {0: 'batch_size'}, 'face_input': {0: 'batch_size'},
                                             'output': {0: 'batch_size'}})

    print("\n✅ 所有【真实训练过】的模型已导出！")


if __name__ == "__main__":
    main()