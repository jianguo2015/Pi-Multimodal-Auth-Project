#!/usr/bin/env python
"""Sandboxed re-training of the attention fusion head (research only).

Port of the frozen ``scripts_pc/04_train_fusion_scheme_b.py``:

* same architecture (``ModalAttentionFusion``), same loss (``BCELoss``), same
  optimiser (``Adam``, lr 0.001), same mini-batch recipe (2 genuine / 1
  stranger-voice / 1 stranger-face) and same epoch count, so the recipe stays
  auditable;
* the two extractors are loaded from ``weights/pytorch/`` **read-only** with
  ``strict=True`` and kept frozen, exactly like the original;
* the new fusion checkpoint is written to ``outputs/training/<run>/`` only;
  ``weights/`` is never touched and the script fails closed on a non-empty
  output directory unless ``--force`` is given.

The original used an unseeded RNG, so its exact loss curve cannot be
reproduced; ``--seed`` fixes that here (a deliberate addition).

Usage::

    python scripts/train/train_fusion.py --features-root outputs/features --epochs 200
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    ROOT,
    device_for_preference,
    project_config,
    sandbox_dir,
    seed_everything,
    write_run_metadata,
)

from multimodal_auth.fusion import ModalAttentionFusion  # noqa: E402
from multimodal_auth.models import FaceExtractor, VoiceExtractor  # noqa: E402


def collect_files(features_root: Path, modality: str):
    """Return ``(me_files, stranger_files)`` for one modality.

    ``User_Me`` (case-insensitive) is the enrolled identity, every other
    subdirectory is a stranger - the same convention as the original script.
    """
    base = features_root / modality
    if not base.is_dir():
        raise FileNotFoundError("feature directory not found: %s" % base)
    me, strangers = [], []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(str(p) for p in directory.glob("*.npy"))
        if directory.name.lower() == "user_me":
            me.extend(files)
        else:
            strangers.extend(files)
    return me, strangers


def load_frozen_extractors(config, device):
    """Load the shipped extractors read-only and freeze them."""
    voice = VoiceExtractor(
        n_mfcc=config.voice.n_mfcc, embed_dim=config.models["voice"].embed_dim
    )
    face = FaceExtractor(embed_dim=config.models["face"].embed_dim)
    voice.load_state_dict(torch.load(str(config.pytorch_path("voice")), map_location="cpu"))
    face.load_state_dict(torch.load(str(config.pytorch_path("face")), map_location="cpu"))
    voice.eval().to(device)
    face.eval().to(device)
    for parameter in list(voice.parameters()) + list(face.parameters()):
        parameter.requires_grad_(False)
    return voice, face


def new_fusion(config, device):
    return ModalAttentionFusion(
        voice_dim=config.models["voice"].embed_dim,
        face_dim=config.models["face"].embed_dim,
        hidden_dim=config.models["fusion"].hidden_dim or 64,
    ).to(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features-root", default=str(ROOT / "outputs" / "features"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", default=datetime.now().strftime("fusion-%Y%m%d-%H%M%S"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report-shipped",
        action="store_true",
        help="also score the shipped fusion head on the genuine grid (read-only)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = project_config()
    features_root = Path(args.features_root).resolve()

    me_voice, stranger_voice = collect_files(features_root, "voice")
    me_face, stranger_face = collect_files(features_root, "face")
    print(
        "[*] samples -> genuine voice:%d face:%d | stranger voice:%d face:%d"
        % (len(me_voice), len(me_face), len(stranger_voice), len(stranger_face))
    )
    if min(len(me_voice), len(me_face), len(stranger_voice), len(stranger_face)) == 0:
        print("[!] need User_Me and at least one stranger directory for both modalities")
        return 2

    seed_everything(args.seed)
    device = device_for_preference(args.device)
    out_dir = sandbox_dir(args.run_name, args.force)

    voice_net, face_net = load_frozen_extractors(config, device)
    fusion = new_fusion(config, device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr)

    def embedding(path: str, net):
        with torch.no_grad():
            tensor = torch.tensor(np.load(path), dtype=torch.float32).unsqueeze(0)
            return net(tensor.to(device))

    def pick(paths, net):
        return embedding(str(np.random.choice(paths)), net)

    history = []
    fusion.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        voices, faces, targets = [], [], []
        for _ in range(2):  # genuine: my voice + my face
            voices.append(pick(me_voice, voice_net))
            faces.append(pick(me_face, face_net))
            targets.append([1.0])
        voices.append(pick(stranger_voice, voice_net))  # my face + stranger voice
        faces.append(pick(me_face, face_net))
        targets.append([0.0])
        voices.append(pick(me_voice, voice_net))  # my voice + stranger face
        faces.append(pick(stranger_face, face_net))
        targets.append([0.0])

        output = fusion(torch.cat(voices, dim=0), torch.cat(faces, dim=0))
        loss = criterion(output, torch.tensor(targets, dtype=torch.float32, device=device))
        loss.backward()
        optimizer.step()

        history.append(round(float(loss.item()), 6))
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print("Epoch [%d/%d] Loss: %.6f" % (epoch + 1, args.epochs, loss.item()))

    checkpoint = out_dir / "attention_fusion_weights.pth"
    torch.save(fusion.state_dict(), str(checkpoint))

    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "features_root": str(features_root),
        "frozen_extractors": {
            "voice": str(config.pytorch_path("voice")),
            "face": str(config.pytorch_path("face")),
        },
        "final_loss": history[-1] if history else None,
        "loss_curve": history,
        "checkpoint": str(checkpoint),
    }
    if args.report_shipped:
        metadata["shipped_reference"] = score_shipped(
            config, me_voice, me_face, voice_net, face_net
        )

    write_run_metadata(out_dir, metadata)
    print("\n[+] fusion checkpoint written to %s" % checkpoint)
    print("    weights/ was only read, never written")
    return 0


def score_shipped(config, me_voice, me_face, voice_net, face_net) -> dict:
    """Score the shipped fusion head on the genuine grid (Phase 0 reference)."""
    shipped = ModalAttentionFusion(
        voice_dim=config.models["voice"].embed_dim,
        face_dim=config.models["face"].embed_dim,
        hidden_dim=config.models["fusion"].hidden_dim or 64,
    )
    shipped.load_state_dict(torch.load(str(config.pytorch_path("fusion")), map_location="cpu"))
    shipped.eval()

    with torch.no_grad():
        voice_embeddings = torch.cat(
            [
                voice_net(torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0))
                for p in sorted(me_voice)
            ]
        )
        face_embeddings = torch.cat(
            [
                face_net(torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0))
                for p in sorted(me_face)
            ]
        )
        scores = [
            float(shipped(voice_embeddings[i: i + 1], face_embeddings[j: j + 1])[0])
            for i in range(voice_embeddings.size(0))
            for j in range(face_embeddings.size(0))
        ]

    result = {
        "pairs": len(scores),
        "mean": round(float(np.mean(scores)), 6),
        "min": round(float(np.min(scores)), 6),
        "max": round(float(np.max(scores)), 6),
        "threshold": config.threshold,
        "accept_rate": round(sum(s >= config.threshold for s in scores) / len(scores), 4),
    }
    print("[*] shipped fusion head on the genuine grid: %s" % result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
