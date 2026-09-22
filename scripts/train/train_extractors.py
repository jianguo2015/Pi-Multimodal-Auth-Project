#!/usr/bin/env python
"""Sandboxed re-training of the two feature extractors (research only).

Port of the frozen ``scripts_pc/03_train_single_models.py`` with three
guarantees the original did not have:

1. **read-only inputs** - features and the config are only opened for reading;
2. **sandboxed outputs** - checkpoints land in ``outputs/training/<run>/`` and
   never in ``weights/``; the script fails closed if that directory already
   holds files unless ``--force`` is passed;
3. **seeded** - ``--seed`` makes a run repeatable, which the original was not.

WARNING: this does not reproduce the shipped checkpoints bit-for-bit. The
original training used no seed and consumed data that is intentionally not in
the repository. Do not overwrite ``weights/``; the published artifacts and
their hashes are the reference (see docs/MODEL_ARTIFACT_MANIFEST.md).

Usage::

    python scripts/train/train_extractors.py --features-root outputs/features --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    ROOT,
    FeatureDataset,
    device_for_preference,
    project_config,
    sandbox_dir,
    seed_everything,
    train_extractor,
    write_run_metadata,
)

from multimodal_auth.models import FaceExtractor, VoiceExtractor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features-root",
        default=str(ROOT / "outputs" / "features"),
        help="directory containing the voice/ and face/ feature folders",
    )
    parser.add_argument("--epochs", type=int, default=None, help="default: config/settings.yaml value")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-name", default=datetime.now().strftime("extractors-%Y%m%d-%H%M%S"))
    parser.add_argument("--force", action="store_true", help="allow writing into a non-empty run dir")
    parser.add_argument("--modality", default="both", choices=["voice", "face", "both"])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = project_config()
    features_root = Path(args.features_root).resolve()

    epochs = args.epochs if args.epochs is not None else 20
    lr = args.lr if args.lr is not None else 0.001
    batch_size = args.batch_size if args.batch_size is not None else 32

    seed_everything(args.seed)
    device = device_for_preference(args.device)
    out_dir = sandbox_dir(args.run_name, args.force)

    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "device": str(device),
        "features_root": str(features_root),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "config": str(config.source),
        "extractors": {},
        "warning": (
            "Sanity/training sandbox only: these checkpoints are NOT the shipped "
            "artifacts and must never be copied over weights/."
        ),
    }

    for modality, extractor_cls, sub in (
        ("voice", VoiceExtractor, "voice"),
        ("face", FaceExtractor, "face"),
    ):
        if args.modality not in (modality, "both"):
            continue
        data_dir = features_root / sub
        if not data_dir.is_dir():
            print("[!] skipping %s: %s does not exist" % (modality, data_dir))
            continue

        print("\n" + "=" * 60)
        print("[*] training the %s extractor" % modality)
        print("=" * 60)
        dataset = FeatureDataset(data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if modality == "voice":
            extractor = extractor_cls(
                n_mfcc=config.voice.n_mfcc, embed_dim=config.models["voice"].embed_dim
            )
        else:
            extractor = extractor_cls(embed_dim=config.models["face"].embed_dim)

        checkpoint = out_dir / ("%s_extractor_weights.pth" % modality)
        summary = train_extractor(
            extractor,
            loader,
            dataset.num_classes,
            extractor.fc.out_features,
            epochs,
            lr,
            checkpoint,
            device,
        )
        summary.update(
            {
                "dataset_dir": str(data_dir),
                "samples": len(dataset),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _digest(checkpoint),
            }
        )
        metadata["extractors"][modality] = summary

    write_run_metadata(out_dir, metadata)
    print("\n[+] training sandbox: %s" % out_dir)
    print("    nothing was written to weights/ (read-only inputs)")
    return 0


def _digest(path: Path) -> str:
    from multimodal_auth.safety import file_digest

    return file_digest(path)


if __name__ == "__main__":
    raise SystemExit(main())
