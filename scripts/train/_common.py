#!/usr/bin/env python
"""Shared helpers for the sandboxed training scripts (research only).

Safety contract, enforced here once for every trainer:

* inputs are **read-only**: feature directories and ``weights/pytorch``
  checkpoints are only ever opened for reading;
* all outputs go to ``<root>/outputs/training/<run-name>/`` through
  :func:`multimodal_auth.safety.prepare_output_dir`, which refuses to write into
  ``weights/``, ``data/``, ``configs/``, ``src/``, ``tests/``, ``app_pi/`` and
  fails closed when the target directory is not empty unless ``--force`` is
  given. This is the guard that the Phase 0 incident showed was missing;
* the shipped artefacts in ``weights/`` are never modified, so a training run
  can never silently invalidate the published model manifest.
"""

from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multimodal_auth.config import load_config  # noqa: E402
from multimodal_auth.safety import prepare_output_dir  # noqa: E402


def seed_everything(seed: int) -> None:
    """Make a run reproducible.

    The original scripts seeded nothing, which is why their results cannot be
    reproduced bit-for-bit today. Documented here as a deliberate addition.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_for_preference(preference: str) -> torch.device:
    if preference == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but unavailable, falling back to CPU")
        preference = "cpu"
    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(preference)


class FeatureDataset(Dataset):
    """``.npy`` features grouped by subdirectory, one class per subdirectory.

    Identical layout and label ordering (sorted directory names) as the
    original ``scripts_pc/03_train_single_models.py``.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError("feature directory not found: %s" % self.data_dir)

        self.file_paths = []
        self.labels = []
        self.label_map = {}
        for index, speaker in enumerate(sorted(p.name for p in self.data_dir.iterdir() if p.is_dir())):
            self.label_map[speaker] = index
            files = sorted(glob.glob(str(self.data_dir / speaker / "*.npy")))
            self.file_paths.extend(files)
            self.labels.extend([index] * len(files))

        self.num_classes = len(self.label_map)
        if not self.file_paths:
            raise FileNotFoundError("no .npy features found under %s" % self.data_dir)
        print(
            " -> loaded %d samples, %d classes: %s"
            % (len(self.file_paths), self.num_classes, ", ".join(sorted(self.label_map)))
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index):
        feature = np.load(self.file_paths[index])
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(
            self.labels[index], dtype=torch.long
        )


def train_extractor(
    extractor: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    embed_dim: int,
    epochs: int,
    lr: float,
    save_path: Path,
    device: torch.device,
) -> dict:
    """Train ``extractor`` with a temporary linear head, keep the extractor.

    Same recipe as the original: ``CrossEntropyLoss`` + ``Adam``, the linear
    classification head is discarded afterwards so only the 128-d embedding
    network is saved.
    """
    classifier_head = nn.Linear(embed_dim, num_classes)
    model = nn.Sequential(extractor, classifier_head).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / max(total, 1)
        mean_loss = total_loss / max(len(dataloader), 1)
        history.append({"epoch": epoch + 1, "loss": round(mean_loss, 6), "accuracy": round(accuracy, 4)})
        print("Epoch[%d/%d] | Loss: %.4f | Acc: %.2f%%" % (epoch + 1, epochs, mean_loss, accuracy))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extractor.state_dict(), str(save_path))
    print("[+] extractor weights written to %s" % save_path)
    return {"epochs": epochs, "lr": lr, "num_classes": num_classes, "history": history}


def write_run_metadata(out_dir: Path, payload: dict) -> None:
    (out_dir / "run.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def sandbox_dir(run_name: str | None, force: bool) -> Path:
    return prepare_output_dir(ROOT, "training", run_name, force=force)


def project_config():
    return load_config(root=ROOT)
