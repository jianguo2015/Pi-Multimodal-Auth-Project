# Privacy and security

This project processes **biometric data** (face images, voice recordings). That
has consequences well beyond ordinary software.

## 1. What personal data exists, and where

| data | location | in the repository? |
| --- | --- | --- |
| raw face photos of the enrolled identity | `data/raw/face/User_Me/` | **no** (`data/` is gitignored) |
| raw voice recordings | `data/raw/voice/User_Me/` | **no** |
| derived features (`.npy`) of the enrolled identity | `data/processed/` | **no** |
| "stranger" face/voice data | `data/raw/*/Stranger_*` | **no** |
| shipped model weights | `weights/` | yes — see the risk note below |
| synthetic placeholder inputs | `examples/sample_*.{jpg,wav}` | yes — synthetic, no person |

The repository contains **no** biometric data of any person. `tests/` never
touches `data/` unless the `private_data` marker is active, and CI can run the
whole public tier with `-m "not private_data"`.

## 2. Risk carried by the published weights

Weights trained on a person's face and voice are *not* anonymous. In principle
they can be probed and, at least partially, inverted. Here the training set
consists of exactly **one** person (the project author) with 5 photos and 10
audio clips, so the shipped `face_extractor_weights.pth` and
`voice_extractor_weights.pth` are effectively a compressed representation of
that person's enrolment.

Before publishing, decide deliberately whether you accept that. If not: publish
only the *architecture* plus a training recipe (which is already in
`scripts/train/`), not the `.pth`/`.onnx` files, and drop
`docs/MODEL_ARTIFACT_MANIFEST.md` accordingly. `tests/test_artifacts.py` reads
the manifest, so removing the artifacts and the manifest rows keeps the suite
consistent.

## 3. Threat model (what this code does and does not defend against)

| threat | defended? |
| --- | --- |
| accidental modification of shipped artifacts | **yes** — `safety.py`, the manifest and the tests |
| a script writing into `weights/` or `data/` | **yes** — refused |
| an attacker replacing a model file | **partially** — detected by hash comparison, but there is no signature or secure loading; `torch.load` runs on local files |
| unreadable/corrupt input producing a fake score | **yes** — preprocessing raises instead of returning `None` |
| presentation attack (photo, replay, deep fake) | **no** — no liveness or anti-spoofing |
| someone reading the embeddings | **no** — embeddings are not encrypted, not hashed and not revocable |
| denial of service via huge input | **no** — no resolution/duration limits, no timeouts |
| model extraction / membership inference | **no** |

Treat this as a demo, not as a security boundary. Never wire it to a real lock,
door or account.

## 4. Data minimisation guidance

If you reuse this code with your own data:

* get explicit consent from anyone whose face/voice you enrol, and tell them how
  long it is kept;
* keep raw media out of version control — the `.gitignore` here already excludes
  `data/`, `private/`, `live_capture.jpg`, `outputs/`;
* store only features if you can, but remember 128-d embeddings are still
  personal data under GDPR-style regimes (and under most biometric statutes);
* delete `data/raw/` after feature extraction if you only need inference;
* do not commit benchmark or training logs that embed absolute paths (Phase 1
  made `scripts/benchmark.py` record repository-relative paths for this reason).

## 5. Open-source and licensing risk

`docs/OPEN_SOURCE_RISK.md` (Phase 0) lists the third-party items: the Haar
cascade from OpenCV, the Orl/Olivetti-style face images used for the synthetic
strangers, and the research-only status of some original data sources. Two
practical points:

* no LICENSE file has been added on purpose — choosing one is the owner's call;
* the Phase 0 audit documents (`docs/PHASE0_*.md`, `docs/CORE_ARCHITECTURE.md`)
  quote the local absolute path, which contains the Windows user name. They are
  the audit record and were deliberately not rewritten; if you publish them as
  is, that user name becomes public. Strip it first if that matters to you.

## 6. Reporting a problem

This repository has no security contact yet. If you publish it, add a
`SECURITY.md` with a contact address and a scope statement ("research
prototype, no security guarantees") before accepting external reports.
