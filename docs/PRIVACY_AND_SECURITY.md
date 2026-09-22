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

## 5. Open-source and licensing risk — resolved for this release

`docs/OPEN_SOURCE_RISK.md` (Phase 0) listed the third-party items and the open
questions about them. The release decision record is
`docs/RELEASE_AUDIT.md`; the operative terms are `LICENSE`,
`MODEL_NOTICE.md` and `THIRD_PARTY_NOTICES.md`.

| Phase 0 question | how it was resolved |
| --- | --- |
| which license, if any? | MIT for the code (`LICENSE`). The trained weights are **not** MIT: they ship under a separate research/evaluation notice, because the Olivetti/ORL face database (research use, credit to AT&T Laboratories Cambridge) contributed to training them |
| the Haar cascade | not redistributed at all — it is loaded at runtime from the installed OpenCV package. Its Intel/BSD-3-style header and the Lienhart/Maydt attribution are recorded in `THIRD_PARTY_NOTICES.md` §2, and `opencv-python-headless` is pinned `<5` because 5.x stopped shipping it |
| local absolute paths in the audit documents | sanitized in Phase 2 for the working tree; the two pre-Phase-2 commits still contain the original text, which is documented in `docs/RELEASE_AUDIT.md` §7.1 |
| reports leaking local paths | benchmark reports already used repository-relative paths; the `scripts/infer.py --json` report was fixed the same way, so `model_files` reads `weights/onnx/...` |
| the "创新方案B" novelty wording | left in place as a frozen historical label and flagged as an open item for the author (`docs/RELEASE_AUDIT.md` §10.5) |

Two questions remain open and are not conclusions this project can reach on its
own: whether parameters trained with a research-use dataset are a derivative work
(§10.6), and the institutional/IP question behind the novelty wording (§10.5).
Commercial use of the shipped weights is therefore not granted.

## 6. Reporting a problem

This repository has no security contact yet. If you publish it, add a
`SECURITY.md` with a contact address and a scope statement ("research
prototype, no security guarantees") before accepting external reports.
