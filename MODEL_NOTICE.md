# Model Notice — `weights/`

The nine files under `weights/` are the **trained parameters** of this project
(three PyTorch `state_dict`s and their six ONNX exports, fp32 and dynamic INT8).
They are covered by the terms below, **not** by the MIT license in `LICENSE`,
which applies to the source code, configuration, scripts and documentation only.

## Scope of the two licenses

`LICENSE` carries the unmodified MIT text.

**MIT applies to the source code.**
The trained model weights under `weights/` are governed by the separate terms
specified in `MODEL_NOTICE.md` (this file) and are **not** covered by MIT.

The `SCOPE OF THIS LICENSE` rider that used to be appended to `LICENSE` now
lives here instead, so that the repository's code license is machine-detectable
as the standard MIT template. The licensing policy itself is unchanged.

## Terms

The model parameters are released for **research, evaluation and non-commercial
use**. You may run, copy, modify and redistribute them, provided that you keep
this notice together with the files. Commercial use is **not** granted: retrain
the networks on data you have the rights to (`scripts/train/`) if you need a
commercial deployment.

They are provided **as is**, without warranty of any kind, and they are **not** a
certified or production-ready biometric system.

## Why the extra terms exist

The three networks are self-defined and self-trained — **no third-party
pretrained weights are used anywhere in this project**. The training data was:

| training data | role | rights status |
| --- | --- | --- |
| 5 face photos + 10 voice clips of the project author | the single enrolled identity (`User_Me`) | owned by the author; **not** in this repository |
| 50 face images from the Olivetti/ORL face database, fetched with `sklearn.datasets.fetch_olivetti_faces` | the "stranger" negatives | publicly distributed **for research use**, credit to AT&T Laboratories Cambridge |
| 25 synthetic noise clips (`np.random.uniform`) | the "stranger" voices | generated, no third-party rights |

Because a research-use dataset contributed to training, the derived parameters
are published under a research/evaluation notice instead of the permissive code
license. See `docs/MODEL_RELEASE.md` for the full decision record.

## Attribution

The Olivetti/ORL face images were created at AT&T Laboratories Cambridge.
"Using these images, please give credit to AT&T Laboratories Cambridge."
`THIRD_PARTY_NOTICES.md` carries the full attribution; nothing from that
database is redistributed here — the frozen script
`scripts_pc/00_setup_strangers.py` shows how the images were obtained.

## Privacy note

`face_extractor_weights.pth` and `voice_extractor_weights.pth` were trained on
**one** person's enrolment (the author, who chose to publish them). Weights
trained on face and voice data are not anonymous: they can in principle be
probed and partially inverted. Do not train and publish models on other
people's biometrics without their informed consent. See
`docs/PRIVACY_AND_SECURITY.md`.
