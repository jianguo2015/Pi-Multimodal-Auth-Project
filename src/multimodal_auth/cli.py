"""Command line interface (``python -m multimodal_auth`` / ``scripts/infer.py``).

Exit codes
----------
0  inference completed
2  input problem (missing / unreadable / malformed image or audio)
3  model artefact problem (missing file, onnxruntime too old, ...)
4  configuration problem
5  REJECT while ``--fail-on-reject`` was requested
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config.loader import Config, load_config
from .decision.policy import validate_threshold
from .errors import ConfigError, InvalidInputError, ModelArtifactError, SafetyError
from .pipeline.pipeline import VALID_BACKENDS, MultimodalPipeline
from .reporting import print_report
from .safety import portable_path
from .safety import assert_not_protected

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_ARTIFACT = 3
EXIT_CONFIG = 4
EXIT_REJECT = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-auth",
        description="Face + voice identity verification prototype (attention-gated fusion).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--face", required=True, help="path to a face image (jpg/png)")
    parser.add_argument("--voice", required=True, help="path to a voice clip (wav/flac)")
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--backend",
        default=None,
        choices=VALID_BACKENDS,
        help="inference backend; default comes from the config file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the accept threshold; default comes from the config file",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON only")
    parser.add_argument("--output", default=None, help="also write the result JSON to this path")
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="exit with code 5 when the decision is REJECT",
    )
    parser.add_argument("--show-config", action="store_true", help="print the effective config and exit")
    return parser


def _run(args) -> int:
    if args.show_config:
        config = load_config(path=args.config)
        print(json.dumps(config.as_dict(), indent=2, ensure_ascii=False))
        return EXIT_OK

    config: Config = load_config(path=args.config)
    threshold = args.threshold
    if threshold is not None:
        validate_threshold(threshold)

    pipeline = MultimodalPipeline(config, backend=args.backend, threshold=threshold)
    result = pipeline.run(args.face, args.voice)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print_report(result, portable_path(config.source, config.root))

    if args.output:
        target = assert_not_protected(Path(args.output), config.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not args.json:
            print("Result written to %s" % target)

    if args.fail_on_reject and not result.accepted:
        return EXIT_REJECT
    return EXIT_OK


def run_infer(argv=None) -> int:
    """Entry point shared by ``python -m multimodal_auth`` and the CLI scripts."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.show_config and (args.face is None or args.voice is None):
        parser.error("--face and --voice are required")

    try:
        return _run(args)
    except ConfigError as exc:
        print("configuration error: %s" % exc, file=sys.stderr)
        return EXIT_CONFIG
    except ModelArtifactError as exc:
        print("model artefact error: %s" % exc, file=sys.stderr)
        return EXIT_ARTIFACT
    except InvalidInputError as exc:
        print("input error: %s" % exc, file=sys.stderr)
        return EXIT_INPUT
    except SafetyError as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return EXIT_ARTIFACT


def main(argv=None) -> int:
    return run_infer(argv)


__all__ = ["build_parser", "main", "run_infer", "EXIT_OK", "EXIT_INPUT",
           "EXIT_ARTIFACT", "EXIT_CONFIG", "EXIT_REJECT"]

