"""Allow ``python -m multimodal_auth`` to run the CLI."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
