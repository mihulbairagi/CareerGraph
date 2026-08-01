"""Command-line entrypoints.

The CLI lives in its own module, separate from the library code in
``careergraph.ingestion``, for two reasons:

1. Argument parsing, ``configure_logging`` and ``sys.exit`` are *application*
   concerns. A library that calls ``sys.exit`` or reconfigures logging on
   import is a library you cannot embed in anything else.
2. ``python -m careergraph.cli`` avoids the ``runpy`` double-import warning you
   get from running a module that its own package ``__init__`` already imported.

Exposed as the ``careergraph-ingest`` console script via ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from careergraph.config import get_settings
from careergraph.ingestion.chunker import ProfileChunker
from careergraph.ingestion.pipeline import IngestionPipeline, load_profile
from careergraph.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="careergraph-ingest",
        description="Chunk the profile document, embed it, and load it into ChromaDB.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Path to the profile JSON (default: PROFILE_PATH from settings).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the collection and re-ingest from scratch. Required after changing "
        "the chunking strategy or the embedding model, since old vectors would "
        "otherwise linger in a different vector space.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the chunks without embedding or writing anything. The fastest way "
        "to see how your profile splits up.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    try:
        if args.dry_run:
            profile = load_profile(args.profile or settings.profile_path)
            chunks = ProfileChunker(settings).chunk(profile)
            # Results on stdout, logs on stderr, so this stays pipeable.
            for chunk in chunks:
                print(f"\n=== {chunk.id}  [{chunk.section.value}] {chunk.title}")
                print(chunk.text)
            print(f"\n{len(chunks)} chunks (dry run, nothing written)")
            return 0

        report = IngestionPipeline(settings).run(profile_path=args.profile, rebuild=args.rebuild)
        print(report.summary())
        return 0
    except (FileNotFoundError, ValueError) as exc:
        # Expected, user-fixable failures: a missing profile or malformed JSON.
        # Report them as a clean message and a non-zero exit, not a traceback.
        logger.error("Ingestion failed", extra={"error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
