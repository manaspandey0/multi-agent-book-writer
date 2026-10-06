"""Command-line entry point.

Examples:
    python main.py --mock
    python main.py --chapters 3 --min-words 600 --max-words 900

Exit codes: 0 published, 2 blocked by mandatory checks, 1 failed or misconfigured.
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from config import ConfigurationError, load_settings
from models.book_models import BookBrief
from utils.logging_config import configure_logging
from workflow.book_workflow import ProgressEvent, create_book_workflow

DEFAULT_TITLE = "Pay Me on UPI: How Digital Payments Changed Small Business in India"
DEFAULT_AUDIENCE = "First-time small-business owners in India"
EXIT_CODES = {"published": 0, "blocked": 2, "failed": 1}


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="INK & EVIDENCE: research and write a short cited book with five agents."
    )
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--audience", default=DEFAULT_AUDIENCE)
    parser.add_argument("--chapters", type=int, default=3)
    parser.add_argument("--min-words", type=int, default=600)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--tone", default="friendly, clear, encouraging and practical")
    parser.add_argument(
        "--domains",
        default="npci.org.in,rbi.org.in,pib.gov.in,gov.in",
        help="Comma-separated preferred source domains.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline with the mock provider and fictional test fixtures (no API key).",
    )
    return parser.parse_args(argv)


def print_progress(event: ProgressEvent) -> None:
    print(f"[{event.elapsed_seconds:6.1f}s] {event.message}", flush=True)


def run_cli(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    overrides = {"llm_provider": "mock"} if arguments.mock else {}
    try:
        settings = load_settings(**overrides)
        configure_logging(settings.log_level)
        brief = BookBrief(
            title=arguments.title,
            audience=arguments.audience,
            chapter_count=arguments.chapters,
            min_words_per_chapter=arguments.min_words,
            max_words_per_chapter=arguments.max_words,
            tone=arguments.tone,
            preferred_source_domains=arguments.domains.split(","),
        )
        workflow = create_book_workflow(settings, progress_callback=print_progress)
    except (ConfigurationError, ValidationError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 1

    if settings.is_mock_mode:
        print("Mock mode: offline provider and fictional test fixtures. Output is not research.")
    result = workflow.generate_book(brief)

    print()
    print(f"Run {result.state.run_id}: {result.publication_status.upper()}")
    for chapter in result.state.chapters:
        print(
            f"  Chapter {chapter.plan.chapter_number}: {chapter.status.value}, "
            f"{chapter.word_count or 0} words, {chapter.revision_count} revision(s)"
            + (f" — {chapter.failure_reason}" if chapter.failure_reason else "")
        )
    for error in result.state.errors:
        print(f"  Error: {error}", file=sys.stderr)
    print(f"Artifacts saved in {result.run_dir}")
    for label, path in result.saved_files.items():
        print(f"  {label}: {path.name}")
    return EXIT_CODES[result.publication_status]


if __name__ == "__main__":
    sys.exit(run_cli())
