"""Command-line interface for the Speak Easy framework."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("speakeasy")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _model_type(value: str) -> str:
    """Validate that the input follows ``source:model_name`` format."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid format '{value}'. Expected 'source:model_name' "
            "(e.g., 'openai:gpt-4o', 'vllm:meta-llama/Llama-3.3-70B-Instruct')."
        )
    return value


FRAMEWORK_CHOICES = [
    "baseline_dr",
    "baseline_gcg",
    "baseline_tap",
    "speakeasy_dr",
    "speakeasy_gcg",
    "speakeasy_tap",
]

SCORER_CHOICES = ["gpt4judge", "harmscore"]


def cmd_run(args: argparse.Namespace) -> None:
    """Run attack frameworks on input data."""
    from tqdm import tqdm

    from speakeasy.backends import get_backend
    from speakeasy.frameworks import get_framework

    if not os.path.exists(args.data_dir):
        logger.error("Data file not found: %s", args.data_dir)
        sys.exit(1)

    with open(args.data_dir) as f:
        data = json.load(f)

    model = get_backend(args.model)

    for framework_name in tqdm(args.frameworks, ascii=True, desc="Frameworks"):
        save_dir = os.path.join(
            args.save_dir,
            os.path.basename(os.path.dirname(args.data_dir)),
            framework_name,
            args.model,
        )
        os.makedirs(save_dir, exist_ok=True)

        framework = get_framework(framework_name, model=model, device=args.device)
        framework.infer(data, save_dir)


def cmd_score(args: argparse.Namespace) -> None:
    """Score query-response pairs using an evaluation model."""
    from speakeasy.evaluators import get_evaluator
    from speakeasy.utils import check_format

    if not check_format(args.data_dir):
        logger.error("Data file '%s' is not properly formatted.", args.data_dir)
        sys.exit(1)

    evaluator = get_evaluator(args.scorer)

    with open(args.data_dir) as f:
        data = json.load(f)

    scored = evaluator.compute_scores(data)

    os.makedirs(os.path.dirname(args.save_dir) or ".", exist_ok=True)
    with open(args.save_dir, "w") as f:
        json.dump(scored, f, indent=4)

    logger.info("Saved scored results to %s", args.save_dir)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="speakeasy",
        description="Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- `speakeasy run` ---
    run_parser = subparsers.add_parser("run", help="Run attack frameworks.")
    run_parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the input JSON file.",
    )
    run_parser.add_argument(
        "--save-dir",
        type=str,
        default="results/",
        help="Directory to save results.",
    )
    run_parser.add_argument(
        "--model",
        type=_model_type,
        required=True,
        help="Model in 'source:name' format (e.g., 'vllm:meta-llama/Llama-3.3-70B-Instruct').",
    )
    run_parser.add_argument(
        "--frameworks",
        type=str,
        nargs="+",
        required=True,
        choices=FRAMEWORK_CHOICES,
        help="Framework(s) to run.",
    )
    run_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index.",
    )

    # --- `speakeasy score` ---
    score_parser = subparsers.add_parser("score", help="Score query-response pairs.")
    score_parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the JSON file with query-response pairs.",
    )
    score_parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to save the scored output JSON.",
    )
    score_parser.add_argument(
        "--scorer",
        type=str,
        required=True,
        choices=SCORER_CHOICES,
        help="Evaluation method.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "score":
        cmd_score(args)
    else:
        parser.print_help()
        sys.exit(1)
