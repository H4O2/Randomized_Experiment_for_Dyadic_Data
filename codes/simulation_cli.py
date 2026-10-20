"""Shared command-line options for the individual simulation scripts.

Each experiment has its own readable driver. This module provides only the
common command-line plumbing; all statistical calculations remain in the core
and experiment modules.
"""

import argparse
from pathlib import Path
import time

import simulation_common as common


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
QUICK_REPLICATIONS = 5


def prepare_output(output):
    """Create the two result directories inside the selected output folder."""
    output = Path(output).resolve()
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "fig").mkdir(parents=True, exist_ok=True)
    return output


def run_script(runner, kind):
    """Parse a standalone script's options and run its experiment.

    ``--quick`` changes only the number of Monte Carlo assignments. It retains
    the full parameter grid, population-generation rules and seed labels.
    """
    default_reps = 10000 if kind == "variance" else 500
    parser = argparse.ArgumentParser(description=runner.__doc__)
    parser.add_argument(
        "--reps", type=int, default=default_reps,
        help=f"Assignments per setting (default: {default_reps}).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run five assignments per setting as an execution check.",
    )
    parser.add_argument(
        "--output", type=Path, default=PACKAGE_ROOT,
        help="Output folder containing results/ and fig/ (default: package root).",
    )
    if kind == "main":
        parser.add_argument(
            "--distr", choices=["uniform", "binomial", "lognormal", "all"],
            default="uniform", help="Coefficient distribution (default: uniform).",
        )
    args = parser.parse_args()
    reps = QUICK_REPLICATIONS if args.quick else args.reps
    if reps < 2:
        parser.error("At least two replications are required.")

    output = prepare_output(args.output)
    started = time.time()
    if kind == "main":
        distributions = (
            ["uniform", "binomial", "lognormal"]
            if args.distr == "all" else [args.distr]
        )
        runner(output, reps, distributions)
    else:
        runner(output, reps)
    common.manifest(output)
    print(
        f"Completed {kind} in {time.time() - started:.1f} seconds. "
        f"Output: {output}",
        flush=True,
    )
