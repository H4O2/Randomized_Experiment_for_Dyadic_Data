"""Run all simulations used in the main paper and supplementary material.

Examples (from this codes directory):
    python run_all.py --quick --output ../runs/quick
    python run_all.py --reps 500 --variance-reps 10000 --output ../runs/full

The individual run_*.py scripts expose the same experiments separately. Seed
labels and computation order are preserved from the updated implementation.
"""

import os

for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import argparse
from pathlib import Path
import time

from simulation_cli import PACKAGE_ROOT, QUICK_REPLICATIONS, prepare_output
from simulation_common import manifest
from run_revised_simulations import main_suites
from run_randomization_designs import randomization_suites
from run_variance_behaviour import variance_suites
from run_sensitivity import sensitivity_suites
from run_subsample import subsample_suites
from run_cluster import cluster_suites


def main():
    """Dispatch the requested suites without changing their seed streams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["all", "main", "caltech", "variance", "sensitivity", "subsampling", "cluster"],
        default="all",
    )
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--variance-reps", type=int, default=10000)
    parser.add_argument(
        "--distributions", nargs="+", default=["uniform", "binomial", "lognormal"],
        choices=["uniform", "binomial", "lognormal"],
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run five assignments per setting, including variance settings.",
    )
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args()
    if args.quick:
        args.reps = QUICK_REPLICATIONS
        args.variance_reps = QUICK_REPLICATIONS
    if min(args.reps, args.variance_reps) < 2:
        parser.error("At least two replications are required.")

    output = prepare_output(args.output)
    started = time.time()
    if args.suite in ["all", "main"]:
        main_suites(output, args.reps, args.distributions)
    if args.suite in ["all", "caltech"]:
        randomization_suites(output, args.reps)
    if args.suite in ["all", "caltech", "variance"]:
        variance_suites(output, args.variance_reps)
    if args.suite in ["all", "sensitivity"]:
        sensitivity_suites(output, args.reps)
    if args.suite in ["all", "subsampling"]:
        subsample_suites(output, args.reps)
    if args.suite in ["all", "cluster"]:
        cluster_suites(output, args.reps)
    manifest(output)
    print(
        f"Completed {args.suite} in {time.time() - started:.1f} seconds. "
        f"Output: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
