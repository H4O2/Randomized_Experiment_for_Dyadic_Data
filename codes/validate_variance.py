"""
validate_variance.py
--------------------
Optional independent 20,000-draw check of the Caltech HT variance.

Run after the main simulation results exist:

    python validate_variance.py

All estimates are saved in results/independent_variance_validation.*. This
check does not replace publication simulation cells or figures. It compares
the empirical Monte Carlo variance with its exact finite-population value.
"""

import os

# Set thread limits before importing numerical libraries.
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')

from pathlib import Path
import time, json

import numpy as np
import pandas as pd

import sim_core as sc


# ---------------------------------------------------------------------------
#   Independent simulation and variance comparison
# ---------------------------------------------------------------------------
def main():
    """Load the fixed Caltech population, simulate, and save the comparison."""
    root = Path(__file__).resolve().parent.parent
    a = np.load(root / 'results/variance/p/0.5/population.npz')
    pop = sc.Population(**{k: a[k] for k in a.files})
    pop.n = int(pop.n)

    # Use a separate assignment stream from the publication simulations.
    label = 'independent_variance_validation'
    t0 = time.time()
    raw = sc.run(pop, .5, 20000, label)

    exact = sc.true_variance(pop, .5)
    mc = raw['HT'].var(ddof=1)

    # Approximate Monte Carlo SE of the empirical variance, based on the
    # dispersion of squared deviations across independent assignments.
    mcse = (
        np.std((raw['HT'] - raw['HT'].mean())**2, ddof=1)
        / np.sqrt(len(raw['HT']))
    )
    result = dict(
        replications=20000,
        seed=sc.seed(label),
        assignment_seed_label=label,
        population_file='results/variance/p/0.5/population.npz',
        exact_variance=exact,
        mc_variance=mc,
        variance_mcse_approx=mcse,
        z=(mc - exact) / mcse,
        elapsed=time.time() - t0,
    )

    # Save both the replicate-level values and a compact numerical summary.
    dest = root / 'results' / 'independent_variance_validation'
    np.savez_compressed(dest.with_suffix('.npz'), **raw)
    pd.DataFrame(raw).to_csv(
        dest.with_suffix('.csv'),
        index=False,
        float_format='%.15g',
    )
    dest.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
#   Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
