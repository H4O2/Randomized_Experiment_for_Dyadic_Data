"""
run_sensitivity.py
------------------
Run the sensitivity analysis over the paper's peer-effect magnitude grid.

For each setting, save the population, individual Monte Carlo replicates,
summaries and metadata. The three figures display relative bias, mean
squared error and confidence-interval coverage.

Usage (from the codes directory):

    python run_sensitivity.py --reps 500
"""
import simulation_common as common

import numpy as np
import matplotlib.pyplot as plt

import sim_core as sc
import sensitivity_core as sens
from simulation_cli import run_script


# ---------------------------------------------------------------------------
# Experiment and saved results
# ---------------------------------------------------------------------------
def sensitivity_suites(out, reps):
    """Run every sensitivity setting on a shared fixed population."""
    pop = sens.sensitivity_population()
    zetas = [0, .01, .02, .05, .1, .2]
    p = .3
    cells = []

    for zeta in zetas:
        label = f'sensitivity/zeta{zeta:g}'
        raw = sens.run_sensitivity_setting(pop, p, reps, zeta)
        cell = common.save_cell(
            out,
            label,
            pop,
            p,
            raw,
            zeta=zeta,
            details={
                'assignment_seed': sc.seed('sensitivity/assignment'),
                'population_seed': sc.seed('sensitivity/population'),
                'graph_seed': sc.seed('sensitivity/graph'),
            },
        )
        cells.append(cell)
        print(label + ': done', flush=True)

    _plot_sensitivity(cells, zetas, out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_sensitivity(cells, zetas, out):
    """Plot the original bias, MSE and coverage summaries."""
    for metric in ['bias', 'mse', 'coverage']:
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = sc.VARS if metric == 'coverage' else sc.METHODS[:4]

        for j, k in enumerate(methods):
            if metric == 'coverage':
                y = [
                    next(
                        v['coverage']
                        for v in c['info']['variance_summary']
                        if v['estimator'] == k
                    )
                    for c in cells
                ]
            elif metric == 'bias':
                y = [
                    np.mean(c['raw'][k] - c['tau']) / c['tau'] * 100
                    for c in cells
                ]
            else:
                y = [np.mean((c['raw'][k] - c['tau']) ** 2) for c in cells]

            ax.plot(
                zetas,
                y,
                '-o',
                color=common.COLORS[j],
                label=common.LABELS[k],
            )

        if metric == 'coverage':
            ax.axhline(.95, color='black', ls='--', label='Nominal 95%')
            ax.set_ylim(0, 1.015)
        if metric == 'bias':
            ax.axhline(0, color='black', lw=.8)

        ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$')
        ax.set_ylabel({
            'bias': 'Relative bias (%)',
            'mse': 'Mean squared error',
            'coverage': 'Coverage probability',
        }[metric])
        ax.grid(alpha=.25)
        ax.legend()
        common.savefig(fig, out, 'sensitivity_' + metric)


if __name__ == '__main__':
    run_script(sensitivity_suites, kind='sensitivity')
