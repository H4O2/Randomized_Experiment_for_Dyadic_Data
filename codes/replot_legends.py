"""Regenerate selected figures with the shared degree notation.

Run this optional script after the corresponding simulation results exist:
    python replot_legends.py

It reads ../results and updates coverage, SE-ratio, and variance figures in
../fig without drawing new random numbers.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sim_core as sc
import simulation_common as ra
from replot_main import load


# ---------------------------------------------------------------------------
#   Reuse the saved draws while updating the selected figure legends.
# ---------------------------------------------------------------------------
def main():
    """Replot the selected main, sensitivity, and Caltech variance figures."""
    out = Path(__file__).resolve().parent.parent
    save = ra.savefig

    def select(fig, where, name):
        """Save coverage and SE-ratio panels from the shared diagnostics."""
        if name.endswith(('_coverage', '_se_ratio')):
            save(fig, where, name)
        else:
            plt.close(fig)

    ra.savefig = select
    for distr in ['uniform', 'binomial', 'lognormal']:
        for sweep, xs, pars, xlabel in [
            (
                'n',
                [200, 500, 1000, 1500, 2000],
                [(n, 10, .3) for n in [200, 500, 1000, 1500, 2000]],
                'Population size',
            ),
            (
                'p',
                [.1, .2, .3, .4, .5],
                [(1000, 10, p) for p in [.1, .2, .3, .4, .5]],
                'Treatment probability',
            ),
            (
                'd',
                [4, 10, 20, 30, 40],
                [(1000, m, .3) for m in [2, 5, 10, 15, 20]],
                'Nominal average degree (2m)',
            ),
        ]:
            cells = [
                load(out / 'results' / f'main/{distr}/{sweep}/n{n}_m{m}_p{p:g}')
                for n, m, p in pars
            ]
            ra.diagnostics(cells, xs, xlabel, out, f'{distr}_{sweep}')
    ra.savefig = save

    # Sensitivity-study coverage uses the same summaries as the original run.
    zetas = [0, .01, .02, .05, .1, .2]
    cells = [load(out / 'results' / f'sensitivity/zeta{x:g}') for x in zetas]
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, k in enumerate(sc.VARS):
        y = [
            next(
                v['coverage'] for v in c['info']['variance_summary']
                if v['estimator'] == k
            )
            for c in cells
        ]
        ax.plot(zetas, y, '-o', color=ra.COLORS[j], label=ra.LABELS[k])
    ax.axhline(.95, color='black', ls='--', label='Nominal 95%')
    ax.set_ylim(0, 1.015)
    ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$')
    ax.set_ylabel('Coverage probability')
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, out, 'sensitivity_coverage')

    # Caltech plots retain the exact and Monte Carlo variance comparisons.
    for sweep, xs in [
        ('degree', [-2, -1, 0, 1, 2, 3, 5]),
        ('p', [.1, .2, .3, .4, .5]),
    ]:
        cells = [load(out / 'results' / f'variance/{sweep}/{x:g}') for x in xs]
        actual = [
            c['info']['actual_degree_change'] if sweep == 'degree' else c['p']
            for c in cells
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            actual, [sc.true_variance(c['pop'], c['p']) for c in cells],
            '-o', color='black', label='Exact variance (HT)',
        )
        ax.plot(
            actual, [c['raw']['HT'].var(ddof=1) for c in cells],
            '--x', color='gray', label='Monte Carlo variance (HT)',
        )
        ax.plot(
            actual, [c['raw']['HA'].var(ddof=1) for c in cells],
            '-s', color=ra.COLORS[4], label='Monte Carlo variance (Hajek)',
        )
        for j, k in enumerate(sc.VARS):
            ax.plot(
                actual, [c['raw'][k].mean() for c in cells],
                '-o', color=ra.COLORS[j], label='Mean ' + ra.LABELS[k],
            )
        ax.set_xlabel(
            'Change in average degree' if sweep == 'degree' else 'Treatment probability'
        )
        ax.set_ylabel('Variance')
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=10)
        save(fig, out, 'variance_vs_' + sweep)
    ra.manifest(out)


if __name__ == '__main__':
    main()
