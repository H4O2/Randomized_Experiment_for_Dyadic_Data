"""Regenerate main and cluster figures from saved Monte Carlo draws.

Run this optional script after the corresponding simulation results exist:
    python replot_main.py

It reads ../results and updates ../fig without drawing new random numbers.
"""

from pathlib import Path
import json

import numpy as np

import sim_core as sc
import simulation_common as ra


# ---------------------------------------------------------------------------
#   Load a saved population, its Monte Carlo draws, and its metadata.
# ---------------------------------------------------------------------------
def load(path):
    """Reconstruct the plotting record saved by simulation_common.save_cell."""
    a = np.load(path / 'population.npz')
    pop = sc.Population(**{k: a[k] for k in a.files})
    pop.n = int(pop.n)
    info = json.loads((path / 'metadata.json').read_text())
    r = np.load(path / 'replicates.npz')
    return dict(
        pop=pop,
        p=info['treatment_probability'],
        raw={k: r[k] for k in r.files},
        tau=info['target'],
        info=info,
    )


# ---------------------------------------------------------------------------
#   Replot the three main sweeps and the cluster-design comparison.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    out = Path(__file__).resolve().parent.parent
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

    for design in ['complete', 'cluster']:
        ls = [100, 200, 300, 400, 500]
        cells = [load(out / 'results' / f'cluster/{design}/L{x}') for x in ls]
        ra.diagnostics(cells, ls, 'Number of five-node cliques', out, 'star_' + design)
        for ext in ['pdf', 'png']:
            (out / 'fig' / f'star_{design}_boxplot.{ext}').replace(
                out / 'fig' / f'star_{design}_relative_bias.{ext}'
            )
    ra.manifest(out)
