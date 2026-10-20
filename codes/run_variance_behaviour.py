"""Variance-estimator behaviour on perturbed Caltech networks.

Run the manuscript configuration with:
    python run_variance_behaviour.py --reps 10000

The degree and probability sweeps share the original coefficient arrays and
assignment uniforms. Network perturbations are nested within each direction.
"""

from pathlib import Path

from simulation_common import COLORS, LABELS, save_cell, savefig
import matplotlib.pyplot as plt
import numpy as np

import sim_core as sc


# ---------------------------------------------------------------------------
#   Variance study: change the degree or the treatment probability.
# ---------------------------------------------------------------------------
def variance_suites(out, variance_reps):
    """Run the two variance sweeps with the original named random streams."""
    graph = sc.load_caltech(Path(__file__).with_name('socfb-Caltech36.mtx'))
    pop = sc.population(graph, 'uniform', 'caltech/population')
    probs = [.1, .2, .3, .4, .5]

    # Use the same coefficients on common ordered edges across perturbations.
    rng = np.random.default_rng(sc.seed('caltech/variance/coefficients'))
    n = pop.n
    coefficients = [
        rng.uniform(a, b, (n, n)) for a, b in [(0, 1), (0, .5), (-.5, 0), (0, .5)]
    ]
    original = set(tuple(sorted(e)) for e in graph.edges())
    edges = sorted(original)
    rng = np.random.default_rng(sc.seed('caltech/variance/perturbation'))
    remove_order = rng.permutation(len(edges))
    missing = [
        (i, j) for i in range(n) for j in range(i + 1, n)
        if (i, j) not in original
    ]
    add_order = rng.permutation(len(missing))
    basepop = sc.population(graph, 'uniform', 'unused', coefficients=coefficients)

    for sweep, xs in [('degree', [-2, -1, 0, 1, 2, 3, 5]), ('p', probs)]:
        cells = []
        actual = []
        for x in xs:
            g = graph.copy()
            p = .3 if sweep == 'degree' else x
            if sweep == 'degree':
                k = round(abs(x) * n / 2)
                if x < 0:
                    g.remove_edges_from(edges[i] for i in remove_order[:k])
                elif x > 0:
                    g.add_edges_from(missing[i] for i in add_order[:k])
            pp = (
                sc.population(g, 'uniform', 'unused', coefficients=coefficients)
                if sweep == 'degree' else basepop
            )
            label = f'variance/{sweep}/{x:g}'
            alabel = 'caltech/variance/assignment'
            raw = sc.run(pp, p, variance_reps, alabel)
            c = save_cell(
                out, label, pp, p, raw,
                details={
                    'assignment_seed': sc.seed(alabel),
                    'coefficient_seed': sc.seed('caltech/variance/coefficients'),
                    'perturbation_seed': sc.seed('caltech/variance/perturbation'),
                    'requested_degree_change': x if sweep == 'degree' else 0,
                    'actual_degree_change': float(
                        pp.degree.mean() - basepop.degree.mean()
                    ),
                },
            )
            cells.append(c)
            actual.append(
                float(pp.degree.mean() - basepop.degree.mean())
                if sweep == 'degree' else p
            )
            print(label + ': done', flush=True)

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
            '-s', color=COLORS[4], label='Monte Carlo variance (Hajek)',
        )
        for j, k in enumerate(sc.VARS):
            ax.plot(
                actual, [c['raw'][k].mean() for c in cells],
                '-o', color=COLORS[j], label='Mean ' + LABELS[k],
            )
        ax.set_xlabel(
            'Change in average degree' if sweep == 'degree' else 'Treatment probability'
        )
        ax.set_ylabel('Variance')
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=10)
        savefig(fig, out, 'variance_vs_' + sweep)


if __name__ == '__main__':
    from simulation_cli import run_script
    run_script(variance_suites, kind='variance')
