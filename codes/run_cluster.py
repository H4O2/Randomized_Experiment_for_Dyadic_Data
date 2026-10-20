"""Compare complete and cluster randomization on the clique-and-hub graphs.

Run the manuscript configuration with:
    python run_cluster.py --reps 500

Each population has five-node cliques connected to a shared hub. The planted
cliques and singleton hub form the explicitly defined treatment clusters.
"""

from simulation_common import diagnostics, save_cell
import networkx as nx
import numpy as np

import sim_core as sc


# ---------------------------------------------------------------------------
#   Assignment-design comparison on planted clusters.
# ---------------------------------------------------------------------------
def cluster_suites(out, reps):
    """Build each clustered population and apply both assignment designs."""
    complete = []
    cluster = []
    ls = [100, 200, 300, 400, 500]
    for ell in ls:
        n = 5 * ell + 1
        hub = n - 1
        g = nx.Graph()
        g.add_nodes_from(range(n))
        rng = np.random.default_rng(sc.seed(f'cluster/graph/{ell}'))
        for k in range(ell):
            nodes = list(range(5 * k, 5 * k + 5))
            g.add_edges_from((i, j) for i in nodes for j in nodes if i < j)
            g.add_edge(hub, int(rng.choice(nodes)))
        ids = np.r_[np.repeat(np.arange(ell), 5), ell]
        pop = sc.population(g, 'cluster', f'cluster/population/{ell}', ids)

        for design, cells in [('complete', complete), ('cluster', cluster)]:
            label = f'cluster/{design}/L{ell}'
            alabel = label + '/assignment'
            n1 = int(np.floor(5 * ell / 3))
            raw = sc.run(pop, 1 / 3, reps, alabel, design, n1=n1)
            cells.append(save_cell(
                out, label, pop, 1 / 3, raw, design,
                details={
                    'L': ell,
                    'assignment_seed': sc.seed(alabel),
                    'population_seed': sc.seed(f'cluster/population/{ell}'),
                    'graph_seed': sc.seed(f'cluster/graph/{ell}'),
                    'partition': (
                        'Planted five-node cliques plus singleton hub '
                        '(new explicit implementation; no Leiden)'
                    ),
                    'treated_count': n1 if design == 'complete' else None,
                    'actual_treatment_fraction': n1 / n if design == 'complete' else 1 / 3,
                },
            ))
            print(label + ': done', flush=True)

    for design, cells in [('complete', complete), ('cluster', cluster)]:
        diagnostics(cells, ls, 'Number of five-node cliques', out, 'star_' + design)
        # Preserve the included legacy filename; its ylabel correctly says error.
        for ext in ['pdf', 'png']:
            (out / 'fig' / f'star_{design}_boxplot.{ext}').rename(
                out / 'fig' / f'star_{design}_relative_bias.{ext}'
            )


if __name__ == '__main__':
    from simulation_cli import run_script
    run_script(cluster_suites, kind='cluster')
