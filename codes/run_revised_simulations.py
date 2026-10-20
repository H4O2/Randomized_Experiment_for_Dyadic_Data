"""Main simulation studies: population size, treatment probability, and degree.

Run all three outcome distributions with:
    python run_revised_simulations.py --reps 500 --distr all

Use --distr uniform (the default) for the main-text distribution. The default
output folders are ../fig and ../results; --output selects another location.
"""

from simulation_common import diagnostics, save_cell
import sim_core as sc


# ---------------------------------------------------------------------------
#   Main studies with shared populations and assignment uniforms.
# ---------------------------------------------------------------------------
def main_suites(out, reps, distributions):
    """Run each distribution across the three parameter sweeps."""
    for distr in distributions:
        pops = {}

        def getpop(n, m):
            """Reuse a population whenever the graph parameters are repeated."""
            key = (n, m)
            if key not in pops:
                pops[key] = sc.population(
                    sc.ba(n, m, f'main/graph/{n}/{m}'),
                    distr,
                    f'main/{distr}/population/{n}/{m}',
                )
            return pops[key]

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
            cells = []
            for n, m, p in pars:
                label = f'main/{distr}/{sweep}/n{n}_m{m}_p{p:g}'
                pop = getpop(n, m)
                # Reuse assignment uniforms for all probability configurations.
                alabel = f'main/{distr}/n{n}_m{m}/assignment'
                raw = sc.run(pop, p, reps, alabel)
                cells.append(save_cell(
                    out, label, pop, p, raw,
                    details={
                        'assignment_seed': sc.seed(alabel),
                        'population_seed': sc.seed(f'main/{distr}/population/{n}/{m}'),
                        'graph_seed': sc.seed(f'main/graph/{n}/{m}'),
                        'attachment_parameter': m,
                        'nominal_average_degree': 2 * m,
                    },
                ))
                print(f'{label}: done ({reps} replications)', flush=True)
            diagnostics(cells, xs, xlabel, out, f'{distr}_{sweep}')


if __name__ == '__main__':
    from simulation_cli import run_script
    run_script(main_suites, kind='main')
