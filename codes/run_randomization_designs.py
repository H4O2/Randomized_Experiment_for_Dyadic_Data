"""Compare Bernoulli and complete randomization on the Caltech network.

Run the manuscript configuration with:
    python run_randomization_designs.py --reps 500

The two designs use the same fixed population and retain their original,
separately seeded assignment streams.
"""

from pathlib import Path

from simulation_common import COLORS, LABELS, save_cell, savefig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import sim_core as sc


# ---------------------------------------------------------------------------
#   Caltech experiments under the two individual-level assignment designs.
# ---------------------------------------------------------------------------
def randomization_suites(out, reps):
    """Run both designs across treatment probabilities and save paired plots."""
    graph = sc.load_caltech(Path(__file__).with_name('socfb-Caltech36.mtx'))
    pop = sc.population(graph, 'uniform', 'caltech/population')
    probs = [.1, .2, .3, .4, .5]

    for design in ['Bernoulli', 'complete']:
        cells = []
        for p in probs:
            label = f'caltech/{design}/p{p:g}'
            alabel = f'caltech/{design}/assignment'
            raw = sc.run(pop, p, reps, alabel, design)
            cells.append(save_cell(
                out, label, pop, p, raw, design,
                details={
                    'assignment_seed': sc.seed(alabel),
                    'population_seed': sc.seed('caltech/population'),
                    'actual_treatment_fraction': (
                        np.floor(p * pop.n) / pop.n if design == 'complete' else p
                    ),
                    'treated_count': (
                        int(np.floor(p * pop.n)) if design == 'complete' else None
                    ),
                },
            ))
            print(label + ': done', flush=True)

        # The manuscript uses a single two-panel figure for each design.
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        pos = np.arange(len(probs))
        handles = []
        for j, k in enumerate(sc.METHODS[:4]):
            vals = [(c['raw'][k] - c['tau']) / c['tau'] * 100 for c in cells]
            b = axs[0].boxplot(
                vals,
                positions=pos + (j - 1.5) * .18,
                widths=.17,
                patch_artist=True,
                showfliers=False,
            )
            for patch in b['boxes']:
                patch.set_facecolor(COLORS[j])
                patch.set_alpha(.75)
            for med in b['medians']:
                med.set_color('black')
            handles.append(Line2D(
                [0], [0], color=COLORS[j], lw=4, label=LABELS[k]
            ))
            axs[1].plot(
                probs,
                [np.mean((c['raw'][k] - c['tau']) ** 2) for c in cells],
                '-o', color=COLORS[j], label=LABELS[k],
            )

        axs[0].set_xticks(pos, probs)
        axs[0].set_ylabel('Relative estimation error (%)')
        axs[0].axhline(0, color='black', lw=.8)
        axs[0].legend(handles=handles, ncol=2)
        axs[1].set_ylabel('Mean squared error')
        axs[1].legend()
        for ax in axs:
            ax.set_xlabel('Nominal treatment probability')
            ax.grid(axis='y', alpha=.25)
        savefig(fig, out, design)


if __name__ == '__main__':
    from simulation_cli import run_script
    run_script(randomization_suites, kind='randomization')
