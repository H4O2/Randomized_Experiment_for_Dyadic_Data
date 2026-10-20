"""
run_subsample.py
----------------
Run dyad Poisson sampling, unit Poisson sampling and matching experiments.

Save the full replicate arrays, global-target summaries, selected-pair
matching summaries and two robustness figures. The sensitivity and
subsampling experiments use the same graph, coefficients and assignments.

Usage (from the codes directory):

    python run_subsample.py --reps 500
"""
import simulation_common as common

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sim_core as sc
import sensitivity_core as sens
import subsample_core as sub
from simulation_cli import run_script


# ---------------------------------------------------------------------------
# Experiment and saved results
# ---------------------------------------------------------------------------
def subsample_suites(out, reps):
    """Run the fixed subsampling experiment and save its results."""
    pop = sens.sensitivity_population()
    p = .3
    zetas = [0, .01, .02, .05, .1, .2]
    pis = [.25, .5, .75, 1.]
    raw = sub.run_subsampling(pop, p, reps, zetas, pis)

    dest = out / 'results' / 'subsampling'
    dest.mkdir(parents=True, exist_ok=True)
    pop.save(dest / 'population.npz')
    np.savez_compressed(dest / 'replicates.npz', **raw)
    _save_summaries(dest, raw, reps)

    common.write_json(dest / 'metadata.json', {
        'n': pop.n,
        'replications': reps,
        'assignment_seed': sc.seed('sensitivity/assignment'),
        'sampling_seed': sc.seed('subsample/sampling'),
        'matching_seed': sc.seed('subsample/matching'),
        'zeta': zetas,
        'pi': pis,
        'matching_target': (
            'Mean full-treatment minus full-control contrast of the selected '
            'unordered pairs; not the global per-unit target'
        ),
        'empty_dyad_samples': int(raw['dyad_empty'].sum()),
        'empty_unit_samples': int(raw['unit_empty'].sum()),
        'mean_matching_size': float(raw['matching_size'].mean()),
    })

    _plot_subsampling(raw, reps, out)
    print('subsampling: done', flush=True)


def _save_summaries(dest, raw, reps):
    """Write bias and MSE summaries using each estimator's own target."""
    full = raw['full_HT']
    dyad = raw['dyad_HT']
    unit = raw['unit_HT']
    match = raw['matching']
    mtarget = raw['matching_target']
    target = raw['global_target']
    zetas = raw['zeta']
    pis = raw['pi']

    rows = []
    for j, zeta in enumerate(zetas):
        for k, pi in enumerate(pis):
            for name, arr in [
                ('Full HT', full[:, j]),
                ('Dyad Poisson HT', dyad[:, j, k]),
                ('Unit Poisson HT', unit[:, j, k]),
            ]:
                err = arr - target[j]
                rows.append(dict(
                    zeta=zeta,
                    pi=pi,
                    method=name,
                    target=target[j],
                    bias=err.mean(),
                    relative_bias=err.mean() / target[j],
                    mse=np.mean(err ** 2),
                    bias_mcse=err.std(ddof=1) / np.sqrt(reps),
                    mse_mcse=(err ** 2).std(ddof=1) / np.sqrt(reps),
                ))
    pd.DataFrame(rows).to_csv(dest / 'summary.csv', index=False)

    mrows = []
    for j, zeta in enumerate(zetas):
        err = match[:, j] - mtarget[:, j]
        mrows.append(dict(
            zeta=zeta,
            mean_selected_target=mtarget[:, j].mean(),
            bias=err.mean(),
            mse=np.mean(err ** 2),
            bias_mcse=err.std(ddof=1) / np.sqrt(reps),
        ))
    pd.DataFrame(mrows).to_csv(dest / 'matching_summary.csv', index=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_subsampling(raw, reps, out):
    """Plot global-target sampling errors and selected-pair matching errors."""
    full = raw['full_HT']
    dyad = raw['dyad_HT']
    unit = raw['unit_HT']
    match = raw['matching']
    mtarget = raw['matching_target']
    target = raw['global_target']
    zetas = raw['zeta']
    pis = raw['pi']

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    k = pis.index(.5)
    for h, (name, a) in enumerate([
        ('Full HT', full),
        ('Dyad Poisson HT', dyad[:, :, k]),
        ('Unit Poisson HT', unit[:, :, k]),
    ]):
        err = a - target
        axs[0].plot(
            zetas, err.mean(axis=0), '-o', color=common.COLORS[h], label=name,
        )
        axs[1].plot(
            zetas, (err ** 2).mean(axis=0), '-o',
            color=common.COLORS[h], label=name,
        )
    for ax in axs:
        ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$')
        ax.grid(alpha=.25)
        ax.legend()
    axs[0].set_ylabel('Bias for global target')
    axs[1].set_ylabel('MSE for global target')
    common.savefig(fig, out, 'subsampling_robustness')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    err = match - mtarget
    axs[0].errorbar(
        zetas,
        err.mean(0),
        yerr=1.96 * err.std(0, ddof=1) / np.sqrt(reps),
        fmt='-o',
    )
    axs[0].set_ylabel('Bias for selected-pair target')
    axs[0].axhline(0, color='black', lw=.8)
    axs[1].plot(zetas, np.mean(err ** 2, axis=0), '-o')
    axs[1].set_ylabel('MSE for selected-pair target')
    for ax in axs:
        ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$')
        ax.grid(alpha=.25)
    common.savefig(fig, out, 'matching_selected_target')


if __name__ == '__main__':
    run_script(subsample_suites, kind='subsampling')
