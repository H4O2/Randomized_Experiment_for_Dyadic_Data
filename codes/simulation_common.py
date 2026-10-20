"""Shared output summaries and figure styling for the simulation drivers.

The named simulation scripts contain the individual experiments. This module
keeps their saved results, diagnostic figures, and environment records uniform.
"""

import os

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')

import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

import sim_core as sc


# ---------------------------------------------------------------------------
#   Figure labels and styling shared by all simulation studies.
# ---------------------------------------------------------------------------
COLORS = ['#2171b5', '#e6550d', '#31a354', '#756bb1', '#636363']
LABELS = {
    'HT': r'$\widehat{\tau}_{HT}$',
    'HA': r'$\widehat{\tau}_{H\acute{a}jek}$',
    'DM': 'Unit aggregate',
    'TR': 'Exposure threshold',
    'PI2': 'SNIPE(2) / 2',
    'V1': r'$\widehat{\sigma}^{2}$',
    'V2_avg': r'$\widehat{\sigma}^{2}(d_1^{\mathcal{G}})$',
    'V2_max': r'$\widehat{\sigma}^{2}(D_{\mathcal{G}})$',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'axes.labelsize': 15,
    'legend.fontsize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
#   Save the population, individual Monte Carlo draws, and summaries.
# ---------------------------------------------------------------------------
def write_json(path, d):
    """Write a JSON record, rejecting non-finite floating-point values."""
    path.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')


def savefig(fig, out, name):
    """Save the same figure as PDF and PNG, then release its resources."""
    fig.tight_layout()
    fig.savefig(out / 'fig' / f'{name}.pdf')
    fig.savefig(out / 'fig' / f'{name}.png', dpi=160)
    plt.close(fig)


def save_cell(out, label, pop, p, raw, design='Bernoulli', zeta=0., details=None):
    """Save one simulation setting and return its data for plotting.

    Point-estimator summaries use the population's target. Coverage summaries
    use the HT estimate and the nonnegative part of each variance estimate.
    """
    dest = out / 'results' / label
    dest.mkdir(parents=True, exist_ok=True)
    pop.save(dest / 'population.npz')
    np.savez_compressed(dest / 'replicates.npz', **raw)
    pd.DataFrame(raw).to_csv(
        dest / 'replicates.csv', index=False, float_format='%.15g'
    )

    tau = pop.target(zeta)
    b = len(raw['HT'])
    summ = []
    for k in sc.METHODS:
        if k not in raw or np.isnan(raw[k]).all():
            continue
        err = raw[k] - tau
        summ.append(dict(
            method=k,
            target=tau,
            mean=float(raw[k].mean()),
            bias=float(err.mean()),
            relative_bias=float(err.mean() / tau),
            mse=float(np.mean(err * err)),
            sd=float(raw[k].std(ddof=1)),
            bias_mcse=float(err.std(ddof=1) / np.sqrt(b)),
            mse_mcse=float((err * err).std(ddof=1) / np.sqrt(b)),
        ))

    varsum = []
    for k in sc.VARS:
        if k not in raw:
            continue
        covered = (
            np.abs(raw['HT'] - tau)
            <= 1.959963984540054 * np.sqrt(np.maximum(raw[k], 0))
        )
        cov = float(covered.mean())
        varsum.append(dict(
            estimator=k,
            mean=float(raw[k].mean()),
            coverage=cov,
            coverage_mcse=float(np.sqrt(cov * (1 - cov) / b)),
            negative_count=int((raw[k] < 0).sum()),
            se_ratio=float(
                np.sqrt(max(raw[k].mean(), 0)) / raw['HT'].std(ddof=1)
            ),
        ))

    info = dict(
        label=label,
        n=pop.n,
        undirected_edges=pop.m,
        ordered_edges=2 * pop.m,
        average_degree=float(pop.degree.mean()),
        max_degree=int(pop.degree.max()),
        degree_second_moment=float(np.mean(pop.degree ** 2)),
        treatment_probability=p,
        design=design,
        zeta=zeta,
        target=tau,
        replications=b,
        master_seed=sc.MASTER_SEED,
        empty_arm_count=int(raw['empty_arm'].sum()),
        empty_threshold_count=int(raw['empty_threshold'].sum()),
        point_summary=summ,
        variance_summary=varsum,
    )
    if design == 'Bernoulli' and zeta == 0:
        info['exact_HT_variance'] = sc.true_variance(pop, p)
    if details:
        info.update(details)

    write_json(dest / 'metadata.json', info)
    pd.DataFrame(summ).to_csv(dest / 'summary.csv', index=False)
    return dict(pop=pop, p=p, raw=raw, tau=tau, info=info)


# ---------------------------------------------------------------------------
#   Common diagnostic plots for a sequence of simulation settings.
# ---------------------------------------------------------------------------
def diagnostics(cells, xs, xlabel, out, prefix):
    """Plot relative errors, MSE, coverage, and estimated-SE diagnostics."""
    pos = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(9, 5))
    handles = []
    for j, k in enumerate(sc.METHODS[:4]):
        a = [(c['raw'][k] - c['tau']) / c['tau'] * 100 for c in cells]
        box = ax.boxplot(
            a,
            positions=pos + (j - 1.5) * .18,
            widths=.17,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box['boxes']:
            patch.set_facecolor(COLORS[j])
            patch.set_alpha(.75)
        for med in box['medians']:
            med.set_color('black')
        handles.append(Line2D([0], [0], color=COLORS[j], lw=5, label=LABELS[k]))
    ax.set_xticks(pos, xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Relative estimation error (%)')
    ax.axhline(0, color='black', lw=.8)
    ax.grid(axis='y', alpha=.25)
    ax.legend(handles=handles, ncol=2)
    savefig(fig, out, prefix + '_boxplot')

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, k in enumerate(sc.METHODS[:4]):
        ax.plot(
            xs,
            [np.mean((c['raw'][k] - c['tau']) ** 2) for c in cells],
            '-o', color=COLORS[j], label=LABELS[k],
        )
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean squared error')
    ax.grid(alpha=.25)
    ax.legend()
    savefig(fig, out, prefix + '_mse')

    if 'V1' in cells[0]['raw']:
        for metric, ylabel in [
            ('coverage', 'Coverage probability'),
            ('se_ratio', 'RMS estimated SE / Monte Carlo SD'),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for j, k in enumerate(sc.VARS):
                yy = [
                    next(
                        v[metric] for v in c['info']['variance_summary']
                        if v['estimator'] == k
                    )
                    for c in cells
                ]
                ax.plot(xs, yy, '-o', color=COLORS[j], label=LABELS[k])
            ax.axhline(
                .95 if metric == 'coverage' else 1,
                color='black', ls='--',
                label='Nominal 95%' if metric == 'coverage' else 'Reference 1',
            )
            ax.set_xticks(xs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
            ax.legend()
            if metric == 'coverage':
                ax.set_ylim(
                    min(
                        .85,
                        min(
                            v['coverage'] for c in cells
                            for v in c['info']['variance_summary']
                        ) - .025,
                    ),
                    1.015,
                )
            savefig(fig, out, prefix + '_' + metric)

    if not np.isnan(cells[0]['raw']['PI2']).all():
        fig, ax = plt.subplots(figsize=(7, 5))
        for j, k in enumerate(sc.METHODS):
            ax.plot(
                xs,
                [np.mean((c['raw'][k] - c['tau']) ** 2) for c in cells],
                '-o', color=COLORS[j], label=LABELS[k],
            )
        ax.set_yscale('log')
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean squared error (log scale)')
        ax.grid(alpha=.25)
        ax.legend()
        savefig(fig, out, prefix + '_snipe2_mse')


# ---------------------------------------------------------------------------
#   Record software versions and generated files for reproducibility.
# ---------------------------------------------------------------------------
def manifest(out):
    """Record the environment and checksums of code, figures, and results."""
    versions = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        networkx=nx.__version__,
        matplotlib=matplotlib.__version__,
        master_seed=sc.MASTER_SEED,
    )
    write_json(out / 'results' / 'versions.json', versions)
    files = []
    for folder in ['codes', 'fig', 'results']:
        for p in sorted((out / folder).rglob('*')):
            if (
                p.is_file()
                and p.name != 'manifest.json'
                and '__pycache__' not in p.parts
            ):
                files.append(dict(
                    path=str(p.relative_to(out)),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    bytes=p.stat().st_size,
                ))
    write_json(out / 'manifest.json', {'master_seed': sc.MASTER_SEED, 'files': files})
