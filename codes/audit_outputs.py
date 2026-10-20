"""
audit_outputs.py
----------------
Check completed results without drawing new simulation assignments.

This script writes all_point_summaries.csv and all_variance_summaries.csv
in the package's results directory, then prints summary diagnostics and
checks the saved replicate arrays. It expects the complete set of suites.

Usage (from the codes directory):

    python audit_outputs.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Read completed cells and write the combined summary tables
# ---------------------------------------------------------------------------
rows = []
negative = []
for meta in sorted((ROOT / 'results').rglob('metadata.json')):
    info = json.loads(meta.read_text())
    if 'point_summary' not in info:
        continue

    raw = np.load(meta.with_name('replicates.npz'))
    pop = np.load(meta.with_name('population.npz'))
    assert len(raw['HT']) == info['replications']
    for k in ['HT', 'HA', 'DM', 'TR']:
        assert np.isfinite(raw[k]).all(), str(meta) + k
    if info['design'] == 'complete':
        assert np.allclose(raw['HT'], raw['HA'], atol=1e-12)

    for item in info['point_summary']:
        rows.append({
            'label': info['label'],
            'n': info['n'],
            'p': info['treatment_probability'],
            'degree': info['average_degree'],
            'max_degree': info['max_degree'],
            **item,
        })
    for item in info['variance_summary']:
        negative.append({'label': info['label'], **item})

frame = pd.DataFrame(rows)
frame.to_csv(ROOT / 'results' / 'all_point_summaries.csv', index=False)
vframe = pd.DataFrame(negative)
vframe.to_csv(ROOT / 'results' / 'all_variance_summaries.csv', index=False)


# ---------------------------------------------------------------------------
# Print point-estimation and variance-estimation summaries
# ---------------------------------------------------------------------------
main = frame[frame.label.str.startswith('main/')]
winners = main.loc[main.groupby('label').mse.idxmin()][['label', 'method', 'mse']]
print('Main MSE winners:\n' + winners.to_string(index=False))
print(
    '\nMain variance ranges:\n'
    + vframe[vframe.label.str.startswith('main/')]
    .groupby('estimator')[['coverage', 'se_ratio', 'negative_count']]
    .agg(['min', 'max'])
    .to_string()
)
print(
    '\nSensitivity:\n'
    + frame[frame.label.str.startswith('sensitivity/')][
        ['label', 'method', 'bias', 'mse', 'bias_mcse']
    ].to_string(index=False)
)
print(
    '\nCluster:\n'
    + frame[frame.label.str.startswith('cluster/')][
        ['label', 'method', 'bias', 'mse', 'bias_mcse']
    ].to_string(index=False)
)


# ---------------------------------------------------------------------------
# Compare Monte Carlo variance with the exact Bernoulli variance
# ---------------------------------------------------------------------------
for path in (ROOT / 'results' / 'variance').rglob('metadata.json'):
    info = json.loads(path.read_text())
    a = np.load(path.with_name('replicates.npz'))['HT']
    truth = info['exact_HT_variance']
    mc = a.var(ddof=1)
    se = np.std((a - a.mean()) ** 2, ddof=1) / np.sqrt(len(a))
    print(
        'variance QA', info['label'],
        'exact', truth,
        'MC', mc,
        'MCSE approx', se,
        'z', (mc - truth) / se,
    )


# ---------------------------------------------------------------------------
# Check the full-sample identities and the sensitivity expectation
# ---------------------------------------------------------------------------
a = np.load(ROOT / 'results' / 'subsampling' / 'replicates.npz')
assert np.allclose(a['full_HT'], a['dyad_HT'][:, :, -1])
assert np.allclose(a['full_HT'], a['unit_HT'][:, :, -1])
assert np.isfinite(a['matching']).all()

for path in (ROOT / 'results' / 'sensitivity').rglob('metadata.json'):
    info = json.loads(path.read_text())
    a = np.load(path.with_name('replicates.npz'))
    pop = np.load(path.with_name('population.npz'))
    degree = np.bincount(
        np.r_[pop['u'], pop['v']],
        minlength=int(pop['n']),
    )
    truth = info['target'] - info['zeta'] * (
        degree.mean() - np.mean(degree > 0)
    )
    se = a['HT'].std(ddof=1) / np.sqrt(len(a['HT']))
    print(
        'sensitivity QA', info['zeta'],
        'analytic EHT', truth,
        'MC', a['HT'].mean(),
        'z', (a['HT'].mean() - truth) / se,
    )

print('All finite / complete-identity / pi=1 identity checks passed.')
