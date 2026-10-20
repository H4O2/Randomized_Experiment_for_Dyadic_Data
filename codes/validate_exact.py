"""
validate_exact.py
-----------------
Optional exhaustive checks of estimators on small, fixed populations.

Enumerating every assignment avoids Monte Carlo error. The checks compare the
vectorized implementation with explicit dense sums, then verify expectations
under Bernoulli, complete, and cluster randomization. Additional checks cover
Poisson subsampling, fixed matching, and the sensitivity model.

Run directly to print the results and save exact_validation.json beside this
script. This does not change any publication simulation cells or figures.
"""

import itertools, json
from pathlib import Path

import numpy as np
import networkx as nx

import sim_core as sc


# ---------------------------------------------------------------------------
#   Numerical comparison and independent dense formulas
# ---------------------------------------------------------------------------
def close(a, b, label):
    """Require two calculations to agree within floating-point tolerance."""
    if not np.allclose(a, b, rtol=2e-11, atol=2e-11):
        raise AssertionError((label, a, b))


def dense_reference(pop, z, p, delta):
    """Evaluate HT, V1, and adjusted V2 using explicit dyad/triple sums.

    This deliberately uses dense matrices and loops instead of the vectorized
    graph sums in sim_core. It is intended only for the small validation graphs.
    """
    n = pop.n
    y, _ = sc.outcomes(pop, z)
    Y = np.zeros((n, n))
    Y[pop.ou, pop.ov] = y
    s = Y + Y.T

    ht = sum(
        Y[i, j] * (
            z[i] * z[j] / p**2
            - (1 - z[i]) * (1 - z[j]) / (1 - p)**2
        )
        for i in range(n)
        for j in range(n)
    ) / n

    # d: pair contribution; t/c: treated/control triple contributions;
    # sq: the additional degree-adjusted squared-outcome contribution.
    d = t = c = sq = 0.
    for i in range(n):
        for j in range(i + 1, n):
            d += s[i, j]**2 * (
                z[i] * z[j] / p**4
                + (1 - z[i]) * (1 - z[j]) / (1 - p)**4
            )
            sq += 2 * delta * s[i, j]**2 * (
                z[i] * z[j] / p**2
                + (1 - z[i]) * (1 - z[j]) / (1 - p)**2
            )

        for j in range(n):
            for k in range(n):
                if len({i, j, k}) < 3:
                    continue
                t += s[i, j] * s[i, k] * z[i] * z[j] * z[k] / p**4
                c += (
                    s[i, j] * s[i, k]
                    * (1 - z[i]) * (1 - z[j]) * (1 - z[k])
                    / (1 - p)**4
                )

    return ht, (d + t + c) / n**2, (d + sq + (1 - p) * t + p * c) / n**2


# ---------------------------------------------------------------------------
#   Exhaustive checks under each design and sampling scheme
# ---------------------------------------------------------------------------
def validate():
    """Run all exact checks and return a list of result dictionaries."""
    reports = []

    # Bernoulli assignment: compare every realization, then its expectation.
    # The binomial case checks that zero outcomes retain their graph support.
    for name, g in [
        ('path4', nx.path_graph(4)),
        ('cycle5', nx.cycle_graph(5)),
        ('star6', nx.star_graph(5)),
        ('zero_support', nx.path_graph(5)),
    ]:
        pop = sc.population(
            g,
            'binomial' if name == 'zero_support' else 'uniform',
            'exact/' + name,
        )
        p = .3
        vals = []
        w = []

        for bits in itertools.product([0., 1.], repeat=pop.n):
            z = np.array(bits)
            r = sc.estimate(pop, z, p)
            weight = p**z.sum() * (1 - p)**(pop.n - z.sum())
            d = dense_reference(pop, z, p, pop.degree.max())
            close(
                [r['HT'], r['V1'], r['V2_max']],
                d,
                name + ' pointwise dense',
            )
            vals.append(r)
            w.append(weight)

        w = np.array(w)
        mean = lambda k: sum(x * r[k] for x, r in zip(w, vals))
        v = sum(x * (r['HT'] - pop.target())**2 for x, r in zip(w, vals))

        close(mean('HT'), pop.target(), name + ' HT unbiased')
        close(mean('PI2'), pop.target(), name + ' SNIPE2 unbiased')
        close(v, sc.true_variance(pop, p), name + ' exact variance')
        close(mean('V1'), sc.expected_v1(pop, p), name + ' V1 expectation')
        close(
            mean('V2_max'),
            sc.expected_v2(pop, p, pop.degree.max()),
            name + ' V2 expectation',
        )
        assert mean('V2_max') >= v - 1e-10

        reports.append(dict(
            graph=name,
            target=pop.target(),
            ht_mean=mean('HT'),
            snipe2_mean=mean('PI2'),
            variance=v,
            mean_V1=mean('V1'),
            mean_V2_max=mean('V2_max'),
        ))

    # Complete randomization: enumerate all assignments with three treated.
    pop = sc.population(nx.cycle_graph(6), 'uniform', 'exact/complete')
    vals = []
    for treated in itertools.combinations(range(6), 3):
        z = np.zeros(6)
        z[list(treated)] = 1
        r = sc.estimate(pop, z, .5, 'complete')
        close(r['HT'], r['HA'], 'complete HT=HA')
        vals.append(r['HT'])
    close(np.mean(vals), pop.target(), 'complete unbiased')

    # Cluster randomization: enumerate all assignments of the three clusters.
    pop.cluster = np.array([0, 0, 1, 1, 2, 2])
    vals = []
    w = []
    p = .3
    for bits in itertools.product([0., 1.], repeat=3):
        bits = np.array(bits)
        z = bits[pop.cluster]
        r = sc.estimate(pop, z, p, 'cluster')
        vals.append(r['HT'])
        w.append(p**bits.sum() * (1 - p)**(3 - bits.sum()))
    close(np.dot(vals, w), pop.target(), 'cluster unbiased')

    # Poisson subsampling: averaging every sampling mask conditional on each
    # assignment equals its full-data HT estimate. The empty mask contributes 0.
    pop = sc.population(nx.path_graph(4), 'uniform', 'exact/subsample')
    p = .3
    pi = .4
    for bits in itertools.product([0., 1.], repeat=4):
        z = np.array(bits)
        y, _ = sc.outcomes(pop, z)
        s = y[:pop.m] + y[pop.m:]
        c = s * (
            z[pop.u] * z[pop.v] / p**2
            - (1 - z[pop.u]) * (1 - z[pop.v]) / (1 - p)**2
        )

        # Sample reciprocal pairs independently with inclusion probability pi.
        de = 0.
        for mask in itertools.product([0., 1.], repeat=pop.m):
            mask = np.array(mask)
            pr = pi**mask.sum() * (1 - pi)**(pop.m - mask.sum())
            de += pr * np.dot(c, mask) / (pop.n * pi)

        # Sample units independently and retain their outgoing ordered totals.
        row = np.bincount(
            pop.ou,
            weights=y * (
                z[pop.ou] * z[pop.ov] / p**2
                - (1 - z[pop.ou]) * (1 - z[pop.ov]) / (1 - p)**2
            ),
            minlength=pop.n,
        )
        ue = 0.
        for mask in itertools.product([0., 1.], repeat=pop.n):
            mask = np.array(mask)
            pr = pi**mask.sum() * (1 - pi)**(pop.n - mask.sum())
            ue += pr * np.dot(row, mask) / (pop.n * pi)

        close([de, ue], [c.sum() / pop.n] * 2, 'Poisson conditional unbiased')

    # Fixed matching targets the selected pair population's mean contrast.
    chosen = np.array([0, 2])
    a, b = pop.pair_potentials()
    expect = 0.
    for bits in itertools.product([0., 1.], repeat=4):
        z = np.array(bits)
        y, _ = sc.outcomes(pop, z)
        s = y[:pop.m] + y[pop.m:]
        c = s * (
            z[pop.u] * z[pop.v] / p**2
            - (1 - z[pop.u]) * (1 - z[pop.v]) / (1 - p)**2
        )
        expect += p**z.sum() * (1 - p)**(4 - z.sum()) * c[chosen].mean()
    close(expect, (a - b)[chosen].mean(), 'matching selected-pair target')

    # The fixed-support sensitivity model has an analytic HT expectation.
    pop = sc.population(nx.path_graph(5), 'uniform', 'exact/sensitivity')
    zeta = .2
    expect = 0.
    for bits in itertools.product([0., 1.], repeat=pop.n):
        z = np.array(bits)
        r = sc.estimate(pop, z, p, zeta=zeta)
        expect += p**z.sum() * (1 - p)**(pop.n - z.sum()) * r['HT']

    analytic = pop.target() + zeta * np.mean(pop.degree > 0)
    close(expect, analytic, 'higher-order sensitivity HT expectation')
    reports.append(dict(
        check='sensitivity HT analytic expectation',
        enumerated=expect,
        analytic=analytic,
        global_target=pop.target(zeta),
    ))
    reports.append(dict(
        check='complete, cluster, empty Poisson samples, selected-pair matching',
        passed=True,
    ))
    return reports


# ---------------------------------------------------------------------------
#   Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    r = validate()
    print(json.dumps(r, indent=2))
    Path(__file__).resolve().with_name('exact_validation.json').write_text(
        json.dumps(r, indent=2) + '\n'
    )
