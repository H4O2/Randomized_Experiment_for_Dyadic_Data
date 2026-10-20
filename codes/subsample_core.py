"""
subsample_core.py
-----------------
Monte Carlo calculations for dyad sampling, unit sampling and matching.

Dyad and unit Poisson samples use shared uniforms across inclusion
probabilities, so samples are nested. A matching is drawn from the fixed
graph independently before treatment and outcomes. Matching estimates are
compared with their selected-pair targets.
"""
import numpy as np

import sim_core as sc


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def greedy_matching(pop, rng):
    """Select disjoint edges by traversing a random ordering of fixed edges."""
    used = np.zeros(pop.n, dtype=bool)
    chosen = []
    for j in rng.permutation(pop.m):
        u = pop.u[j]
        v = pop.v[j]
        if not used[u] and not used[v]:
            chosen.append(j)
            used[u] = used[v] = True
    return np.array(chosen, dtype=int)


# ---------------------------------------------------------------------------
# Replicate calculations
# ---------------------------------------------------------------------------
def run_subsampling(pop, p, reps, zetas, pis):
    """Return all draws, targets and empty-sample counts for the experiment.

    The random streams and their draw order agree with the updated
    simulation: matching, treatment, dyad uniforms, then unit uniforms.
    Each replicate reuses these draws across all zeta and pi settings.
    """
    q = 1 - p
    n = pop.n
    rz = np.random.default_rng(sc.seed('sensitivity/assignment'))
    rs = np.random.default_rng(sc.seed('subsample/sampling'))
    rm = np.random.default_rng(sc.seed('subsample/matching'))

    full = np.zeros((reps, len(zetas)))
    dyad = np.zeros((reps, len(zetas), len(pis)))
    unit = np.zeros_like(dyad)
    match = np.zeros_like(full)
    mtarget = np.zeros_like(full)
    msize = np.zeros(reps, dtype=int)
    dyadempty = np.zeros((reps, len(pis)), dtype=int)
    unitempty = np.zeros_like(dyadempty)

    for b in range(reps):
        # Draw matching independently of treatment and potential outcomes.
        chosen = greedy_matching(pop, rm)
        msize[b] = len(chosen)
        z = (rz.random(n) < p).astype(float)
        du = rs.random(pop.m)
        uu = rs.random(n)

        # Inverse-probability weights for treated and control dyads.
        tt = z[pop.u] * z[pop.v] / p ** 2
        cc = (1 - z[pop.u]) * (1 - z[pop.v]) / q ** 2

        for j, zeta in enumerate(zetas):
            y, _ = sc.outcomes(pop, z, zeta)
            s = y[:pop.m] + y[pop.m:]
            c = s * (tt - cc)
            full[b, j] = c.sum() / n
            row = np.bincount(
                pop.ou,
                weights=y * np.r_[tt - cc, tt - cc],
                minlength=n,
            )

            for k, pi in enumerate(pis):
                dm = du < pi
                um = uu < pi
                dyad[b, j, k] = c[dm].sum() / (n * pi)
                unit[b, j, k] = row[um].sum() / (n * pi)
                dyadempty[b, k] = not dm.any()
                unitempty[b, k] = not um.any()

            a, bb = pop.pair_potentials()
            match[b, j] = c[chosen].mean()
            mtarget[b, j] = (a - bb)[chosen].mean() + 2 * zeta

    target = np.array([pop.target(z) for z in zetas])
    return {
        'full_HT': full,
        'dyad_HT': dyad,
        'unit_HT': unit,
        'matching': match,
        'matching_target': mtarget,
        'matching_size': msize,
        'global_target': target,
        'zeta': zetas,
        'pi': pis,
        'dyad_empty': dyadempty,
        'unit_empty': unitempty,
    }
