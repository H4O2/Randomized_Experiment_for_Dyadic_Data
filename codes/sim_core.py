"""
sim_core.py
-----------
Shared simulation functions for the dyadic-data study.

Each simulated population has a fixed, known undirected support graph. Every
edge supports two ordered outcomes, including outcomes that are numerically
zero. Neighbourhoods are therefore defined by the graph, not by observed values.

Potential outcomes follow the model

    Y_ij(z) = alpha_ij + beta_ij * z_i + gamma_ij * z_j
              + lambda_ij * z_i * z_j + zeta * peer_i(z),

where peer_i(z) is the treated fraction of unit i's graph neighbours. The
ordinary dyadic model uses zeta = 0. This module does not simulate empirical data.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
#   Reproducible seeds and output names
# ---------------------------------------------------------------------------
MASTER_SEED = 20261018

# HT: Horvitz--Thompson; HA: Hajek; DM: unit-total difference estimator;
# TR: threshold estimator; PI2: second-order SNIPE estimator, divided by two.
METHODS = ['HT', 'HA', 'DM', 'TR', 'PI2']

# V1 is the unadjusted estimator. V2 uses average or maximum graph degree.
VARS = ['V1', 'V2_avg', 'V2_max']


def seed(label):
    """Return a deterministic seed for a named graph, population, or run.

    Labels allow experiments to share a population while using separate
    assignment streams. Keep both MASTER_SEED and the labels unchanged when
    reproducing the supplied results.
    """
    return int.from_bytes(
        hashlib.sha256(f'{MASTER_SEED}:{label}'.encode()).digest()[:8],
        'little',
    )


# ---------------------------------------------------------------------------
#   Fixed finite populations
# ---------------------------------------------------------------------------
@dataclass
class Population:
    """A graph and the coefficients of its ordered potential outcomes.

    u and v contain the endpoints of the m undirected edges. Coefficient
    arrays have length 2*m: the first half follows (u, v), and the second
    follows (v, u). Derived arrays ou and ov store these ordered endpoints.
    cluster, when provided, gives the cluster membership of each unit.
    """

    n: int
    u: np.ndarray
    v: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    lam: np.ndarray
    cluster: np.ndarray | None = None

    def __post_init__(self):
        self.m = len(self.u)
        self.ou = np.r_[self.u, self.v]
        self.ov = np.r_[self.v, self.u]
        self.degree = np.bincount(np.r_[self.u, self.v], minlength=self.n)

    def target(self, zeta=0.):
        """Return the all-treated versus all-control contrast per unit."""
        return float(
            np.sum(self.beta + self.gamma + self.lam) / self.n
            + zeta * 2 * self.m / self.n
        )

    def pair_potentials(self):
        """Return reciprocal-pair totals under all treatment and all control.

        The returned arrays contain Y_ij(1,1) + Y_ji(1,1) and
        Y_ij(0,0) + Y_ji(0,0) for the dyadic model (zeta = 0).
        """
        a = self.alpha + self.beta + self.gamma + self.lam
        return (
            a[:self.m] + a[self.m:],
            self.alpha[:self.m] + self.alpha[self.m:],
        )

    def save(self, path):
        """Save the graph and coefficients in a compressed NumPy archive."""
        d = {
            k: getattr(self, k)
            for k in ['n', 'u', 'v', 'alpha', 'beta', 'gamma', 'lam']
        }
        if self.cluster is not None:
            d['cluster'] = self.cluster
        np.savez_compressed(path, **d)


def graph_edges(g):
    """Return sorted endpoints, giving coefficients a reproducible order."""
    e = np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=int)
    return e[:, 0], e[:, 1]


def population(g, distr, label, cluster=None, coefficients=None):
    """Construct a fixed population on graph g.

    If coefficients are supplied, they are four matrices containing alpha,
    beta, gamma, and lambda. Otherwise, draw coefficients from the selected
    distribution. Drawing occurs once per population, before randomization.
    """
    u, v = graph_edges(g)
    n = g.number_of_nodes()
    rng = np.random.default_rng(seed(label))
    k = 2 * len(u)

    if coefficients is not None:
        # Extract the two ordered outcomes for every undirected graph edge.
        ou = np.r_[u, v]
        ov = np.r_[v, u]
        a, b, c, d = [x[ou, ov] for x in coefficients]
    elif distr in ('uniform', 'cluster'):
        a = rng.uniform(0, 1, k)
        b = rng.uniform(0, 1 if distr == 'cluster' else .5, k)
        c = rng.uniform(-.5, 0, k)
        d = rng.uniform(0, .5, k)
    elif distr == 'binomial':
        a = np.zeros(k)
        b = rng.binomial(1, .2, k).astype(float)
        c = rng.binomial(1, .1, k).astype(float)
        d = rng.binomial(1, .1, k).astype(float)
    elif distr == 'lognormal':
        a = rng.lognormal(1, .5, k)
        b = rng.lognormal(.2, .1, k)
        c = rng.lognormal(.2, .1, k)
        d = rng.lognormal(.2, .1, k)
    else:
        raise ValueError(distr)

    return Population(n, u, v, a, b, c, d, cluster)


# ---------------------------------------------------------------------------
#   Graph construction and network input
# ---------------------------------------------------------------------------
def ba(n, m, label):
    """Build a reproducible Barabasi--Albert graph with n units."""
    return nx.barabasi_albert_graph(n, m, seed=seed(label))


def load_caltech(path):
    """Read the supplied Caltech coordinate file, keeping every vertex.

    The file has a single-% header. Parse its coordinate data explicitly,
    convert one-based vertex labels to zero-based labels, and check the
    dimensions and edge count. Isolated vertices are retained.
    """
    rows = []
    shape = None

    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('%'):
                continue

            vals = [int(x) for x in line.split()[:3]]
            if shape is None:
                shape = vals
                continue

            rows.append((vals[0] - 1, vals[1] - 1))

    if (
        shape is None
        or len(shape) != 3
        or shape[0] != shape[1]
        or len(rows) != shape[2]
    ):
        raise ValueError('Invalid MatrixMarket dimensions/coordinate count')

    if any(
        i == j or min(i, j) < 0 or max(i, j) >= shape[0]
        for i, j in rows
    ):
        raise ValueError('Invalid edge')

    g = nx.Graph()
    g.add_nodes_from(range(shape[0]))
    g.add_edges_from(rows)
    if g.number_of_edges() != shape[2]:
        raise ValueError('Duplicate undirected edges')

    return g


# ---------------------------------------------------------------------------
#   Observed outcomes and graph sums
# ---------------------------------------------------------------------------
def neighbour_sum(pop, edge_weights):
    """Sum undirected edge weights at each of their two endpoints."""
    return (
        np.bincount(pop.u, weights=edge_weights, minlength=pop.n)
        + np.bincount(pop.v, weights=edge_weights, minlength=pop.n)
    )


def outcomes(pop, z, zeta=0.):
    """Return ordered outcomes and each unit's treated-neighbour fraction.

    An isolated unit has a neighbour fraction of zero. Numerical zero
    outcomes do not remove neighbours from the fixed support graph.
    """
    peer = (
        np.bincount(pop.u, weights=z[pop.v], minlength=pop.n)
        + np.bincount(pop.v, weights=z[pop.u], minlength=pop.n)
    ) / np.maximum(pop.degree, 1)

    return (
        pop.alpha
        + pop.beta * z[pop.ou]
        + pop.gamma * z[pop.ov]
        + pop.lam * z[pop.ou] * z[pop.ov]
        + zeta * peer[pop.ou],
        peer,
    )


# ---------------------------------------------------------------------------
#   Point and variance estimators for one treatment assignment
# ---------------------------------------------------------------------------
def estimate(pop, z, p, design='Bernoulli', zeta=0., with_variance=True):
    """Compute the estimators for one assignment on a fixed population.

    Complete randomization uses the exact joint assignment probabilities.
    Cluster randomization distinguishes within-cluster and between-cluster
    edges. Variance formulas and PI2 are evaluated only under Bernoulli
    randomization; PI2 is NaN for the other designs.

    HA and TR use zero for the contribution of an empty arm or threshold
    group. The returned empty_arm and empty_threshold indicators record
    those cases. Variance estimates are returned without clipping.
    """
    z = np.asarray(z, dtype=float)
    n = pop.n
    m = pop.m
    q = 1 - p

    # Joint treatment/control probabilities for each reciprocal pair.
    if design == 'complete':
        n1 = int(z.sum())
        p = n1 / n
        q = 1 - p
        if min(n1, n - n1) < 2:
            raise ValueError('Need at least two units in each arm')

        pt = n1 * (n1 - 1) / (n * (n - 1))
        pc = (n - n1) * (n - n1 - 1) / (n * (n - 1))
    elif design == 'cluster':
        same = pop.cluster[pop.u] == pop.cluster[pop.v]
        pt = np.where(same, p, p * p)
        pc = np.where(same, q, q * q)
    elif design == 'Bernoulli':
        pt = p * p
        pc = q * q
    else:
        raise ValueError(design)

    y, peer = outcomes(pop, z, zeta)
    s = y[:m] + y[m:]
    tt = z[pop.u] * z[pop.v]
    cc = (1 - z[pop.u]) * (1 - z[pop.v])

    # Inverse-probability weighted dyadic totals and unit counts.
    tnum = np.sum(tt * s / pt)
    cnum = np.sum(cc * s / pc)
    nt = z.sum() / p
    nc = (n - z.sum()) / q

    ht = (tnum - cnum) / n
    ha = (tnum / nt if nt else 0.) - (cnum / nc if nc else 0.)

    # Unit totals include both incoming and outgoing ordered outcomes.
    yi = (
        np.bincount(pop.ou, weights=y, minlength=n)
        + np.bincount(pop.ov, weights=y, minlength=n)
    )
    dm = np.sum(yi * (z / p - (1 - z) / q)) / n

    # Threshold groups require a nonempty, fixed graph neighbourhood.
    it = (z == 1) & (peer >= p) & (pop.degree > 0)
    ic = (z == 0) & ((1 - peer) >= q) & (pop.degree > 0)
    tr = (
        (yi[it].mean() if it.any() else 0.)
        - (yi[ic].mean() if ic.any() else 0.)
    )

    # Second-order SNIPE score on the closed neighbourhood (including self).
    pi2 = np.nan
    if design == 'Bernoulli':
        x = (z - p) / (p * q)
        t = (
            x
            + np.bincount(pop.u, weights=x[pop.v], minlength=n)
            + np.bincount(pop.v, weights=x[pop.u], minlength=n)
        )
        x2 = x * x
        ss = (
            x2
            + np.bincount(pop.u, weights=x2[pop.v], minlength=n)
            + np.bincount(pop.v, weights=x2[pop.u], minlength=n)
        )
        score = t + (1 - 2 * p) * (t * t - ss) / 2

        # Aggregate unit totals have all-treated/all-control contrast 2*tau.
        pi2 = np.sum(yi * score) / (2 * n)

    out = dict(zip(METHODS, [ht, ha, dm, tr, pi2]))
    out['empty_arm'] = float(nt == 0 or nc == 0)
    out['empty_threshold'] = float(not it.any() or not ic.any())

    if with_variance and design == 'Bernoulli':
        # Pair contribution: sum of squared reciprocal-pair outcomes.
        diag = np.sum(tt * s * s / p**4 + cc * s * s / q**4) / n**2

        # For each unit, accumulate pair totals (a) and squared totals (b)
        # over treated (t) and control (c) neighbours. The identity a^2 - b
        # removes diagonal products and leaves distinct-neighbour products.
        at = (
            np.bincount(pop.u, weights=s * z[pop.v], minlength=n)
            + np.bincount(pop.v, weights=s * z[pop.u], minlength=n)
        )
        bt = (
            np.bincount(pop.u, weights=s * s * z[pop.v], minlength=n)
            + np.bincount(pop.v, weights=s * s * z[pop.u], minlength=n)
        )
        ac = (
            np.bincount(pop.u, weights=s * (1 - z[pop.v]), minlength=n)
            + np.bincount(pop.v, weights=s * (1 - z[pop.u]), minlength=n)
        )
        bc = (
            np.bincount(pop.u, weights=s * s * (1 - z[pop.v]), minlength=n)
            + np.bincount(pop.v, weights=s * s * (1 - z[pop.u]), minlength=n)
        )

        tri_t = np.sum(z * (at * at - bt)) / p**4 / n**2
        tri_c = np.sum((1 - z) * (ac * ac - bc)) / q**4 / n**2
        square = 2 * np.sum(tt * s * s / p**2 + cc * s * s / q**2) / n**2

        # The adjusted estimators use q*tri_t + p*tri_c. Keep these weights
        # separate from the unadjusted V1 expression.
        out.update(
            V1=diag + tri_t + tri_c,
            V2_avg=diag + pop.degree.mean() * square + q * tri_t + p * tri_c,
            V2_max=diag + pop.degree.max() * square + q * tri_t + p * tri_c,
        )

    return out


# ---------------------------------------------------------------------------
#   Exact variance and expectations under Bernoulli randomization
# ---------------------------------------------------------------------------
def true_variance(pop, p):
    """Return the exact HT variance for a fixed dyadic population (zeta=0).

    a and b are reciprocal-pair potential outcomes under all treatment and
    all control. The cross sum collects products of distinct edges that
    share a unit; disjoint edges have zero covariance under Bernoulli design.
    """
    q = 1 - p
    a, b = pop.pair_potentials()
    t = a - b
    diag = np.sum(a * a / p**2 + b * b / q**2 - t * t)

    def cross(x):
        sx = neighbour_sum(pop, x)
        return np.sum(sx * sx) - 2 * np.sum(x * x)

    return float((diag + cross(a) / p + cross(b) / q - cross(t)) / pop.n**2)


def expected_v1(pop, p):
    """Return the exact expectation of V1 in the dyadic Bernoulli model."""
    a, b = pop.pair_potentials()
    t = a - b
    return (
        true_variance(pop, p)
        + (np.sum(neighbour_sum(pop, t)**2) - np.sum(t * t)) / pop.n**2
    )


def expected_v2(pop, p, delta):
    """Return the exact expectation of V2 with a fixed degree adjustment."""
    a, b = pop.pair_potentials()
    q = 1 - p

    def cross(x):
        return np.sum(neighbour_sum(pop, x)**2) - 2 * np.sum(x * x)

    return float(
        (
            np.sum(a * a / p**2 + b * b / q**2)
            + 2 * delta * np.sum(a * a + b * b)
            + q * cross(a) / p
            + p * cross(b) / q
        ) / pop.n**2
    )


# ---------------------------------------------------------------------------
#   Monte Carlo driver for one fixed population and design
# ---------------------------------------------------------------------------
def run(pop, p, reps, label, design='Bernoulli', zeta=0., n1=None):
    """Draw assignments and collect one array per returned estimator.

    Complete randomization treats floor(p*n) units unless n1 is supplied.
    Cluster randomization draws one treatment indicator per cluster.
    The graph and potential-outcome coefficients remain fixed across draws.
    """
    rng = np.random.default_rng(seed(label))
    rows = []

    for _ in range(reps):
        if design == 'complete':
            z = np.zeros(pop.n)
            z[
                rng.choice(
                    pop.n,
                    int(np.floor(p * pop.n)) if n1 is None else n1,
                    replace=False,
                )
            ] = 1
        elif design == 'cluster':
            z = rng.binomial(1, p, int(pop.cluster.max()) + 1)[pop.cluster]
        else:
            z = (rng.random(pop.n) < p).astype(float)

        rows.append(estimate(pop, z, p, design, zeta))

    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
