"""
sensitivity_core.py
-------------------
Population and Monte Carlo helpers for the sensitivity analysis.

The fixed graph, coefficients and assignment seed are shared with the
subsampling experiment. Estimators and potential outcomes are implemented
once, in sim_core.py.
"""
import sim_core as sc


# ---------------------------------------------------------------------------
# Shared finite population and treatment assignments
# ---------------------------------------------------------------------------
def sensitivity_population():
    """Build the fixed population used at every peer-effect magnitude."""
    graph = sc.ba(1000, 10, 'sensitivity/graph')
    return sc.population(graph, 'uniform', 'sensitivity/population')


def run_sensitivity_setting(pop, p, reps, zeta):
    """Use the same assignment stream for each value of zeta."""
    return sc.run(
        pop,
        p,
        reps,
        'sensitivity/assignment',
        zeta=zeta,
    )
