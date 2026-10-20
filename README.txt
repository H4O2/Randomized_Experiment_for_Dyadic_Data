SIMULATION CODE
===============
Causal inference with dyadic data in randomized experiments

This package uses the updated statistical calculations, organized into separate
experiment scripts in the style of the earlier code release. The calculations,
seed labels, parameter grids and figure settings are preserved. Code is written
with expanded statements, explanatory comments and documented functions.

1. FOLDER CONTENTS
------------------
  codes/                     Simulation scripts, dependencies and Caltech input.
  README.txt                 Setup and reproduction commands.
  LICENSE                    Existing software license.
  THIRD_PARTY_NOTICES.txt    Attribution for the public network input.

Simulation results and figures are generated when the scripts run. This
repository contains simulation code and the required public Caltech network
input. Confidential WeChat data and empirical-analysis code are not included.

2. INSTALLATION
----------------
The publication outputs record Python 3.10.9 with:
  NumPy 1.23.5, pandas 1.5.3, NetworkX 2.8.4, Matplotlib 3.7.0.

From this package's root directory, with Python 3.10 installed:
  python3.10 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r codes/requirements.txt

On Windows, create the environment with `py -3.10 -m venv .venv`, then use
.venv\Scripts\python.exe in place of python in subsequent commands.

codes/requirements.txt pins the four direct libraries, not all transitive
dependencies. Use Python 3.10 for these historical package versions.
Use the recorded versions when reproducing publication figures. Other versions
can cause tiny floating-point or rendering differences. PDF timestamps also
prevent byte-identical PDF files across runs.

3. RUN ALL PAPER AND SUPPLEMENT SIMULATIONS
------------------------------------------
Run the following commands from the codes/ directory:
  cd codes

Small execution check, using five assignments per setting:
  python run_all.py --quick --output ../runs/quick

Full reproduction with the publication replication counts:
  python run_all.py --reps 500 --variance-reps 10000

The full command generates ../results/ and ../fig/. This includes all three
coefficient distributions, both Caltech randomization designs, the variance
experiments, sensitivity, subsampling and the cluster experiment.

To keep a new run separate, append --output ../runs/full to the full command.
Existing outputs at the selected location can be overwritten. Quick runs test
execution only; their results must not replace full-run publication outputs.
Use a different output directory when changing the number of replications.

4. RUN INDIVIDUAL EXPERIMENTS
----------------------------
All commands below also run from codes/. Default outputs are ../results/ and
../fig/, so different experiments can populate the same release directory.

Main-paper BA simulations (uniform coefficients):
  python run_revised_simulations.py --reps 500 --distr uniform

Supplementary coefficient distributions:
  python run_revised_simulations.py --reps 500 --distr binomial
  python run_revised_simulations.py --reps 500 --distr lognormal

Caltech Bernoulli and complete randomization:
  python run_randomization_designs.py --reps 500

Variance behaviour under degree/probability changes:
  python run_variance_behaviour.py --reps 10000

Higher-order peer-effect sensitivity:
  python run_sensitivity.py --reps 500

Dyad/unit Poisson sampling and selected-pair matching:
  python run_subsample.py --reps 500

Hub-and-clique complete/cluster randomization:
  python run_cluster.py --reps 500

Each experiment script accepts --quick and --output. The main BA script accepts
--distr all to run all three distributions. Use --help for supported options.
The combined run_all.py also retains --suite and --distributions. Its caltech
suite runs both randomization and variance experiments; the standalone
run_randomization_designs.py runs only the randomization comparison.

The earlier archive's parallel --n-jobs and date-labelled output options are
not used in this implementation. Random draws follow the updated sequential
streams so that the scientific calculations and reproducible results agree.

5. SCRIPT MAP
--------------
Experiment drivers:
  run_revised_simulations.py    Main BA experiments and supplementary distributions.
  run_randomization_designs.py  Bernoulli versus complete randomization on Caltech.
  run_variance_behaviour.py     Caltech degree/probability variance experiments.
  run_sensitivity.py            Higher-order peer-effect sensitivity.
  run_subsample.py              Dyad/unit subsampling and matching.
  run_cluster.py                Hub-and-clique randomization experiment.
  run_all.py                    Runs the selected combination of experiments.

Shared calculations:
  sim_core.py                  Populations, outcomes, estimators and variances.
  sensitivity_core.py          Shared sensitivity population and assignment stream.
  subsample_core.py            Poisson sampling and selected-pair matching.
  simulation_common.py         Result summaries, plots and output records.
  simulation_cli.py            Shared command-line options and output directories.

Optional tools, retained from the updated implementation:
  audit_outputs.py              Writes combined summary CSVs from saved results.
  replot_main.py                Replots main/cluster figures from saved arrays.
  replot_legends.py             Replots selected coverage, SE and variance figures.
  validate_exact.py             Exhaustive small-graph estimator checks.
  validate_variance.py          Independent 20,000-assignment variance check.

The saved-result tools look for results/ beside codes/. After a default full
run, for example, `python audit_outputs.py` creates all_point_summaries.csv and
all_variance_summaries.csv under ../results/. Replotting writes figure files.
The two replot tools do not jointly regenerate every figure; a full run does.
validate_exact.py writes exact_validation.json beside itself. The independent
variance check requires a completed variance experiment and writes three
independent_variance_validation files under ../results/.

6. SIMULATION SETTINGS AND OUTPUTS
----------------------------------
Master seed: 20261018. sim_core.seed derives the separate random streams from
fixed experiment labels. These labels and random-number draw order are retained.

The main grids use n=200,500,1000,1500,2000; p=.1,.2,.3,.4,.5; and BA attachment
m=2,5,10,15,20. The other parameters stay at n=1000, m=10, p=.3. The plotted
nominal average degree is 2m. The uniform, binomial and lognormal coefficient
distributions have the same parameters as the updated implementation.

There are 83 ordinary simulation cells, plus one shared block containing
24 subsampling settings. Ordinary cells use 500 assignments, and the 12 variance
cells use 10,000. Sensitivity uses zeta=0,.01,.02,.05,.1,.2; sampling probabilities
are .25,.5,.75,1. The cluster experiment uses 100,200,300,400,500 five-node
cliques and one hub.

Each ordinary cell stores population.npz, replicates.npz, replicates.csv,
metadata.json and summary.csv. NPZ files preserve full array precision; replicate
CSVs use 15 significant digits. Metadata include targets, seeds, Monte Carlo
standard errors, coverage and empty-group/negative-variance counts. A completed
run also records its library versions and output-file manifest.

A full run creates 58 PDF/PNG figure pairs: 48 simulation figure files used in
the paper and supplement, plus 10 additional diagnostics. Illustrative diagrams
are not simulation outputs. The publication filenames
are retained so generated figures can be matched to the LaTeX sources.

HT = Horvitz-Thompson; HA = Hajek; DM = unit aggregate;
TR = exposure-threshold comparison; PI2 = SNIPE(2), divided by two for the
dyadic target. V1 is the original variance estimate; V2_avg and V2_max use the
known graph's average and maximum degrees. Confidence-interval diagnostics use
HT and are computed only for independent Bernoulli assignment. They use the
nonnegative part of each variance estimate when taking square roots.

The fixed graph retains eligible dyads whose outcome is zero. Complete and
cluster randomization use their exact joint probabilities. Empty arms/samples
use the updated zero-contribution conventions; no replicates are dropped.
Matching is assessed against its selected-pair target, separately from the
global target. The lognormal robustness model is unbounded; it is not a check
of a theorem requiring uniformly bounded potential outcomes.

Keep socfb-Caltech36.mtx beside sim_core.py. Its parser retains all 769 vertices
and 16,656 undirected edges. There is no synthetic-network fallback.

7. ATTRIBUTION
---------------
See LICENSE for the existing software license and
THIRD_PARTY_NOTICES.txt for the public-network attribution. The software license
does not override third-party terms.
