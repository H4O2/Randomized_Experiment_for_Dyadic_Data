# Causal inference with dyadic data in randomized experiments

Code for the paper by Yilin Li, Lu Deng, Yong Wang and Wang Miao.

Simulation code for the main paper and supplementary material. The scripts generate the simulated populations, estimates, variance diagnostics, summaries and figures.

This repository contains the simulation code and its public Caltech network input. The semi-synthetic example CSV and confidential empirical data are not included. These scripts reproduce the simulation studies; they do not reproduce the confidential WeChat application.

## Installation

The publication outputs record Python 3.10.9, NumPy 1.23.5, pandas 1.5.3, NetworkX 2.8.4 and Matplotlib 3.7.0. With Python 3.10 installed, run the following from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r codes/requirements.txt
```

On Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r codes/requirements.txt
.\.venv\Scripts\Activate.ps1
```

The requirements file pins the four direct libraries, not all transitive dependencies. Use Python 3.10 with these historical package versions. Different library versions can produce small floating-point or rendering differences; PDF timestamps can also differ between runs.

## Run the simulations

Run the remaining commands from `codes/`:

```bash
cd codes
```

For a short execution check, using five assignments per setting:

```bash
python run_all.py --quick --output ../runs/quick
```

For the full main-paper and supplementary simulations:

```bash
python run_all.py --reps 500 --variance-reps 10000
```

The full command writes `results/` and `fig/` in the repository root. It includes all three coefficient distributions, both Caltech randomization designs, the variance experiments, sensitivity, subsampling and the cluster experiment.

To keep a new run in a separate directory:

```bash
python run_all.py --reps 500 --variance-reps 10000 --output ../runs/full
```

Existing files at the selected output location can be overwritten. Quick runs check execution and are not publication results. Use separate output directories when changing replication counts.

### Individual experiments

| Experiment | Command |
| --- | --- |
| Main BA simulations, uniform coefficients | `python run_revised_simulations.py --reps 500 --distr uniform` |
| Supplementary binomial coefficients | `python run_revised_simulations.py --reps 500 --distr binomial` |
| Supplementary lognormal coefficients | `python run_revised_simulations.py --reps 500 --distr lognormal` |
| Caltech Bernoulli and complete randomization | `python run_randomization_designs.py --reps 500` |
| Variance under changes in degree or treatment probability | `python run_variance_behaviour.py --reps 10000` |
| Higher-order peer-effect sensitivity | `python run_sensitivity.py --reps 500` |
| Dyad/unit Poisson sampling and selected-pair matching | `python run_subsample.py --reps 500` |
| Hub-and-clique complete and cluster randomization | `python run_cluster.py --reps 500` |

Each experiment script accepts `--quick` and `--output`. The main BA script accepts `--distr all` to run all three coefficient distributions.

`run_all.py` also accepts `--suite` and `--distributions`. Its `caltech` suite runs both the randomization comparison and variance experiments; the standalone `run_randomization_designs.py` runs only the randomization comparison. Use `--help` for the available options.

## File guide

All Python scripts, `requirements.txt` and the required `socfb-Caltech36.mtx` input are in `codes/`.

| File | Purpose |
| --- | --- |
| `run_all.py` | Run all or selected experiment suites. |
| `run_revised_simulations.py` | Main BA experiments and supplementary coefficient distributions. |
| `run_randomization_designs.py` | Bernoulli versus complete randomization on Caltech. |
| `run_variance_behaviour.py` | Caltech degree and treatment-probability variance experiments. |
| `run_sensitivity.py` | Higher-order peer-effect sensitivity. |
| `run_subsample.py` | Dyad/unit subsampling and selected-pair matching. |
| `run_cluster.py` | Hub-and-clique randomization experiment. |
| `sim_core.py` | Populations, outcomes, estimators and variances. |
| `sensitivity_core.py` | Shared sensitivity population and assignment stream. |
| `subsample_core.py` | Poisson sampling and selected-pair matching calculations. |
| `simulation_common.py` | Summaries, plots and output records. |
| `simulation_cli.py` | Shared command-line options and output directories. |
| `audit_outputs.py` | Audit saved results and write combined summary CSVs. |
| `replot_main.py` | Replot main and cluster figures from saved arrays. |
| `replot_legends.py` | Replot selected coverage, standard-error and variance figures. |
| `validate_exact.py` | Exhaustive small-graph estimator checks. |
| `validate_variance.py` | Independent variance check using 20,000 assignments. |

The saved-result tools read `results/` beside `codes/`; they do not accept an alternative output directory. After a full run with the default output location:

```bash
python audit_outputs.py
python replot_main.py
python replot_legends.py
python validate_exact.py
python validate_variance.py
```

`audit_outputs.py` writes `results/all_point_summaries.csv` and `results/all_variance_summaries.csv`. Replotting writes figure files, but the two replot tools do not jointly regenerate every figure; the full simulation run does. `validate_exact.py` writes `exact_validation.json` in `codes/`. `validate_variance.py` requires saved variance-experiment results and writes three `independent_variance_validation` files under `results/`.

## Settings and outputs

The master seed is **20261018**. `sim_core.seed` derives deterministic streams from fixed experiment labels. The code preserves these labels and the sequential order of random draws.

- Main grids: `n = 200, 500, 1000, 1500, 2000`; `p = .1, .2, .3, .4, .5`; BA attachment parameter `m = 2, 5, 10, 15, 20`. Parameters not being varied remain at `n = 1000`, `m = 10`, `p = .3`. The plotted nominal average degree is `2m`.
- Sensitivity: `zeta = 0, .01, .02, .05, .1, .2`.
- Sampling probabilities: `.25, .5, .75, 1`.
- Cluster experiment: `100, 200, 300, 400, 500` five-node cliques and one hub.

A full run saves 83 experiment cells and one shared block containing 24 subsampling settings. The 12 variance cells use 10,000 assignments each; other experiments use 500.

Each experiment cell saves `population.npz`, `replicates.npz`, `replicates.csv`, `metadata.json` and `summary.csv`. The shared subsampling block also includes matching summaries. NPZ files retain full array precision; replicate CSVs use 15 significant digits. Records include targets, seeds, Monte Carlo standard errors, coverage and empty-group/negative-variance counts. A completed run records library versions and an output-file manifest.

The full run generates 58 PDF/PNG figure pairs: 48 simulation figures used in the paper and supplement and 10 additional diagnostics. Publication filenames are retained to match the LaTeX sources. Illustrative diagrams are not simulation outputs.

### Statistical conventions

HT denotes Horvitz–Thompson; HA denotes Hájek; DM denotes unit aggregate; TR denotes exposure-threshold comparison; PI2 denotes SNIPE(2), divided by two for the dyadic target. V1 is the original variance estimate; V2_avg and V2_max use the known graph's average and maximum degrees.

Confidence-interval diagnostics use HT under independent Bernoulli assignment and take square roots of the nonnegative part of each variance estimate. Fixed graphs retain eligible dyads with zero outcomes. Complete and cluster randomization use their exact joint probabilities. Empty arms and samples follow the implemented zero-contribution conventions; no replicates are dropped. Matching is evaluated against its selected-pair target, separately from the global target.

The lognormal robustness model is unbounded and does not check a theorem that assumes uniformly bounded potential outcomes.

## Public network input and attribution

Keep `codes/socfb-Caltech36.mtx` beside `sim_core.py`. The parser retains all 769 vertices and 16,656 undirected edges; the Caltech experiments require this input.

The input is the [Caltech36 network from Network Repository](https://networkrepository.com/socfb-Caltech36.php). See the [source data policy](https://networkrepository.com/policy.php) and [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for attribution and redistribution terms. The software license does not override the network's terms.

Requested source references:

- Ryan A. Rossi and Nesreen K. Ahmed (2015). *The Network Data Repository with Interactive Graph Analytics and Visualization*. AAAI.
- Amanda L. Traud, Peter J. Mucha and Mason A. Porter (2012). *Social structure of Facebook networks*. Physica A, 391(16), 4165–4180.
- Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha and Mason A. Porter (2011). *Comparing Community Structure to Characteristics in Online Collegiate Social Networks*. SIAM Review, 53(3), 526–543.

The software is distributed under the existing [MIT license](LICENSE). Python dependencies retain their respective licenses and are installed separately.

See [CITATION.cff](CITATION.cff) for software citation metadata. Please cite the accompanying paper and the relevant network sources when using these materials.
