MINIMAL SIMULATION REPLICATION FILES
====================================
Causal inference with dyadic data in randomized experiments
Yilin Li, Lu Deng, Yong Wang, and Wang Miao
https://github.com/H4O2/Randomized_Experiment_for_Dyadic_Data

CONTENTS
  codes/run_all.py              Simulation suites and figure generation.
  codes/sim_core.py             Population and estimator calculations.
  codes/socfb-Caltech36.mtx      Required public network input.
  requirements.txt             Pinned direct dependencies.
  README.md / README.txt        Instructions.
  LICENSE / CITATION.cff        MIT software license and citation metadata.
  THIRD_PARTY_NOTICES.txt       Network attribution and data terms.
  .gitignore                   Excludes generated outputs and temporary files.

SETUP (from the repository root; publication Python version: 3.10.9)
  python3.10 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
On Windows use the corresponding .venv\Scripts activation command.
Direct versions: NumPy 1.23.5, pandas 1.5.3, NetworkX 2.8.4, Matplotlib 3.7.0.
Transitive dependencies are not fully locked.

FULL RUN
  python codes/run_all.py --suite all --reps 500 --variance-reps 10000 --output runs/full

SHORT EXECUTION CHECK
  python codes/run_all.py --suite all --reps 5 --variance-reps 5 --output runs/smoke

INDIVIDUAL SUITES
  --suite main          Main outcome-distribution and parameter-grid studies.
  --suite caltech       Caltech randomization comparison AND variance studies.
  --suite variance      Only the Caltech variance studies.
  --suite sensitivity   Higher-order peer effects.
  --suite subsampling   Poisson subsampling and selected-pair matching.
  --suite cluster       Hub-and-clique complete/cluster randomization.
For uniform outcomes alone, use --suite main --distributions uniform.

OUTPUTS
The selected output directory receives results/, fig/, and a manifest.
The program saves populations, replicate NPZ/CSV estimates, per-cell summaries,
parameters, seeds and actual software versions. Master seed: 20261018.
Use a fresh output path for each run; existing files at that path may be replaced.
Generated figures/results are ignored by Git and are not shipped in this folder.
The separate full journal archive preserves the publication populations and
outputs, schematic blob.pdf, and independent 20,000-draw verification materials.

SCOPE AND TERMS
The Caltech input has 769 vertices and 16,656 undirected edges. Its supplied
single-percent header is preserved and handled by the explicit parser.
The outcome coefficients and treatment assignments are simulated. Numeric zero
outcomes remain observed. Some studies intentionally share populations/draws.
See README.md for the suite mapping and THIRD_PARTY_NOTICES.txt for attribution.
The MIT software license does not replace the third-party data terms.
Confidential WeChat data and application code are excluded; these files
reproduce simulation studies, not the empirical illustration.
