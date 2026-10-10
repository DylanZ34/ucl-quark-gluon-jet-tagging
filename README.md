# Quark and gluon jet tagging with CMS Open Data

Dylan Zhang — research with Professor Mario Campanelli, UCL.

This project compares K-means, logistic regression, random forests, feed-forward neural networks, and constituent transformers for distinguishing quark- and gluon-labeled jets. The final paper reports test ROC AUCs of 0.7051 for K-means and 0.7582–0.7715 for supervised methods; small differences among supervised models are not established as statistically significant.

## Research question
How much does increasing model complexity improve quark/gluon discrimination for this simulated CMS sample and these input representations?

## Structure
- `src/preprocessing/`: original feature extraction and plotting scripts (one filename normalized).
- `src/models/`: original training, comparison and scan scripts.
- `src/cmssw/`: shell workflow containing the C++ analyzer and CMS configuration.
- `data/`: 200-row feature sample (first 100 rows per truth label), not the full research dataset.
- `results/`: saved small result tables; earlier and final runs remain separated.
- `figures/`: existing result figures.
- `docs/`: provenance, limitations and validation.

## Setup and run
The original local environment records Python **3.11.9**. Create a virtual environment and run `python -m pip install -r requirements.txt`. Dependencies reflect imports; exact original package versions are not fully recoverable.

For a portable, small input demonstration, from the repository root:
```sh
python src/prepare_sample.py
cd work
python ../src/models/compare_all_methods.py
```
This trains the classical models on the sample. It skips transformer results unless those have been generated. **Sample scores do not reproduce the paper.** The scripts write into the ignored `work/` directory.

For full analysis, place a matching timestamped `jets_*.csv` and `constituents_*.csv` pair in `work/`; then run `calculate_substructure_features.py`, `calculate_extended_features.py`, and the desired training scripts using their relative paths under `src/`. Run from `work/`. These original scripts select input files by modification time; keep only a matching input pair there to avoid mixing runs. The 16-configuration scan is `src/models/scan_transformer_hyperparameters.py`. Preserve its validation-selected result separately from the standalone transformer output.

## Data
The paper identifies CMS simulated QCD data for 2016 conditions, DOI [10.7483/OPENDATA.CMS.5NJW.EBHC](https://doi.org/10.7483/OPENDATA.CMS.5NJW.EBHC). The exact ROOT input URL is in `src/cmssw/run_analysis.sh`. That script requires Linux, CMSSW_10_6_30, `/cvmfs/cms.cern.ch`, and an existing `Demo/DemoAnalyzer` package under `/code/CMSSW_10_6_30/src`; it writes analyzer/configuration files in that package. It is not a standalone Python setup and was not run during packaging. [TODO: document the original container image and CMSSW package bootstrap.]

Full constituent CSVs (94.8 MB and 475.6 MB), other full datasets, model checkpoints, archives, environment files and shell history are excluded. Recreate CSVs through the original CMS workflow. Event/run/luminosity identifiers in the sample are physics simulation identifiers, not personal IDs.

## Paper and acknowledgments
[Paper](https://docs.google.com/document/d/172zIVji5am8gRNdvkr6SW-L-gPiuvIUMEQE9ldYTV-I/edit) · [Research overview](https://docs.google.com/document/d/1oRTRlT-RCV8ua2SgPOj4rHBo6vuPR0mvHnstiuMG1xY/edit)

Acknowledgments: Professor Mario Campanelli; UCL; CMS Collaboration and CERN Open Data. See the paper for scientific references. This packaging does not assign a new license to the research or third-party material.

## Reproducibility caveats
The final paper Table 1 matches `results/method_comparison_results_with_eta/all_methods_summary_with_eta.csv`. The selected scan transformer has test AUC 0.7694028, while the standalone transformer and earlier comparison have different values. The current comparison script reads the standalone transformer directory; substitute the validation-selected scan predictions only deliberately and document that choice. The paper says quark codes 1–8; the extractor implements absolute codes 1–6. [TODO: reconcile this and the paper's jet selection statement.] Train/test splitting is by jet rather than grouped by collision event, and the transformer sorts rows before splitting whereas the classical script does not; identical test membership across methods is not established. No confidence intervals or new full-data reproduction are claimed.
