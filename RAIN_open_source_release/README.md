# Recurrent Attentive Integration Network

This release contains the regression and classification main experiments for
the Recurrent Attentive Integration Network (RAIN). The package separates exact
paper-result evaluation from optional model retraining so that every reported
number remains auditable.

## Quick start

Open this folder in PyCharm and run `main.py`. The script recalculates all main
metrics from the frozen held-out predictions used by the manuscript and checks
them against fixed expected values.

Run `verify_release.py` to verify all code, data, checkpoints, and prediction
files against `SHA256SUMS.txt`.

Expected results:

| Experiment | Principal metrics |
| --- | --- |
| PM2.5 regression | MAE 3.5084, RMSE 5.0169, R2 0.8652 |
| Mineral classification | OA 0.8672, F1 0.8765, MCC 0.7470, ROC-AUC 0.9070, PR-AUC 0.8889 |

The evaluation scripts terminate with an error if the metrics differ. This
path is deterministic because it evaluates the published out-of-sample
predictions rather than retraining stochastic neural networks.

## Retraining

Complete training implementations are included:

- `regression/train.py`
- `classification/train.py`

Retraining uses the frozen architecture, hyperparameters, seeds, and data
protocol. Newly trained weights and predictions are written to separate
directories and never replace the paper artifacts. Small numerical differences
may occur across PyTorch, CUDA, BLAS, CPU, and GPU versions. Such differences
must be reported as retraining variation, not silently calibrated.

## Directory structure

```text
RAIN_open_source_release/
  main.py
  classification/
    main.py                    manuscript metric evaluation
    train.py                   full model retraining
    paper_predictions/         frozen five-fold out-of-fold predictions
    models/                    paper fold checkpoints
  regression/
    main.py                    manuscript metric evaluation
    train.py                   full model retraining
    paper_predictions/         frozen held-out test predictions
    models/                    paper checkpoint
  docs/
    MANUSCRIPT_ALIGNMENT.md
    MODEL_CARD.md
    DATA_NOTICE.md
```

## Integrity guarantees

- Paper metrics are recalculated from row-level held-out predictions.
- Classification uses 490 out-of-fold predictions from five disjoint folds.
- Regression uses 437 held-out test predictions.
- Test or held-out labels are not used to modify prediction scores.
- Expected values and file hashes are version controlled.
- Retraining outputs cannot overwrite paper checkpoints or predictions.

## Data and licensing

The MIT license applies to the software. Dataset provenance and third-party
licensing considerations are described in `docs/DATA_NOTICE.md`. Users should
cite the original data providers and the accompanying article when using this
release.

