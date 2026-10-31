# RAIN model card

## Intended use

RAIN integrates domain-wide predictor-response associations with
geographically adaptive local interactions for continuous or categorical
spatial prediction. The included experiments address PM2.5 prediction across
China and mineral prospectivity mapping in the western Meguma terrane.

## Evidence provided

- complete training implementations;
- fixed data splits and sampling protocol;
- paper checkpoints;
- row-level held-out predictions;
- deterministic metric-reproduction scripts;
- SHA-256 file manifest.

## Limitations

The experiments do not establish universal spatial generalization. Results are
specific to the reported datasets, splits, predictors, and candidate-negative
sampling protocol. Retraining can vary slightly across numerical runtimes.
Prospectivity scores are research outputs and should not be interpreted as
confirmed mineral resources without independent geological validation.

## Leakage controls

Regression test labels are used only after predictions are fixed. In
classification, each sample is evaluated only in its held-out fold. Frozen
predictions are used solely for transparent metric reproduction and are never
modified by the evaluation scripts.

