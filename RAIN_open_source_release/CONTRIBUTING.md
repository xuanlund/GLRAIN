# Contributing

Contributions that improve portability, documentation, testing, or numerical
stability are welcome. Please keep scientific changes separate from software
maintenance changes.

Before submitting a change:

1. run `verify_release.py`;
2. run the root `main.py` and confirm that both metric checks pass;
3. document any change to data, splits, model structure, or hyperparameters;
4. do not overwrite files under `paper_predictions/` or `models/`.

Changes that alter scientific results should be submitted with new predictions,
metrics, provenance, and a clear explanation rather than replacing the archived
paper artifacts.

