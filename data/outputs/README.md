# Output Files

This directory holds generated CSVs and short provenance notes.

## Current recommendation

- `ver4.1.csv`
  Best externally confirmed candidate so far.
  This is the current target regime for further development: conservative copula generation with support-preserving mixed-column handling and without the later aggressive conditional blanking changes.

- `ver3.csv`
  Previous strong external baseline.
  Kept as the historical reference point that first showed the conservative direction could work.

## Existing files

- `ver2.csv`
  Earlier exploratory output.
  Kept for comparison against later copula-driven runs.

- `ver3.csv`
  High-performing baseline used as the practical reference point for later work.
  The current development direction is to make future copula outputs look more like this file and then improve incrementally.

- `copula_submission.csv`
  Generated from the newer copula path after aggressive mixed-column value/state modeling.
  Local checks were acceptable, but hidden-score behavior appears much worse than `ver3.csv`.
  Treat this file as a regression case, not the target behavior.

- `ver4.1.csv`
  Strong conservative copula output that currently has the best known external leaderboard score.
  Treat this file as the main reference when evaluating later changes.

- `ver6.csv`
  Generated from the same conservative copula artifact after tightening hierarchical logic in the repair/export path.
  It improved the local proxy but regressed badly on the external leaderboard, so it should be treated as a cautionary example of over-cleaning.

- `ver7_*.csv`
  Conservative seed-search candidates generated from the same copula artifact.
  These are intended to explore sample variance inside the `ver4.1` regime without changing the underlying generation logic.

- `ctgan_from_artifact.csv`
  Output generated from a saved CTGAN artifact.
  Useful as a model-family comparison point rather than the preferred submission path.

- `test.csv`
  General exploratory output retained for debugging and format checks.

## Sidecar notes

New outputs generated through the code now also write a same-name markdown sidecar, for example:

- `my_submission.csv`
- `my_submission.md`

The markdown file records the model artifact, config, repair/privacy settings, and any notable generation notes.
