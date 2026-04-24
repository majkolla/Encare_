# Encare Synthetic Data Pipeline

This project generates synthetic clinical tabular data for the Encare hackathon. The best review path is the repaired Gaussian copula solution: it learns marginal distributions and cross-column dependence from `data/data.csv`, samples synthetic rows, repairs known clinical constraints, removes exact privacy matches, and writes a schema-valid CSV.


## Summary


1. Load `data/data.csv`.
2. Infer a schema for every column.
3. Convert heterogeneous columns into numeric/categorical model features.
4. Fit a Gaussian copula model.
5. Sample new latent Gaussian rows.
6. Invert sampled rows back to original clinical column formats.
7. Apply deterministic rule repair.
8. Drop duplicate and exact source-record matches.
9. Coerce the final CSV to the source schema.
10. Validate with local proxy metrics.

The primary model is configured by `configs/copula_best.yaml` and `configs/base.yaml`.


## Best Solution Configuration

`configs/copula_best.yaml`:

```yaml
model: copula
models:
  - copula

copula:
  model: copula
  repair: true
  privacy_filter: true
  privacy_min_distance: 0.0
```
