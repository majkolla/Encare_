# Encare Synthetic Data Pipeline

This project generates synthetic clinical tabular data for the Encare hackathon. The recommended review path is the repaired Gaussian copula solution: it learns marginal distributions and cross-column dependence from `data/data.csv`, samples synthetic rows, repairs known clinical constraints, removes exact privacy matches, and writes a schema-valid CSV.


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
10. Save the trained model artifact and generated output files.

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


## Further Investigation 

A Gaussian copula assumes a smooth dependencies between the different marginlized distribution (https://en.wikipedia.org/wiki/Copula_(statistics)). Since there is no reason to assume this smoothness, and instead we want to build something that can approximate the non smooth dependencies, this can be done by using for example a CTGAN in addition to the copula $q(x) = q_{copula}\lambda + (1 - \lambda)q_{CTAGN}$. Furthermore, since we have categorical combinations, we can consider creating a mixtue synthesiser, (this is an idea built after a conversation with Eric Herwin) where build it through categorical sheets: $q(x) = \sum p(s) q_s(x_{rest}|s)$ Simply put, instead of approximating the total distribution we say patients in different categorical groups may follow different distributions. The problem with this approach is that the data may be split in further pieces, therefore increasing the chance overfit,copy rows, learn noisy/random correlations etc. Therefore, to begin, the next important implementation is a model that mixes CTGAN and copula.
