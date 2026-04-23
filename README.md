# Encare Synthetic Data Pipeline

This project generates synthetic clinical tabular data for the Encare hackathon. The best review path is the repaired Gaussian copula solution: it learns marginal distributions and cross-column dependence from `data/data.csv`, samples synthetic rows, repairs known clinical constraints, removes exact privacy matches, and writes a schema-valid CSV.

The implementation is deliberately small enough for a judge to inspect quickly. It is not a general synthetic-data framework; it is a practical competition pipeline for a mixed clinical table with numeric fields, categorical fields, dates, times, missing values, and conditional child fields.

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

## For review 

S
- `src/main.py`: full orchestration, train/validation split, model ranking, final refit, final submission write.
- `configs/copula_best.yaml`: best solution config. It selects the copula model and enables repair/privacy filtering.
- `configs/base.yaml`: shared parameters, output paths, score weights, and defaults.
- `src/models/gaussian_copula_model.py`: core model implementation and the main file to review mathematically.
- `src/data/schema.py`: schema inference for numeric, categorical, date, time, id-like, constant, and mixed columns.
- `src/data/mixed.py`: split/restore logic for columns that contain both numeric values and tokens such as `Unknown`.
- `src/generate.py`: sampling loop, oversampling, deduplication, privacy filter, and row selection.
- `src/rules/constraints.py`: clinical constraints that are built from the source schema.
- `src/rules/repair.py`: deterministic post-generation repair logic.
- `src/submit.py`: final schema coercion and submission validation.
- `src/eval/score.py`: local proxy score aggregation.
- `src/eval/marginals.py`, `src/eval/dependencies.py`, `src/eval/discriminator.py`, `src/eval/privacy.py`, `src/eval/logic.py`: metric details.


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

Relevant inherited defaults from `configs/base.yaml`:

| Parameter | Value | Meaning |
| --- | --- | --- |
| `seed` | `42` | Reproducible split and sampling seed. |
| `val_frac` | `0.2` | 20% validation split for local model comparison. |
| `n_rows_multiplier` | `1.0` | Final output has at least the source row count. |
| `output_dir` | `data/outputs` | Generated CSV/report location. |
| `artifact_dir` | `data/artifacts` | Saved model/report location. |
| `score_weights.marginal` | `0.30` | Weight for one-column distribution fidelity. |
| `score_weights.dependency` | `0.30` | Weight for cross-column relationship fidelity. |
| `score_weights.discriminator` | `0.20` | Weight for real-vs-synthetic indistinguishability. |
| `score_weights.privacy` | `0.15` | Weight for exact match, duplicate, and nearest-neighbor privacy. |
| `score_weights.logic` | `0.05` | Weight for clinical/rule consistency. |
| `copula.repair` | `true` | Apply deterministic clinical repairs after sampling. |
| `copula.privacy_filter` | `true` | Drop duplicates and exact source-record matches. |
| `copula.privacy_min_distance` | `0.0` | Do not enforce extra nearest-neighbor distance beyond exact-match filtering. |

Important internal defaults used by `src/generate.py` and `src/models/gaussian_copula_model.py`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mixed_column_strategy` | `split` | Mixed numeric/token columns are modeled as value plus state columns. |
| `snap_numeric_max_unique` | `32` | Low-cardinality numeric samples snap back to observed numeric support. |
| `oversample_factor` | `1.1` | Each generation attempt samples 10% more rows than still needed. |
| `max_attempts` | `4` | Maximum sample/filter attempts to reach target rows. |
| `selection_strategy` | `head` | Keep first rows after filtering unless `balanced` is configured. |
| `selection_max_unique` | `12` | Max cardinality for categorical balancing features. |
| `selection_missingness_weight` | `0.35` | Weight for missingness balancing if balanced selection is used. |
| `selection_deficit_bias` | `1.5` | Extra penalty for dropping rows that represent under-supplied categories. |
| `include_conditional_blanks` | `false` unless set | Builds extra parent/child blanking rules only when enabled. |
| `derived_repair_mode` | `overwrite` | Recompute derived targets such as BMI/dates from inputs. |

## Pipeline Logic

### 1. Schema Inference

`src/data/schema.py` converts the raw table into a `Schema`. For each column `j`, it records:

- `kind`: one of `numeric`, `categorical`, `binary`, `id_like`, `datetime`, `time`, or `constant`.
- `missing_rate`: `m_j = count_missing_j / n`.
- `unique_count`: number of observed non-null values.
- `allowed_values`: observed category support for low-cardinality categorical fields.
- `mixed_value_kind`: marks columns that contain numeric values plus non-numeric tokens.

The main heuristics are:

- ID-like if unique ratio is at least `0.98`, values are monotonic numeric, or the name looks like an identifier.
- Date-like if the name mentions date or at least 95% of sampled non-null values match `YYYY-MM-DD`.
- Time-like if the name or values match `HH:MM`.
- Numeric-like if at least 80% of non-null values parse as numeric, or at least 50% parse as numeric with at least 10 unique values.
- Mixed numeric/token if a column has both parseable numeric values and non-numeric tokens.

### 2. Mixed Column Encoding

Some clinical columns contain values like numbers plus `Unknown`. These are not safe to model as only numeric or only categorical. `src/data/mixed.py` splits each mixed column `x_j` into:

- `x_j__value`: numeric/date/time value, with tokens coerced to missing.
- `x_j__state`: one of `__VALUE__`, `__MISSING__`, or the original non-numeric token.

For example:

```text
42       -> value=42, state=__VALUE__
Unknown  -> value=NaN, state=Unknown
missing  -> value=NaN, state=__MISSING__
```

The copula learns both the numeric value distribution and the token/missingness state distribution. During inverse transform, `restore_mixed_column` combines them back into the original single-column format.

## Gaussian Copula Model

The model in `src/models/gaussian_copula_model.py` uses a rank-Gaussian copula. The idea is:

- Model each column's marginal distribution separately.
- Transform every column into a latent standard-normal-like variable.
- Estimate a correlation matrix over those latent variables.
- Sample from a multivariate normal with that correlation.
- Invert each sampled latent variable back through the stored marginal distribution.

Let the training table have `n` rows and modeled features `j = 1, ..., p` after mixed-column splitting.

### Numeric Features

For numeric feature `j`, let:

- `m_j` be the missing rate.
- `x_i` be a non-missing observed value.
- `r_i` be the average rank of `x_i` among non-missing values.
- `k_j` be the number of non-missing observations.
- `Phi` be the standard normal CDF.

The implementation maps observed numeric values into latent probabilities:

```text
u_i = m_j + (1 - m_j) * r_i / (k_j + 1)
z_i = Phi^{-1}(clip(u_i, eps, 1 - eps))
```

Missing numeric values are mapped to the middle of the missing mass:

```text
u_missing = m_j / 2
z_missing = Phi^{-1}(clip(u_missing, eps, 1 - eps))
```

The model stores:

- `missing_rate = m_j`
- sorted observed values
- empirical quantile probabilities
- whether all observed values are integers
- optional observed numeric support for low-cardinality numeric columns

During sampling:

```text
z*_j ~ latent Gaussian
u*_j = Phi(z*_j)
```

Then:

- if `u*_j < m_j`, output missing.
- otherwise compute `q = (u*_j - m_j) / (1 - m_j)`.
- interpolate `q` through the empirical quantile function.
- round if the original values were integer-like.
- snap to observed support if the column had at most `snap_numeric_max_unique` unique values.

This preserves the empirical one-column distribution while allowing the copula correlation matrix to preserve dependence.

### Categorical Features

For categorical feature `j`, the model estimates category probabilities:

```text
p_c = count(category = c) / n
```

Missing values are treated as a category token `__MISSING__`. Each category receives the midpoint of its cumulative probability interval:

```text
lower_c = sum_{d before c} p_d
mid_c = lower_c + p_c / 2
z_c = Phi^{-1}(clip(mid_c, eps, 1 - eps))
```

During sampling, the model draws `u*_j = Phi(z*_j)` and applies the inverse categorical CDF:

```text
sample c where cumulative_p_{c-1} < u*_j <= cumulative_p_c
```

This keeps the observed category frequencies while fitting category associations through the latent correlation matrix.

### Constant Features

Constant columns are not part of the latent correlation fit. The model stores the constant value and its missing rate, then recreates that column after sampling.

### Correlation Matrix

After every modeled feature is transformed to latent values, the matrix is:

```text
Z in R^{n x p}
R = corr(Z)
```

`R` is cleaned and regularized before sampling:

1. Replace `NaN` and infinities with `0`.
2. Symmetrize with `(R + R^T) / 2`.
3. Set diagonal to `1`.
4. Eigen-decompose `R = Q Lambda Q^T`.
5. Clip eigenvalues below `1e-4`.
6. Reconstruct and rescale so the diagonal is again `1`.

This makes the covariance numerically usable for multivariate normal sampling.

### Sampling

For each synthetic row:

```text
z* ~ Normal(0, R)
u* = Phi(z*)
```

Each feature is then inverted with the numeric or categorical inverse transform described above. Mixed columns are restored from their value/state features. Finally, the original source column order is restored.

## Repair and Clinical Logic

The copula samples plausible rows statistically, but clinical data also has deterministic relationships. `src/rules/constraints.py` builds constraints, and `src/rules/repair.py` applies them.

The repair sequence is:

1. `clip_ranges`: numeric columns are clipped to observed min/max ranges. Age is constrained to `[0, 120]`.
2. `normalize_categories`: invalid generated categories are set to missing.
3. `enforce_conditional_blanks`: if a parent field is inactive, child fields are blanked.
4. `recompute_derived_fields`: derived fields are recomputed.
5. `drop_or_resample_invalid_rows`: rows still violating hard constraints are removed, with optional resampling.

Examples of derived rules:

BMI:

```text
BMI = weight_kg / (height_cm / 100)^2
```

Postoperative dates:

```text
target_date = operation_date + duration_in_days
```

Conditional blank rules follow parent/child clinical structure. For example, if a complication parent starts with `No`, its complication subtype children should be blank.

## Privacy Filtering

`src/eval/privacy.py` implements two privacy protections during generation:

1. Drop duplicate synthetic rows.
2. Drop exact records that match a source row across all columns.

If `privacy_min_distance` or `privacy_min_distance_quantile` is configured above zero, the filter also uses nearest-neighbor distance:

1. Encode real and synthetic rows into numeric feature space.
2. Standardize continuous features.
3. Fit nearest neighbors on real rows.
4. Keep synthetic rows whose nearest real distance is at least the resolved threshold.

For the best copula config, `privacy_min_distance = 0.0`, so the practical privacy filter is deduplication plus exact source-record removal.

## Generation Loop

`src/generate.py` handles output generation. It repeatedly samples enough rows to survive repair and filtering.

For each attempt:

```text
needed = target_rows - current_rows
batch_size = max(round(needed * oversample_factor), needed)
batch = model.sample(batch_size)
batch = repair_dataframe(batch)
batch = filter_privacy_violations(batch)
synthetic = concat(existing, batch)
synthetic = drop_duplicates(synthetic)
synthetic = select_target_rows(synthetic)
```

The default row selection is `head`, which keeps generation simple and deterministic after filtering. A `balanced` mode exists for experiments: it scores overrepresented category/missingness states higher for dropping while protecting rare states.

For a balancing feature value `v`, the row drop score contribution is:

```text
surplus_v = max(pool_rate_v - real_rate_v, 0) / pool_rate_v
deficit_v = max(real_rate_v - pool_rate_v, 0) / real_rate_v
score_v = weight * surplus_v - weight * deficit_bias * deficit_v
```

Rows with high surplus scores are dropped first, but the implementation avoids deleting the last row of a protected category level.

## Local Evaluation Math

The local score is a proxy used for model selection. It is not guaranteed to equal the hidden competition score.

`src/eval/score.py` computes:

```text
S_total =
  0.30 * S_marginal
+ 0.30 * S_dependency
+ 0.20 * S_discriminator
+ 0.15 * S_privacy
+ 0.05 * S_logic
```

### Marginal Score

Numeric columns use:

- Kolmogorov-Smirnov score: `1 - KS(real, synthetic)`.
- Wasserstein score: `1 / (1 + W(real, synthetic) / std(real))`.

Categorical columns use:

- Total variation score: `1 - 0.5 * sum_c |p_real(c) - p_syn(c)|`.
- Jensen-Shannon score: `1 - JSD(p_real, p_syn) / log(2)`.

The aggregate marginal score is the mean of available metric families.

### Dependency Score

Dependency scoring compares relationship matrices from real and synthetic data:

- Pearson correlation for numeric features.
- Spearman correlation for numeric features.
- Cramer's V for categorical pairs.
- Spearman correlation on a mixed encoded matrix.

The basic pattern is:

```text
mean_abs_diff = mean(abs(M_real - M_syn))
score = max(0, 1 - scaled_mean_abs_diff)
```

### Discriminator Score

`src/eval/discriminator.py` trains a random forest classifier to distinguish real rows from synthetic rows. If AUC is close to `0.5`, the synthetic data is hard to distinguish.

```text
S_discriminator = max(0, 1 - abs(AUC - 0.5) / 0.5)
```

### Privacy Score

Privacy combines exact-match rate, duplicate rate, and nearest-source distance:

```text
distance_component = median_distance / (1 + median_distance)
S_privacy = clip((1 - exact_match_rate) * (1 - duplicate_rate) * distance_component, 0, 1)
```

### Logic Score

Logic score measures constraint violation rates:

- range violations
- invalid category violations
- derived field violations
- conditional blank violations

```text
S_logic = max(0, 1 - mean_violation_rate)
```

## How To Run

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the best copula path:

```bash
python -m src.train --model copula --config configs/copula_best.yaml --data data/data.csv
```

Generate a CSV from a saved model:

```bash
python -m src.generate \
  --model-path data/artifacts/copula/model.pkl \
  --config configs/copula_best.yaml \
  --data data/data.csv \
  --output data/outputs/submission.csv
```

Run the full sweep:

```bash
python main.py --config configs/base.yaml --data data/data.csv
```

Validate a generated CSV quickly:

```bash
python validator.py --synthetic data/outputs/submission.csv --quick
```

Run the full local proxy score:

```bash
python validator.py --synthetic data/outputs/submission.csv
```

## Outputs

Generated CSVs go to `data/outputs/`. Model artifacts and metric reports go to `data/artifacts/`. These directories can become large and are ignored for sharing. For judging, the important part is the source implementation and the compact run commands above, not the full historical output archive.
