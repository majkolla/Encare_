# Encare Synthetic Data Hackathon

This repo now contains a structured synthetic-data pipeline with:

- schema inference and missingness profiling
- train/validation splitting
- an independent baseline sampler
- a custom Gaussian Copula synthesizer
- local multi-metric evaluation
- rule-based repair and privacy filtering
- optional CTGAN and hybrid wrappers

## Layout

- `configs/`: experiment configs
- `data/`: source data, generated outputs, and artifacts
- `src/data/`: loading, schema inference, preprocessing, splits
- `src/models/`: baseline, copula, CTGAN, hybrid, adaptive extra
- `src/eval/`: marginals, dependencies, discriminator, privacy, logic, scoring, reports
- `src/rules/`: constraint inference and repair helpers
- `src/main.py`: full pipeline orchestration
- `src/train.py`: single-model train/evaluate entrypoint
- `src/generate.py`: sample/repair/privacy-filter helpers
- `src/submit.py`: schema enforcement and submission validation

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

What each command does:

- `python -m venv .venv`
Creates an isolated Python environment inside the repo so this project uses its own interpreter and packages instead of whatever is installed globally.

- `source .venv/bin/activate`
Makes `python` and `pip` point at that virtual environment in your current shell, so the next commands install into and run from `.venv`.

- `pip install -r requirements.txt`
Installs the packages this repo imports, including `sdv` for CTGAN support. This is why `main.py`, `src.train`, and `validator.py` can run without import errors afterward.

## GPU

CTGAN is the only model here that can use your NVIDIA GPU. The config now defaults to `cuda: auto`, which means:

- if `torch.cuda.is_available()` is `True`, CTGAN will ask SDV to train on CUDA
- if CUDA is not available, CTGAN falls back to CPU with a warning instead of crashing
- the shipped CTGAN config is intentionally conservative on memory: smaller batch size, smaller network layers, smaller embedding size, `pac: 1`, and constant-column dropping

Quick check before training:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Why this works:

- `torch.cuda.is_available()` tells you whether the PyTorch build inside `.venv` can actually see a CUDA device
- `torch.cuda.device_count()` tells you how many GPUs that environment can use

If you want a lower-level driver check too:

```bash
nvidia-smi
```

That works because it queries the NVIDIA driver directly, independent of Python.

## Run

End-to-end experiment sweep:

```bash
python main.py --config configs/base.yaml --data data/data.csv
```

What it does:

- loads the source CSV
- infers the schema and constraints
- creates a train/validation split
- trains the models listed in `configs/base.yaml`
- evaluates each run locally
- refits the best available model on the full dataset
- writes artifacts and a submission CSV

Why it works:

- `main.py` is a thin wrapper over `src/main.py`
- the config and data paths are now resolved from the repo root, so the command works even if you launch it from another directory

Single-model training:

```bash
python -m src.train --model copula --config configs/copula.yaml --data data/data.csv
```

What it does:

- trains just one model instead of the full sweep
- evaluates it on one train/validation split
- saves its metrics and model artifact
- does not write a submission CSV

Why it works:

- `python -m src.train` runs the `src/train.py` module as part of the `src` package, so imports like `from src...` resolve correctly
- the model-specific config is merged with `configs/base.yaml`, so shared settings still apply
- this command is meant for model-level validation, so it stops after saving reusable artifacts under `data/artifacts/`

Single-model CTGAN training with GPU auto-detection:

```bash
python -m src.train --model ctgan --config configs/ctgan.yaml --data data/data.csv
```

What it does:

- trains just the CTGAN model
- uses `cuda: auto` from the CTGAN config, so it will use CUDA if PyTorch can see your GPU
- trains with conservative memory-oriented defaults so it does not try to allocate the much larger SDV defaults on this table

If your desktop still becomes unstable:

- lower `batch_size` further, for example to `16`
- lower `embedding_dim` to `16`
- set `max_training_rows` in the CTGAN config, for example `2000` or `4000`, as an emergency fallback

Why it works:

- the CTGAN wrapper resolves the CUDA setting before constructing the SDV synthesizer
- the same CTGAN CUDA setting is also reused by the hybrid model, because the hybrid internally trains a CTGAN component plus a copula component

Safer first CTGAN run on a desktop machine:

```bash
python -m src.train --model ctgan --config configs/ctgan_safe.yaml --data data/data.csv
```

What it does:

- trains CTGAN with a much smaller batch size and smaller networks
- limits the training rows to `2000` as a stability-first fallback

Why it works:

- this sharply reduces RAM, swap, and GPU-memory pressure compared with the full CTGAN config
- it gives you a stable first run to verify that your machine can complete training before you scale the settings back up

Generate a CSV from a saved single-model artifact:

```bash
python -m src.generate --model-path data/artifacts/ctgan/model.pkl --config configs/ctgan_safe.yaml --data data/data.csv --output data/outputs/ctgan_from_artifact.csv
```

What it does:

- loads a previously trained model artifact
- samples synthetic rows
- applies repair and privacy filtering from the config
- validates the final schema and writes a CSV to `data/outputs/`

Why it works:

- `src.generate` is the sampling/export step for a saved artifact, while `src.train` is only the fit-and-evaluate step
- CTGAN artifact loading is CPU-safe, so a model trained with CUDA can still be loaded later for generation

Fast precheck on a generated CSV:

```bash
python validator.py --synthetic data/outputs/run_YYYYMMDD_HHMMSS/best_submission.csv --quick
```

What it does:

- checks schema, column order, dtypes, and row count
- runs the quick sanity report from `validator.py`

Why it works:

- `validator.py` now resolves `data/` and `configs/` relative to the repo root
- `--quick` skips the slower discriminator/privacy score and gives you a fast local check

Full precheck with the local proxy score:

```bash
python validator.py --synthetic data/outputs/run_YYYYMMDD_HHMMSS/best_submission.csv
```

What it does:

- runs the same format checks as quick mode
- also computes the full local proxy score from `src/eval`

Why it works:

- it reuses the same schema inference and scoring code as the main pipeline, so the precheck is aligned with the local model-selection logic

Outputs are written under `data/outputs/` and `data/artifacts/`.
