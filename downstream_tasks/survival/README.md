# Survival Analysis

This folder contains code for bag-level survival prediction with an **ABMIL** model.

## Repository layout

- `main.py`: cross-validation training and evaluation entry point.
- `dataset.py`: survival label loading, case aggregation, fold handling, and feature loading.
- `network.py`: ABMIL survival model.
- `engine.py`, `loss.py`: training loop and survival losses.
- `utils/`: optimizer, scheduler, seed, and utility helpers.
- `dataset_excel/demo_survival_labels.csv`: synthetic label table showing the expected schema.
- `run_survival.sh`: example launcher.

## Install

```bash
python -m pip install -r requirements.txt
```

## Inputs

The label table can be CSV or Excel. Required columns are:

- `filename`
- `event_time`
- `event_status`
- `case_id`
- `split`

The code also supports the full format with `dataset`, `case`, `slide`, `event_time`, `event_status`, and `Fold 0` to `Fold 4` columns.

Feature files are not included. Provide a feature root containing one `.pt` tensor per slide. For example, when using:

```bash
--root /path/to/features --feature GRACE
```

the code expects files under:

```text
/path/to/features/GRACE/<slide_id>.pt
```

where `<slide_id>` matches the label-table filename without the suffix.

## Run

```bash
python main.py \
  --study demo_survival \
  --feature GRACE \
  --root /path/to/features \
  --excel_file dataset_excel/demo_survival_labels.csv \
  --output_dir result
```

`run_survival.sh` is a convenience wrapper for the same command. Override paths as needed:

```bash
PYTHON_BIN=/path/to/python bash run_survival.sh \
  --root /path/to/features \
  --excel_file /path/to/labels.csv
```

Outputs are created at runtime below:

```text
result/<study>/<model>/<feature>/<timestamp>/
```
