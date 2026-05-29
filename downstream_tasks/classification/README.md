# Classification

This folder contains the downstream **ABMIL** pipeline for slide-level classification from pre-extracted features.

## Setup

```bash
conda env create -f dstr.yml
conda activate dstr
```

## Input Requirements

The pipeline expects:

- a feature directory containing per-slide `.pt` files
- an Excel or CSV split file

The split table must contain at least these columns:

- `slide`
- `label`
- `split`

Optional columns such as `case`, `dataset`, `study`, `cohort`, or `source` are supported by the current dataset loader.

## Demo Run

Edit `FEATURE_PATH` in `scripts/run.sh`, then run:

```bash
cd downstream_tasks/classification
bash scripts/run.sh
```

## Direct Launch Example

```bash
python main.py \
  --model ABMIL \
  --root /path/to/feature/root \
  --feature GRACE \
  --study demo \
  --excel_file dataset_excel/demo.xlsx \
  --num_epoch 50 \
  --batch_size 1 \
  --lr 2e-4 \
  --seed 1234 \
  --tqdm
```

## Outputs

The script writes logs to `logs/` and model outputs to `results/`.
