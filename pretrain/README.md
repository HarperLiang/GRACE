# Pretraining

This folder contains the **DINO-style LoRA continual pretraining** code for pathology patch data stored in HDF5 files.

## Files

- `main_dino.py`: main pretraining entry point.
- `dataset.py`: HDF5 patch dataset loader.
- `myenv.yml`: Conda environment for pretraining.
- `run.slurm`: single-node Slurm launch example.
- `run_with_submitit.py`: optional submitit launcher.

## Environment

```bash
conda env create -f myenv.yml
conda activate dino_lora
```

The environment uses PyTorch `2.5.1+cu121` and torchvision `0.20.1+cu121`.

## Data

`dataset.py` expects a root directory containing `.h5` files. 

## Run

Single-node distributed example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $CONDA_PREFIX/bin/torchrun --nproc_per_node=8 main_dino.py \
                                              --arch virchow2 \
                                              --data_path /path/to/data \
                                              --output_dir /path/to/output \
                                              --saveckp_freq 1 \
                                              --batch_size_per_gpu 48 \
                                              --patch_size 14 \
                                              --local_crops_size 98 > /path/to/log.log 2>&1 
```

