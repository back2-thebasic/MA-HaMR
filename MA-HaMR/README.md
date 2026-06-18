# MA-HaMR

Memory-Augmented Amortized Dyn-HaMR — standalone research codebase.

**Dyn-HaMR is used only as an offline teacher.** This repo reads its output files via paths; it does not import or modify `Dyn-HaMR/dyn-hamr`.

## Repository layout

```
/data/SuC/
├── Dyn-HaMR/                 # upstream teacher (code only)
└── MA-HaMR/                  # this project — all new code

/extra/SuC/                   # large artifacts on extended drive
├── data/raw/                 # HOT3D, H2O, ...
├── data/mahmr/processed/     # Step 1 exports
├── dynhamr_io/videos/        # mp4 inputs for Dyn-HaMR
├── vipe_results/             # VIPE camera outputs
└── experiments/outputs/logs/ # Dyn-HaMR hydra logs
```

See [docs/project_layout.md](docs/project_layout.md) for the full directory spec.

## Quick start — Step 1 mini (demo1)

```bash
cd /data/SuC/MA-HaMR

# export expert labels from an existing Dyn-HaMR log
python scripts/export_expert_labels.py \
  --config configs/export/demo1.yaml

# sanity check exported tensors
python scripts/validate_labels.py \
  --processed-dir ../data/mahmr/processed/demo1
```

## Install

Use the same conda env as Dyn-HaMR (torch + numpy). Then:

```bash
pip install -e .
```
