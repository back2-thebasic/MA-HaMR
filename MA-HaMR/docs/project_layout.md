# MA-HaMR Project Layout

Design goal: **complete separation from Dyn-HaMR source**. Dyn-HaMR stays an external teacher; MA-HaMR only reads its outputs and public assets.

## Top-level layout

| Path | Role | Modify? |
|------|------|---------|
| `/data/SuC/Dyn-HaMR/` | Teacher pipeline (SLAM/VIPE + optimization) | No new MA-HaMR code |
| `/data/SuC/MA-HaMR/` | Student model, data export, training, eval | All new code here |
| `/extra/SuC/data/` | Datasets & processed tensors | Generated artifacts |
| `/extra/SuC/experiments/` | Run logs from both projects | Generated artifacts |

## `MA-HaMR/` — source code

```
MA-HaMR/
├── README.md
├── pyproject.toml              # pip install -e .
├── configs/
│   ├── export/                 # Step 1: pseudo-label export
│   │   └── demo1.yaml
│   ├── data/                   # Step 2: dataset / feature dump (later)
│   └── train/                  # Step 3–4: training (later)
├── scripts/                    # CLI entry points (thin wrappers)
│   ├── export_expert_labels.py
│   └── validate_labels.py
├── mahmr/                      # importable Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── schema.py           # tensor names, shapes, dtypes
│   │   └── dynhamr_log.py      # read Dyn-HaMR npz/json (no dyn-hamr import)
│   ├── export/
│   │   └── expert_exporter.py  # Step 1 logic
│   └── utils/
│       ├── io.py
│       └── config.py
├── tests/                      # unit tests (invariance, export smoke)
└── docs/
    └── project_layout.md
```

## `data/` — artifacts (not in git ideally)

```
data/
├── raw/                        # optional: symlinks to benchmark videos
│   └── demo1.mp4 -> ...
├── dynhamr_logs/               # optional symlinks to experiments/outputs/...
│   └── demo1 -> ../../experiments/outputs/logs/.../demo1-all-shot-0-0--1
└── mahmr/
    ├── processed/              # Step 1 output (.pth per sequence)
    │   └── demo1/
    │       ├── expert_label.pth
    │       ├── init_state.pth
    │       ├── valid_mask.pth
    │       ├── meta.json
    │       └── summary.json
    ├── features/               # Step 2: img_feat, kp_2d dumps (later)
    └── manifests/
        └── sequences.jsonl     # registry of all exported sequences
```

## `experiments/` — run outputs

```
experiments/
├── outputs/                    # Dyn-HaMR hydra logs (already exists)
│   └── logs/video-custom/...
└── mahmr/                      # MA-HaMR training (later)
    ├── checkpoints/
    ├── tensorboard/
    └── eval/
```

## Dependency rules

1. **MA-HaMR → Dyn-HaMR**: only filesystem reads (`np.load`, `json.load`, `torch.load` on teacher outputs).
2. **Never** `import` from `Dyn-HaMR/dyn-hamr` inside `mahmr/` (avoids env coupling & accidental edits).
3. **MANO / camera geometry** for training: either duplicate minimal helpers in `mahmr/geometry/` or add Dyn-HaMR as optional path in scripts only — not in core package.
4. **Configs** reference absolute paths under `/extra/SuC/data/` and `/extra/SuC/experiments/`, not paths inside `Dyn-HaMR/`.

## Step roadmap → directory mapping

| Instruction step | Where it lives |
|------------------|----------------|
| 1. Expert pseudo labels | `scripts/export_expert_labels.py`, `data/mahmr/processed/` |
| 2. Feature dump | `scripts/dump_features.py` (later), `data/mahmr/features/` |
| 3. Memory module | `mahmr/models/memory.py`, `tests/test_memory.py` |
| 4. Training | `scripts/train.py`, `configs/train/`, `experiments/mahmr/` |
| 5–6. Eval / ablation | `scripts/eval.py`, `mahmr/eval/` |
