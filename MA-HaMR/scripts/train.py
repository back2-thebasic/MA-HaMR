#!/usr/bin/env python3
"""Train MA-HaMR Step 4 hybrid-supervision loop."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mahmr.data.dataset import MAHaMRSequenceDataset
from mahmr.geometry import MANOJointLayer
from mahmr.loss import MAHaMRLoss, build_window_residual_targets
from mahmr.models.refinement import MAHaMRRefiner
from mahmr.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MA-HaMR")
    parser.add_argument("--config", default="configs/train/dev.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device or cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 0)))

    dataset = MAHaMRSequenceDataset(**cfg["dataset"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 1)),
        shuffle=bool(cfg["train"].get("shuffle", True)),
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=False,
    )

    model_cfg = cfg.get("model", {})
    model = MAHaMRRefiner(**model_cfg).to(device)
    geometry_cfg = cfg.get("geometry", {})
    mano_layer = None
    if geometry_cfg.get("use_mano", False):
        mano_layer = MANOJointLayer(
            model_path=geometry_cfg.get("mano_model_path", "/data/SuC/Dyn-HaMR/_DATA/data/mano"),
            flat_hand_mean=bool(geometry_cfg.get("flat_hand_mean", False)),
        ).to(device)
        mano_layer.eval()
        for param in mano_layer.parameters():
            param.requires_grad_(False)
    criterion = MAHaMRLoss(weights=cfg.get("loss_weights", {}), mano_layer=mano_layer).to(device)
    lr = float(cfg["train"].get("lr", 1e-4))
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    resume_from = cfg["train"].get("resume_from")
    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if cfg["train"].get("resume_optimizer", False) and "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        print(f"resumed_from={resume_from} step={ckpt.get('step')}")

    output_dir = cfg["train"].get("output_dir", "/extra/SuC/experiments/mahmr/dev")
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    max_steps = args.max_steps or cfg["train"].get("max_steps")
    log_every = int(cfg["train"].get("log_every", 1))
    plot_every = int(cfg["train"].get("plot_every", 0))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    epochs = int(cfg["train"].get("epochs", 1))

    scheduler = None
    if cfg["train"].get("lr_schedule") == "cosine" and max_steps:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=int(max_steps), eta_min=float(cfg["train"].get("lr_min", lr * 0.05))
        )

    step = 0
    metric_rows: List[Dict[str, float]] = []
    _init_metrics_csv(metrics_path)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            targets = build_window_residual_targets(batch, device=device)

            output = model(
                batch["features"],
                memory_values=targets["packed_residual"],
                write_mask=targets["valid_mask"],
                reset_memory=True,
                return_attention=False,
            )
            losses = criterion(output, batch, targets)

            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            step += 1
            row = _collect_metrics(epoch, step, losses, output, targets)
            metric_rows.append(row)
            _append_metrics_csv(metrics_path, row)
            if step % log_every == 0:
                print(_format_log(row))
            if plot_every > 0 and step % plot_every == 0:
                write_training_plots(metric_rows, plot_dir)
            if max_steps is not None and step >= int(max_steps):
                write_training_plots(metric_rows, plot_dir)
                save_checkpoint(output_dir, model, optim, cfg, step)
                return

    write_training_plots(metric_rows, plot_dir)
    save_checkpoint(output_dir, model, optim, cfg, step)


def save_checkpoint(output_dir: str, model: torch.nn.Module, optim: torch.optim.Optimizer, cfg: Dict[str, Any], step: int) -> None:
    path = os.path.join(output_dir, "last.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "config": cfg,
            "step": step,
        },
        path,
    )
    print(f"saved_checkpoint={path}")


def _collect_metrics(
    epoch: int,
    step: int,
    losses: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    row: Dict[str, float] = {"epoch": float(epoch), "step": float(step)}
    for key in (
        "loss",
        "loss_distill",
        "loss_proj",
        "loss_proj_huber",
        "loss_bone",
        "loss_consistency",
        "loss_scale",
        "loss_prior",
    ):
        row[key] = float(losses[key].detach().cpu()) if key in losses else 0.0
    row["writes"] = float(targets["valid_mask"].sum().detach().cpu())
    row["memory_size"] = float(output["memory_size"])
    row["pred_abs_mean"] = float(output["packed_residual"].detach().abs().mean().cpu())
    row["scale_mean"] = float(output["world_scale_refined"].detach().mean().cpu())
    row["scale_min"] = float(output["world_scale_refined"].detach().min().cpu())
    return row


def _format_log(row: Dict[str, float]) -> str:
    parts = [f"epoch={int(row['epoch'])}", f"step={int(row['step'])}"]
    for key in ("loss", "loss_distill", "loss_proj", "loss_proj_huber", "loss_bone", "loss_consistency", "loss_scale"):
        parts.append(f"{key}={row[key]:.6f}")
    parts.append(f"writes={int(row['writes'])}")
    parts.append(f"memory_size={int(row['memory_size'])}")
    return " ".join(parts)


def _init_metrics_csv(path: str) -> None:
    fields = _metric_fields()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def _append_metrics_csv(path: str, row: Dict[str, float]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_metric_fields())
        writer.writerow(row)


def _metric_fields() -> List[str]:
    return [
        "epoch",
        "step",
        "loss",
        "loss_distill",
        "loss_proj",
        "loss_proj_huber",
        "loss_bone",
        "loss_consistency",
        "loss_scale",
        "loss_prior",
        "writes",
        "memory_size",
        "pred_abs_mean",
        "scale_mean",
        "scale_min",
    ]


def write_training_plots(rows: List[Dict[str, float]], plot_dir: str) -> None:
    if not rows:
        return
    os.makedirs(plot_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(plot_dir), ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        print(f"plot_warning={exc}")
        return

    steps = [r["step"] for r in rows]
    smooth = max(1, min(25, len(rows) // 20))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_series(axes[0, 0], steps, rows, ["loss", "loss_distill"], smooth)
    axes[0, 0].set_title("Total / Distill Loss")
    _plot_series(axes[0, 1], steps, rows, ["loss_proj", "loss_consistency"], smooth)
    axes[0, 1].set_title("Proxy Projection / Memory Consistency")
    _plot_series(axes[1, 0], steps, rows, ["loss_bone", "loss_scale"], smooth)
    axes[1, 0].set_title("Regularizers")
    _plot_series(axes[1, 1], steps, rows, ["pred_abs_mean", "scale_mean"], smooth)
    axes[1, 1].set_title("Prediction Magnitudes")
    for ax in axes.reshape(-1):
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_curves.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_series(ax, steps, rows, ["writes", "memory_size"], smooth=1)
    ax.set_title("Memory Writes / Size")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "memory_curves.png"), dpi=160)
    plt.close(fig)


def _plot_series(ax: Any, steps: List[float], rows: List[Dict[str, float]], keys: List[str], smooth: int) -> None:
    for key in keys:
        values = [r[key] for r in rows]
        ax.plot(steps, values, alpha=0.25, linewidth=1.0, label=f"{key} raw")
        if smooth > 1:
            ax.plot(steps, _moving_average(values, smooth), linewidth=2.0, label=f"{key} ma{smooth}")


def _moving_average(values: List[float], window: int) -> List[float]:
    out = []
    acc = 0.0
    for i, value in enumerate(values):
        acc += value
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


if __name__ == "__main__":
    main()
