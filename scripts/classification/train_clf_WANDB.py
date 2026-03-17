"""
Classification training script — logs to Weights & Biases.
See train_clf.py for the MLflow variant.
"""

import argparse
import os
import sys
from pathlib import Path

import wandb
import torch

os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
folder_root = Path(__file__).resolve().parents[1]
if str(folder_root) not in sys.path:
    sys.path.insert(0, str(folder_root))

from src.models.factory import build_model
from train_utils import (
    load_configs, create_dataframe, load_records,
    build_splits, build_dataloaders, build_optimizer_and_scheduler,
    run_training_loop, resolve_run_name,
    save_checkpoint, save_metrics_csv, fold_summary_entry, summarize_folds,
)


def main(args):
    setup, transforms_cfg = load_configs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg  = setup["data"]
    model_cfg = setup["model"]
    split_cfg = setup["split"]
    train_cfg = setup["training"]
    dl_cfg    = setup["dataloader"]

    csv_path = create_dataframe(
        path_to_dataset=data_cfg["path_to_dataset"],
        scans_excluded=data_cfg.get("scans_excluded") or [],
        centers_excluded=data_cfg.get("centers_excluded"),
        data_cfg=data_cfg,
    )
    records = load_records(csv_path)
    split_indices, fold_ids, n_splits, cv_enabled = build_splits(
        records, split_cfg, cv_fold=getattr(args, "cv_fold", None)
    )

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = save_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    base_run_name = resolve_run_name(args, model_cfg, save_dir)

    fold_summaries = []
    for local_idx, (train_idx, val_idx) in enumerate(split_indices):
        fold_id = fold_ids[local_idx]
        train_files = [records[i] for i in train_idx]
        val_files   = [records[i] for i in val_idx]
        print(
            f"[fold {fold_id}/{n_splits}] "
            f"train={len(train_files)} | val={len(val_files)} | "
            f"pos={sum(d['Label'] == 1 for d in train_files)}"
        )

        train_loader, val_loader = build_dataloaders(train_files, val_files, transforms_cfg, dl_cfg)
        model = build_model(model_cfg).to(device)
        criterion, optimizer, scheduler = build_optimizer_and_scheduler(model, train_cfg, args.epochs)

        run_name = f"{base_run_name}_fold{fold_id:02d}" if cv_enabled else base_run_name
        run = wandb.init(
            project=args.experiment_name or "MAE-classifier",
            name=run_name,
            group=base_run_name if cv_enabled else None,
            reinit=True,
            config={
                "setup_config": args.setup_config,
                "nb_MIPs": data_cfg.get("nb_MIPs"),
                "batch_size": dl_cfg["batch_size"],
                "lr": train_cfg["optimizer"]["lr"],
                "model": model_cfg["name"],
                "cv_enabled": cv_enabled,
                "cv_fold": fold_id,
                "cv_n_splits": n_splits,
                **{f"model_args.{k}": v for k, v in model_cfg.get("model_args", {}).items()},
            },
        )

        def log_fn(epoch, train_metrics, val_metrics, lr):
            wandb.log({
                **train_metrics, **val_metrics,
                "fold": fold_id, "epoch": epoch + 1, "lr": lr,
                "train/loss": train_metrics["train_loss"],
                "val/loss":   val_metrics["val_loss"],
                "train/acc":  train_metrics["train_acc"],
                "val/acc":    val_metrics["val_acc"],
                "train/auc":  train_metrics["train_auc"],
                "val/auc":    val_metrics["val_auc"],
                "train/prec": train_metrics["train_prec"],
                "val/prec":   val_metrics["val_prec"],
                "train/rec":  train_metrics["train_rec"],
                "val/rec":    val_metrics["val_rec"],
            }, step=epoch)

        history = run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.epochs, fold_id, log_fn=log_fn,
        )

        metrics_path = save_metrics_csv(history, metrics_dir, run_name, fold_id)
        ckpt_path = save_checkpoint(
            model, optimizer, scheduler, save_dir,
            run_name, args.epochs, fold_id, n_splits, history,
        )
        wandb.save(str(metrics_path))
        wandb.save(str(ckpt_path))
        run.finish()

        fold_summaries.append(fold_summary_entry(fold_id, train_files, val_files, history))

    summarize_folds(fold_summaries, metrics_dir, base_run_name, cv_enabled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-config", default="../configs/mip_setup.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--cv-fold", type=int, default=None,
                        help="Run a single fold (1-based). Requires cross_validation.enabled=true.")
    args = parser.parse_args()
    main(args)
