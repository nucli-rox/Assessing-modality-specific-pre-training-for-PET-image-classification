"""
Shared utilities for classification training scripts.
Handles data discovery, config loading, dataset splitting,
data loading, optimizer/scheduler setup, and training loop.
"""

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ToTensord,
)
import monai.transforms.compose as compose_mod
import monai.transforms.transform as transform_mod
import monai.utils as monai_utils
import monai.utils.misc as monai_misc
from monai.transforms.transform import Randomizable

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
folder_root = Path(__file__).resolve().parents[1]
if str(folder_root) not in sys.path:
    sys.path.insert(0, str(folder_root))

from utils.trainer_steps import train_one_epoch, validate_one_epoch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = ("*.nii.gz", "*.nii")
FIXED_MAX = int(np.iinfo(np.uint32).max)

# ---------------------------------------------------------------------------
# Monai RNG overflow fix
# Monai's default MAX_SEED can overflow on some systems; cap it to uint32 max.
# ---------------------------------------------------------------------------

monai_utils.MAX_SEED = FIXED_MAX
monai_misc.MAX_SEED = FIXED_MAX
compose_mod.MAX_SEED = FIXED_MAX
transform_mod.MAX_SEED = FIXED_MAX


def _safe_set_random_state(self, seed=None, state=None):
    if seed is not None:
        if isinstance(seed, np.generic):
            seed = seed.item()
        self.R = np.random.RandomState(int(seed) % FIXED_MAX)
        return self
    if state is not None:
        if not isinstance(state, np.random.RandomState):
            raise TypeError(
                f"Expected np.random.RandomState, got {type(state).__name__}"
            )
        self.R = state
        return self
    self.R = np.random.RandomState()
    return self


Randomizable.set_random_state = _safe_set_random_state

# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------


def _strip_known_suffixes(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return name.split(".")[0]
    return path.stem


def _find_pet_files(root_dir: Path, mip_subdir=None) -> list:
    files = []
    subdirs = [mip_subdir] if mip_subdir else ["*_MIPs", None]
    for subdir in subdirs:
        for ext in SUPPORTED_EXTS:
            pattern = f"*/fdg/pet/{subdir}/{ext}" if subdir else f"*/fdg/pet/{ext}"
            files.extend(root_dir.glob(pattern))
    return sorted(set(files))


def _extract_center(file_path: Path) -> str:
    parts = file_path.parts
    if "fdg" in parts:
        fdg_idx = parts.index("fdg")
        if fdg_idx > 0:
            return parts[fdg_idx - 1]
    return file_path.parents[2].name if len(file_path.parents) > 2 else "unknown"


def _resolve_label(
    file_path: Path, patient_id: str, center: str, data_cfg: dict
) -> int:
    label_from = data_cfg.get("label_from", "patient_id_suffix")
    label_map = data_cfg.get("label_map", {"N": 0, "AN": 1})

    if label_from == "patient_id_suffix":
        token = patient_id.split("_")[-1]
    elif label_from == "filename_suffix":
        token = _strip_known_suffixes(file_path).split("_")[-1]
    elif label_from == "parent_dir":
        token = file_path.parent.name
    elif label_from == "center":
        token = center
    else:
        raise ValueError(f"Unsupported label_from: '{label_from}'")

    if token in label_map:
        return int(label_map[token])
    if str(token) in label_map:
        return int(label_map[str(token)])
    raise ValueError(
        f"Could not map label token '{token}'. Set data.label_from/label_map in your config."
    )


def create_dataframe(
    path_to_dataset, scans_excluded=None, centers_excluded=None, data_cfg=None
):
    """Scan the dataset directory, build a patient CSV, and cache it to disk."""
    root_dir = Path(path_to_dataset)
    scans_excluded = set(scans_excluded or [])
    centers_excluded = set(centers_excluded or [])
    data_cfg = data_cfg or {}

    save_dir = root_dir / "dataframes"
    save_dir.mkdir(parents=True, exist_ok=True)
    excluded_tag = "all" if not centers_excluded else "_".join(sorted(centers_excluded))
    save_path = (
        save_dir
        / f"df_{data_cfg.get('dataset_tag', 'dataset')}_without_{excluded_tag}.csv"
    )

    if not save_path.exists():
        nb_mips = data_cfg.get("nb_MIPs")
        mip_subdir = data_cfg.get("mip_subdir") or (
            f"{nb_mips}_MIPs" if nb_mips else None
        )
        files = _find_pet_files(root_dir, mip_subdir=mip_subdir)
        if not files:
            raise ValueError(
                f"No PET files found under {root_dir}/<center>/fdg/pet/. "
                "Set data.mip_subdir in your config if MIPs are in a named subfolder."
            )
        rows = []
        for file_path in files:
            center = _extract_center(file_path)
            if center in centers_excluded:
                continue
            patient_id = _strip_known_suffixes(file_path)
            if patient_id in scans_excluded or file_path.name in scans_excluded:
                continue
            rows.append(
                {
                    "PatientID": patient_id,
                    "center": center,
                    "Label": _resolve_label(file_path, patient_id, center, data_cfg),
                    "image": str(file_path),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(
                "No eligible files found after applying exclusion filters."
            )
        df.to_csv(save_path, index=False)
        print(f"Dataset saved ({len(df)} patients): {save_path}")
    return save_path


def load_records(csv_path) -> list:
    """Load the patient CSV into a list of dicts ready for MONAI Dataset."""
    df = pd.read_csv(csv_path)
    records = [
        {
            "PatientID": row["PatientID"],
            "Label": row["Label"],
            "image": row["image"],
            "center": row.get("center", "unknown"),
        }
        for row in df.to_dict("records")
    ]
    if not records:
        raise ValueError("No records available for training.")
    return records


# ---------------------------------------------------------------------------
# Transform utilities
# ---------------------------------------------------------------------------


def instantiate_class(spec: dict):
    """Instantiate a transform from a config dict with a `_target_` key."""
    spec = dict(spec)
    target = spec.pop("_target_")
    module_name, class_name = target.rsplit(".", 1)
    return getattr(import_module(module_name), class_name)(**spec)


def build_chain(spec_list: list) -> Compose:
    return Compose([instantiate_class(s) for s in spec_list])


def build_transforms(transforms_cfg: dict):
    """Return (train_transform, val_transform) from the transforms config."""
    deterministic = Compose(
        [
            LoadImaged(keys=("image",)),
            EnsureChannelFirstd(keys=("image",)),
            EnsureTyped(keys=("image", "Label")),
        ]
    )
    train_transform = Compose([deterministic, build_chain(transforms_cfg["random"])])
    val_transform = Compose([deterministic, ToTensord(keys=("image", "Label"))])
    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _resolve_existing_path(path_value, setup_path=None) -> Path:
    """Resolve a relative or absolute path against several candidate roots."""
    p = Path(str(path_value)).expanduser()
    if p.is_absolute() and p.exists():
        return p
    anchors = []
    if setup_path is not None:
        anchors.append(setup_path.parent)
    anchors.extend([Path.cwd(), repo_root, Path(__file__).resolve().parent])
    for base in anchors:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    if not p.is_absolute():
        matches = list((repo_root / "configs").rglob(p.name))
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(f"Could not resolve path: {path_value}")


def load_configs(args):
    """Load setup YAML and the transforms YAML it references."""
    setup_path = _resolve_existing_path(args.setup_config)
    with open(setup_path) as f:
        setup = yaml.safe_load(f)
    transforms_path = _resolve_existing_path(
        setup["transforms_config"], setup_path=setup_path
    )
    with open(transforms_path) as f:
        transforms_cfg = yaml.safe_load(f)
    return setup, transforms_cfg


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def build_splits(records: list, split_cfg: dict, cv_fold=None):
    """Return (split_indices, fold_ids, n_splits, cv_enabled)."""
    strat_labels = [f"{r['Label']}_{r['center']}" for r in records]
    cv_cfg = split_cfg.get("cross_validation", {})
    cv_enabled = bool(cv_cfg.get("enabled", False))
    n_splits = int(cv_cfg.get("n_splits", 10)) if cv_enabled else 1

    if cv_enabled:
        if n_splits < 2:
            raise ValueError("cross_validation.n_splits must be >= 2")
        y = (
            strat_labels
            if split_cfg.get("stratify", True)
            else [r["Label"] for r in records]
        )
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=split_cfg["seed"]
        )
        split_indices = list(splitter.split(np.arange(len(records)), y))
        if cv_fold is not None:
            if cv_fold < 1 or cv_fold > n_splits:
                raise ValueError(f"--cv-fold must be in [1, {n_splits}]")
            split_indices = [split_indices[cv_fold - 1]]
            fold_ids = [cv_fold]
        else:
            fold_ids = list(range(1, n_splits + 1))
    else:
        all_indices = np.arange(len(records))
        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=split_cfg["val_ratio"],
            random_state=split_cfg["seed"],
            stratify=strat_labels if split_cfg["stratify"] else None,
        )
        split_indices = [(np.array(train_idx), np.array(val_idx))]
        fold_ids = [1]

    return split_indices, fold_ids, n_splits, cv_enabled


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_dataloaders(train_files, val_files, transforms_cfg, dl_cfg):
    """Build train and validation DataLoaders, with optional weighted sampling."""
    train_transform, val_transform = build_transforms(transforms_cfg)
    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_transform)

    batch_size = dl_cfg["batch_size"]
    num_workers = dl_cfg["num_workers"]
    pin_memory = dl_cfg["pin_memory"]

    sampler = None
    if dl_cfg.get("weighted_sampler", {}).get("enabled", False):
        labels = np.array([d["Label"] for d in train_files])
        class_counts = np.bincount(labels)
        weights = (1.0 / class_counts)[labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=dl_cfg["weighted_sampler"].get("replacement", True),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"[loader] batch={batch_size} | sampler={'weighted' if sampler else 'shuffle'} | workers={num_workers}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------


def build_optimizer_and_scheduler(model, train_cfg: dict, epochs: int):
    """Build criterion, Adam optimizer, and SequentialLR scheduler from config."""
    criterion = instantiate_class(train_cfg["criterion"])

    opt_cfg = train_cfg["optimizer"]
    if opt_cfg["type"] != "adam":
        raise ValueError(f"Unsupported optimizer type: '{opt_cfg['type']}'")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )

    sched_cfg = train_cfg["scheduler"]
    if sched_cfg["type"] != "sequential":
        raise ValueError(f"Unsupported scheduler type: '{sched_cfg['type']}'")

    cold_cfg = sched_cfg["cold"]
    if cold_cfg["type"] == "constant":
        cold = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=cold_cfg["factor"], total_iters=cold_cfg["total_iters"]
        )
    elif cold_cfg["type"] == "linear":
        cold = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cold_cfg["factor"],
            total_iters=cold_cfg["total_iters"],
        )
    else:
        raise ValueError(f"Unsupported cold scheduler: '{cold_cfg['type']}'")

    main_cfg = sched_cfg["main"]
    if main_cfg["type"] == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=main_cfg["step_size"], gamma=main_cfg["gamma"]
        )
    elif main_cfg["type"] == "cosineannealing":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - cold_cfg["total_iters"]),
            eta_min=float(main_cfg["min_lr"]),
        )
    else:
        raise ValueError(f"Unsupported main scheduler: '{main_cfg['type']}'")

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[cold, main], milestones=[cold_cfg["total_iters"]]
    )
    return criterion, optimizer, scheduler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    fold_id: int,
    log_fn=None,
) -> dict:
    """Run one full training loop (all epochs) for a single fold.

    Args:
        log_fn: optional callable ``log_fn(epoch, train_metrics, val_metrics, lr)``
                called after each epoch — use this to hook in MLflow or W&B logging.

    Returns:
        history: dict of lists, one entry per epoch, e.g.
                 {"train_loss": [...], "val_auc": [...], ...}
    """
    metric_keys = ["loss", "acc", "prec", "rec", "auc"]
    history = {f"train_{k}": [] for k in metric_keys}
    history.update({f"val_{k}": [] for k in metric_keys})

    device = next(model.parameters()).device

    for epoch in range(epochs):
        print(f"\nFold {fold_id} | Epoch {epoch + 1}/{epochs}")
        train_metrics, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics, _ = validate_one_epoch(model, val_loader, criterion, device)

        for k in metric_keys:
            history[f"train_{k}"].append(train_metrics.get(f"train_{k}"))
            history[f"val_{k}"].append(val_metrics.get(f"val_{k}"))

        scheduler.step()

        if log_fn is not None:
            log_fn(epoch, train_metrics, val_metrics, optimizer.param_groups[0]["lr"])

    return history


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------


def resolve_run_name(args, model_cfg: dict, save_dir: Path) -> str:
    fallback = save_dir.name or model_cfg["name"]
    cli_name = getattr(args, "run_name", None)
    return (
        fallback
        if cli_name in (None, "", "LPsmall", "give_this_a_better_name!")
        else cli_name
    )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    run_name: str,
    epochs: int,
    fold_id: int,
    n_splits: int,
    history: dict,
) -> Path:
    ckpt_path = save_dir / f"{run_name}.pth"
    torch.save(
        {
            "epoch": epochs,
            "fold": fold_id,
            "n_splits": n_splits,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def save_metrics_csv(
    history: dict, metrics_dir: Path, run_name: str, fold_id: int
) -> Path:
    path = metrics_dir / f"{run_name}_metrics.csv"
    pd.DataFrame({"fold": fold_id, **history}).to_csv(path, index=False)
    print(f"Saved metrics: {path}")
    return path


def fold_summary_entry(
    fold_id: int, train_files: list, val_files: list, history: dict
) -> dict:
    val_auc = pd.to_numeric(pd.Series(history["val_auc"]), errors="coerce")
    return {
        "fold": fold_id,
        "train_size": len(train_files),
        "val_size": len(val_files),
        "val_loss_last": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc_last": history["val_acc"][-1] if history["val_acc"] else None,
        "val_auc_last": history["val_auc"][-1] if history["val_auc"] else None,
        "val_auc_best": float(val_auc.max()) if not val_auc.dropna().empty else np.nan,
    }


def summarize_folds(
    fold_summaries: list, metrics_dir: Path, base_run_name: str, cv_enabled: bool
):
    summary_df = pd.DataFrame(fold_summaries)
    name = (
        f"{base_run_name}_cv_summary.csv"
        if cv_enabled
        else f"{base_run_name}_summary.csv"
    )
    path = metrics_dir / name
    summary_df.to_csv(path, index=False)
    print(f"Saved summary: {path}")
    if cv_enabled and len(summary_df) > 1:
        agg_cols = ["val_loss_last", "val_acc_last", "val_auc_last", "val_auc_best"]
        print("[cv] means:", summary_df[agg_cols].mean(numeric_only=True).to_dict())
        print("[cv] stds:", summary_df[agg_cols].std(numeric_only=True).to_dict())
