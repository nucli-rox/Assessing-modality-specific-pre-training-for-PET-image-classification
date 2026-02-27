import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
import yaml

from nucli_train.val.evaluators import EVALUATORS_REGISTRY


class ImageFolderAsDict(Dataset):
    """
    Wraps a torchvision dataset (e.g., ImageFolder or Subset) so the training
    loop can always access batch["data"] just like the MIP dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        return {"data": image, "label": target}


def _select_subset_indices(base_dataset, subset_size, seed, per_class, class_ids=None):
    """
    Pick a deterministic subset of indices from an ImageFolder dataset.

    If per_class is True, this tries to draw an equal number of samples per class.
    If per_class is False, this draws a random subset across all images.
    """

    num_samples = len(base_dataset)
    if subset_size is None or subset_size <= 0 or subset_size >= num_samples:
        return list(range(num_samples))

    rng = random.Random(seed)
    if not per_class:
        indices = range(num_samples)
        if class_ids is not None:
            indices = [
                i
                for i, target in enumerate(base_dataset.targets)
                if target in class_ids
            ]
            if subset_size is None or subset_size <= 0 or subset_size >= len(indices):
                return indices
        return rng.sample(list(indices), subset_size)

    # Build a per-class index list from ImageFolder targets
    class_to_indices = {}
    for idx, target in enumerate(base_dataset.targets):
        if class_ids is not None and target not in class_ids:
            continue
        class_to_indices.setdefault(target, []).append(idx)

    num_classes = len(class_to_indices)
    per_class_limit = max(1, subset_size // num_classes)

    selected = []
    for _, indices in class_to_indices.items():
        rng.shuffle(indices)
        selected.extend(indices[:per_class_limit])

    # If we are short (e.g., due to rounding), fill from the remaining pool.
    if len(selected) < subset_size:
        allowed = (
            set(range(num_samples))
            if class_ids is None
            else set(i for i, t in enumerate(base_dataset.targets) if t in class_ids)
        )
        remaining = list(allowed - set(selected))
        rng.shuffle(remaining)
        selected.extend(remaining[: (subset_size - len(selected))])

    return selected[:subset_size]


def build_transform(is_train, cfg):
    """
    Create ImageNet-style transforms matching ConvNeXt / MAE defaults.
    This is adapted from the official ConvNeXt code with extra comments.
    """

    resize_im = cfg["input_size"] > 32
    imagenet_default_mean_and_std = cfg.get("imagenet_default_mean_and_std", True)
    mean = (
        IMAGENET_DEFAULT_MEAN
        if imagenet_default_mean_and_std
        else IMAGENET_INCEPTION_MEAN
    )
    std = (
        IMAGENET_DEFAULT_STD
        if imagenet_default_mean_and_std
        else IMAGENET_INCEPTION_STD
    )

    if is_train:
        # Training augmentations:
        # - random resized crop
        # - color jitter / auto-augment (if enabled)
        # - random erase (if enabled)
        transform = create_transform(
            input_size=cfg["input_size"],
            is_training=True,
            color_jitter=cfg.get("color_jitter"),
            auto_augment=cfg.get("aa"),
            interpolation=cfg.get("train_interpolation", "bicubic"),
            re_prob=cfg.get("reprob", 0.0),
            re_mode=cfg.get("remode", "const"),
            re_count=cfg.get("recount", 1),
            mean=mean,
            std=std,
        )
        transform.transforms.append(ShiftMinToZero(fixed_shift=1.860))
        if not resize_im:
            # For small images (e.g., CIFAR), replace the first transform
            # with a simpler random crop + padding.
            transform.transforms[0] = transforms.RandomCrop(
                cfg["input_size"], padding=4
            )
        return transform

    # Validation / inference transforms:
    # - resize (optionally warping at 384+)
    # - center crop
    # - normalize
    t = []
    if resize_im:
        if cfg["input_size"] >= 384:
            # Warp to a fixed square size for large inputs
            t.append(
                transforms.Resize(
                    (cfg["input_size"], cfg["input_size"]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            )
        else:
            # Standard ImageNet resize + center crop
            crop_pct = cfg.get("crop_pct")
            if crop_pct is None:
                crop_pct = 224 / 256
            size = int(cfg["input_size"] / crop_pct)
            t.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
            t.append(transforms.CenterCrop(cfg["input_size"]))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    t.append(ShiftMinToZero(fixed_shift=1.860))
    return transforms.Compose(t)


def build_imagenet_dataset(is_train, cfg):
    """
    Build an ImageNet ImageFolder dataset, optionally limited to a subset.
    """

    split = "train" if is_train else "val"
    root = Path(cfg["data_root"]) / split
    base_dataset = datasets.ImageFolder(root, transform=build_transform(is_train, cfg))

    subset_size_key = "train_subset_size" if is_train else "val_subset_size"
    subset_size = cfg.get(subset_size_key)
    per_class = cfg.get("subset_per_class", True)
    seed = cfg.get("subset_seed", 42)
    subset_classes = cfg.get("subset_classes")
    subset_num_classes = cfg.get("subset_num_classes")
    class_seed = cfg.get("subset_class_seed", seed)

    class_ids = None
    if subset_classes:
        class_ids = [base_dataset.class_to_idx[c] for c in subset_classes]
    elif subset_num_classes:
        rng = random.Random(class_seed)
        all_classes = list(base_dataset.class_to_idx.values())
        subset_num_classes = min(subset_num_classes, len(all_classes))
        class_ids = rng.sample(all_classes, subset_num_classes)

    indices = _select_subset_indices(
        base_dataset,
        subset_size=subset_size,
        seed=seed,
        per_class=per_class,
        class_ids=class_ids,
    )
    subset = Subset(base_dataset, indices)
    return ImageFolderAsDict(subset)


class ShiftMinToZero:
    """
    Post-normalization shift.
    If fixed_shift is set, uses a constant shift (invertible in evaluator).
    Otherwise falls back to per-image min shift.
    """

    def __init__(self, per_channel=True, fixed_shift=None):
        self.per_channel = per_channel
        self.fixed_shift = fixed_shift

    def __call__(self, x):
        # x is CxHxW after Normalize
        if self.fixed_shift is not None:
            return x + float(self.fixed_shift)

        if self.per_channel:
            return x - x.amin(dim=(1, 2), keepdim=True)
        return x - x.amin()


def build_imagenet_data(cfg):
    """
    Returns the train_data and val_loaders dicts expected by nucli_train.Trainer.
    """

    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

    split_from_train = bool(cfg.get("split_from_train", False))
    val_split = float(cfg.get("val_split", 0.0) or 0.0)
    split_seed = int(cfg.get("split_seed", 42))

    if split_from_train and val_split > 0.0:
        # Build from the train folder only, then split into train/val.
        base_train = datasets.ImageFolder(
            Path(cfg["data_root"]) / "train",
            transform=build_transform(True, cfg),
        )
        subset_size = cfg.get("train_subset_size")
        per_class = cfg.get("subset_per_class", True)
        seed = cfg.get("subset_seed", 42)
        subset_classes = cfg.get("subset_classes")
        subset_num_classes = cfg.get("subset_num_classes")
        class_seed = cfg.get("subset_class_seed", seed)

        class_ids = None
        if subset_classes:
            class_ids = [base_train.class_to_idx[c] for c in subset_classes]
        elif subset_num_classes:
            rng = random.Random(class_seed)
            all_classes = list(base_train.class_to_idx.values())
            subset_num_classes = min(subset_num_classes, len(all_classes))
            class_ids = rng.sample(all_classes, subset_num_classes)

        indices = _select_subset_indices(
            base_train,
            subset_size=subset_size,
            seed=seed,
            per_class=per_class,
            class_ids=class_ids,
        )
        rng = random.Random(split_seed)
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_split))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]

        train_subset = Subset(base_train, train_idx)
        val_subset = Subset(
            datasets.ImageFolder(
                Path(cfg["data_root"]) / "train",
                transform=build_transform(False, cfg),
            ),
            val_idx,
        )
        train_dataset = ImageFolderAsDict(train_subset)
        val_dataset = ImageFolderAsDict(val_subset)
    else:
        train_dataset = build_imagenet_dataset(is_train=True, cfg=cfg)
        val_dataset = None
    train_data = {
        "dataset": train_dataset,
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": cfg["train"]["num_workers"],
    }
    if cfg.get("val") and cfg["val"].get("batch_size"):
        if val_dataset is None:
            val_dataset = build_imagenet_dataset(is_train=False, cfg=cfg)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["val"]["batch_size"],
            shuffle=False,
            num_workers=cfg["val"]["num_workers"],
            drop_last=False,
        )
        val_loaders = {
            "imagenet_val": {
                "interval": cfg["val"].get("global_eval_interval", 1),
                "loader": val_loader,
                "evaluators": [],
            }
        }

        evaluators = [
            EVALUATORS_REGISTRY.get("MAE-evaluator-imagenet")(
                f"results/imagenet",
                imagenet_default_mean_and_std=cfg.get(
                    "imagenet_default_mean_and_std", True
                ),
                fixed_post_norm_shift=cfg.get("fixed_post_norm_shift", 0.0),
            )
        ]
        val_loaders = {
            "IN_val": {
                "interval": cfg["val"].get("global_eval_interval", 1),
                "loader": val_loader,
                "evaluators": evaluators,
            }
        }

    return train_data, val_loaders
