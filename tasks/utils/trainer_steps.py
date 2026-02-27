import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    model.train()
    running_loss = 0.0
    preds_all, labels_all, probs_all, conf_all = [], [], [], []

    for batch in tqdm(loader, desc=desc, leave=True):
        inputs = batch["image"].to(device)
        labels = batch["Label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        softmax_outputs = torch.softmax(outputs, dim=1).detach()
        conf, preds = softmax_outputs.max(dim=1)

        conf_all.extend(conf.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        probs_all.extend(softmax_outputs.cpu().numpy())

    conf_np = np.asarray(conf_all)
    preds_np = np.asarray(preds_all)
    labels_np = np.asarray(labels_all)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (preds_np == labels_np).mean()
    epoch_prec = precision_score(labels_np, preds_np, zero_division=0)
    epoch_rec = recall_score(labels_np, preds_np, zero_division=0)
    probs_np = np.vstack(probs_all)
    if probs_np.shape[1] == 2:
        epoch_auc = roc_auc_score(labels_np, probs_np[:, 1])
    else:
        epoch_auc = roc_auc_score(
            labels_np, probs_np, multi_class="ovr", average="macro"
        )

    metrics = {
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "train_prec": epoch_prec,
        "train_rec": epoch_rec,
        "train_auc": epoch_auc,
    }
    conf_stats = {
        "conf_all": conf_np,
        "preds_all": preds_np,
        "labels_all": labels_np,
    }
    return metrics, conf_stats


def validate_one_epoch(model, loader, criterion, device, desc="Validating"):
    model.eval()
    val_loss = 0.0
    val_preds_all, val_labels_all, val_probs_all, val_conf_all = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            inputs = batch["image"].to(device)
            labels = batch["Label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            softmax_outputs = torch.softmax(outputs, dim=1)
            conf, preds = softmax_outputs.max(dim=1)

            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())
            val_probs_all.extend(softmax_outputs.cpu().numpy())
            val_conf_all.extend(conf.cpu().numpy())

    vpreds_np = np.asarray(val_preds_all)
    vlabels_np = np.asarray(val_labels_all)
    vconf_np = np.asarray(val_conf_all)

    val_epoch_loss = val_loss / len(loader.dataset)
    val_epoch_acc = (vpreds_np == vlabels_np).mean()
    val_epoch_prec = precision_score(vlabels_np, vpreds_np, zero_division=0)
    val_epoch_rec = recall_score(vlabels_np, vpreds_np, zero_division=0)
    val_probs_np = np.vstack(val_probs_all)
    if val_probs_np.shape[1] == 2:
        val_epoch_auc = roc_auc_score(vlabels_np, val_probs_np[:, 1])
    else:
        val_epoch_auc = roc_auc_score(
            vlabels_np, val_probs_np, multi_class="ovr", average="macro"
        )

    metrics = {
        "val_loss": val_epoch_loss,
        "val_acc": val_epoch_acc,
        "val_prec": val_epoch_prec,
        "val_rec": val_epoch_rec,
        "val_auc": val_epoch_auc,
    }
    conf_stats = {
        "val_conf_all": vconf_np,
        "val_preds_all": vpreds_np,
        "val_labels_all": vlabels_np,
    }
    return metrics, conf_stats
